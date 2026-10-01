"""Train scDRMAE on a paired RNA + ATAC dataset stored as two .h5ad files.

Reimplements the training loop from main_clustering.ipynb (5-fold CV,
masked-reconstruction + KL loss, KMeans clustering, NMI/ARI/AMI reporting)
for data that is already in AnnData/h5ad form instead of the CSV/mtx format
that read_dataset1() in the notebook expects.

After CV, writes {method}_{dataset}.npz (see export.py / EXPORT_NPZ_GUIDE.md):
the model of one fold encodes ALL cells and KMeans clusters them. Disable
with --no-export.
"""

import argparse
import os
import random

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import KFold

from datasets import apply_noise
from evaluate import evaluate
from export import save_embedding
from model import scDRMAE
from util import AverageMeter

LABEL_KEY_CANDIDATES = ["cell_type", "Cluster", "cluster", "CellType", "label", "Group", "trueType_y"]

# Same per-dataset (epochs, lr) switch as args cell in main_clustering.ipynb.
DATASET_HPARAMS = {
    "InHouse": {"epochs": 20, "learning_rate": 0.001},
    "human cell line mixture": {"epochs": 50, "learning_rate": 0.001},
    "10x": {"epochs": 20, "learning_rate": 0.002},
}
DEFAULT_HPARAMS = {"epochs": 100, "learning_rate": 0.002}  # the notebook's "else" branch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def find_label_key(adata, label_key):
    if label_key is not None:
        if label_key not in adata.obs:
            raise KeyError(f"'{label_key}' not in obs columns: {list(adata.obs.columns)}")
        return label_key
    for key in LABEL_KEY_CANDIDATES:
        if key in adata.obs:
            return key
    raise KeyError(
        "Could not auto-detect a ground-truth label column in .obs. "
        f"Available columns: {list(adata.obs.columns)}. Pass --label-key explicitly."
    )


def to_dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def TFIDF(count_mat):
    count_mat = count_mat.T
    divide_title = np.tile(np.sum(count_mat, axis=0), (count_mat.shape[0], 1))
    nfreqs = 1.0 * count_mat / divide_title
    multiply_title = np.tile(
        np.log(1 + 1.0 * count_mat.shape[1] / np.sum(count_mat, axis=1)).reshape(-1, 1),
        (1, count_mat.shape[1]),
    )
    return sp.csr_matrix(np.multiply(nfreqs, multiply_title)).T


def preprocess_modality(adata, n_top_genes, datatype, apply_tfidf):
    """Mirrors cells 12-16 of main_clustering.ipynb for one modality."""
    adata = adata[:, (adata.X > 0).sum(0) >= adata.shape[0] * 0.01].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(n_top_genes, adata.shape[1]))
    adata = adata[:, adata.var["highly_variable"]].copy()

    # NOTE: the published notebook calls normalize(..., datatype='AAC') for ATAC
    # (typo for 'ATAC'), so the TF-IDF branch below never actually runs there -
    # it silently falls back to plain scaling, same as the RNA branch.
    # Pass --apply-atac-tfidf to opt into the (probably intended) TF-IDF path
    # instead of literally reproducing that typo.
    if datatype == "ATAC" and apply_tfidf:
        adata.X = TFIDF(adata.X.copy())
        adata.X = adata.X / np.max(adata.X)
    sc.pp.scale(adata)
    return adata


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, rna, atac, labels):
        self.rna = rna
        self.atac = atac
        self.labels = labels

    def __len__(self):
        return len(self.rna)

    def __getitem__(self, idx):
        return self.rna[idx], self.atac[idx], self.labels[idx]


def inference(net, loader, device):
    net.eval()
    feats, labs = [], []
    with torch.no_grad():
        for x, x1, y in loader:
            feats.extend(net.feature(x.float().to(device), x1.float().to(device)).cpu().numpy())
            labs.extend(y.numpy())
    return np.array(feats), np.array(labs)


def encode_all(net, x_rna, x_atac, device, batch_size):
    """Encode every cell in original row order, in batches of `batch_size`
    (same batching as data_loader_all in the notebook).

    Slices the arrays directly instead of iterating a DataLoader: a DataLoader
    draws from the global torch RNG even with shuffle=False, which would shift
    the random stream of the folds that follow.
    """
    net.eval()
    feats = []
    with torch.no_grad():
        for s in range(0, len(x_rna), batch_size):
            x = torch.from_numpy(x_rna[s:s + batch_size]).float().to(device)
            x1 = torch.from_numpy(x_atac[s:s + batch_size]).float().to(device)
            feats.append(net.feature(x, x1).cpu().numpy())
    return np.concatenate(feats)


def pick_export_fold(aris, export_fold):
    if export_fold is not None:
        return export_fold, "user"
    aris = np.asarray(aris)
    return int(np.argmin(np.abs(aris - np.median(aris)))), "closest_to_median_test_ari"


def export_npz(args, rna, label_key, n_classes, fold_latents, aris, nmis):
    fold, selection = pick_export_fold(aris, args.export_fold)
    emb = fold_latents[fold]
    # Same clustering call as the per-fold test step, applied to all cells.
    y_pred = KMeans(n_clusters=n_classes).fit_predict(emb)
    y_true = rna.obs[label_key].astype(str).to_numpy()
    if label_key != "Group":
        print(f"[export][WARN] y_true is taken from obs['{label_key}'], the guide expects obs['Group'] "
              "- pass --label-key Group if that column exists.")

    os.makedirs(args.export_dir, exist_ok=True)
    save_embedding(
        out_path=os.path.join(args.export_dir, f"{args.method_name}_{args.dataset}.npz"),
        emb=emb,
        cell_ids=rna.obs_names.to_numpy().astype(str),
        y_true=y_true,
        y_pred=y_pred,
        method=args.method_name,
        ari=adjusted_rand_score(y_true, y_pred),
        nmi=normalized_mutual_info_score(y_true, y_pred),
        emb_source=f"model.feature (h00) of fold {fold} on all cells, batch_size={args.batch_size}",
        fold=fold,
        fold_selection=selection,
        fold_test_ari=aris[fold],
        fold_test_nmi=nmis[fold],
        cv_mean_ari=np.mean(aris),
        cv_mean_nmi=np.mean(nmis),
    )


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rna", default="data/SNARE/RNA.h5ad")
    p.add_argument("--atac", default="data/SNARE/ATAC.h5ad")
    p.add_argument("--label-key", default=None)
    p.add_argument("--dataset", default="SNARE")
    p.add_argument("--n-top-genes", type=int, default=3000)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--apply-atac-tfidf",
        action="store_true",
        help="Actually run TF-IDF on the ATAC modality (see NOTE in preprocess_modality).",
    )
    p.add_argument("--no-export", action="store_true", help="Skip writing the .npz after training.")
    p.add_argument("--export-dir", default="output")
    p.add_argument("--method-name", default="scDRMAE", help="Method name in the .npz and its filename.")
    p.add_argument(
        "--export-fold",
        type=int,
        default=None,
        help="Fold whose model encodes all cells for the .npz (default: fold with test ARI closest to the median).",
    )
    args = p.parse_args()
    if args.export_fold is not None and not 0 <= args.export_fold < args.folds:
        p.error(f"--export-fold must be in [0, {args.folds - 1}]")
    return args


def main():
    args = get_args()
    setup_seed(args.seed)
    device = torch.device(args.device)

    hp = DATASET_HPARAMS.get(args.dataset, DEFAULT_HPARAMS)
    epochs, lr = hp["epochs"], hp["learning_rate"]
    print(f"dataset={args.dataset} epochs={epochs} lr={lr} device={device}")

    rna = sc.read_h5ad(args.rna)
    atac = sc.read_h5ad(args.atac)
    if rna.n_obs != atac.n_obs:
        raise ValueError(f"RNA has {rna.n_obs} cells but ATAC has {atac.n_obs}; expected paired cells.")
    if not rna.obs_names.equals(atac.obs_names):
        print("[WARN] RNA and ATAC obs_names differ (order or format); cells are paired by row "
              "position and the .npz uses RNA barcodes.")

    label_key = find_label_key(rna, args.label_key)
    classes, labels = np.unique(rna.obs[label_key].values, return_inverse=True)
    n_classes = len(classes)
    print(f"label_key={label_key} n_classes={n_classes}")

    x_rna = preprocess_modality(rna, args.n_top_genes, "RNA", False)
    x_atac = preprocess_modality(atac, args.n_top_genes, "ATAC", args.apply_atac_tfidf)

    x_rna_all = to_dense(x_rna.X).astype(np.float32)
    x_atac_all = to_dense(x_atac.X).astype(np.float32)
    print(f"RNA features={x_rna_all.shape[1]} ATAC features={x_atac_all.shape[1]}")

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    nmis, aris, amis = [], [], []
    fold_latents = {}  # fold -> (N, d) latent of all cells, for the .npz export

    for fold, (train_idx, val_idx) in enumerate(kf.split(x_rna_all)):
        x_rna_train = torch.from_numpy(x_rna_all[train_idx])
        x_atac_train = torch.from_numpy(x_atac_all[train_idx])
        x_rna_test = torch.from_numpy(x_rna_all[val_idx])
        x_atac_test = torch.from_numpy(x_atac_all[val_idx])
        y_train, y_test = labels[train_idx], labels[val_idx]

        M1, M2 = x_rna_train.shape[1], x_atac_train.shape[1]

        train_loader = torch.utils.data.DataLoader(
            PairedDataset(x_rna_train, x_atac_train, y_train),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            PairedDataset(x_rna_train, x_atac_train, y_train),
            batch_size=args.batch_size, shuffle=False, drop_last=False)
        test_loader = torch.utils.data.DataLoader(
            PairedDataset(x_rna_test, x_atac_test, y_test),
            batch_size=args.batch_size, shuffle=False, drop_last=False)

        mask_probas = [0.4] * M1
        mask_probas1 = [0.4] * M2

        model = scDRMAE(
            num_genes=M1, num_ATAC=M2, hidden_size=128,
            masked_data_weight=0.75, mask_loss_weight=0.7,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_ari, best_state, best_epoch = 0, None, 0
        for epoch in range(epochs):
            model.train()
            meter = AverageMeter()
            for x, x1, y in train_loader:
                x, x1 = x.float().to(device), x1.float().to(device)
                x_c, mask = apply_noise(x, mask_probas)
                x1_c, mask1 = apply_noise(x1, mask_probas1)
                optimizer.zero_grad()
                _, loss = model.loss_mask(x_c, x, mask, x1_c, x1, mask1, epoch, epochs)
                loss.backward()
                optimizer.step()
                meter.update(loss.detach().cpu().numpy())

            latent, true_label = inference(model, val_loader, device)
            pred_label = KMeans(n_clusters=n_classes, n_init=6).fit_predict(latent)
            _, ari, _, _ = evaluate(true_label, pred_label)
            if ari > best_ari:
                best_ari, best_epoch, best_state = ari, epoch, model.state_dict()

        print(f"[fold {fold}] best_epoch={best_epoch} best_train_ari={best_ari:.4f}")
        model.load_state_dict(best_state)
        model.eval()
        latent, true_label = inference(model, test_loader, device)
        pred_label = KMeans(n_clusters=n_classes).fit_predict(latent)
        nmi, ari, acc, ami = evaluate(true_label, pred_label)
        print(f"[fold {fold}] test: NMI={nmi:.4f} ARI={ari:.4f} AMI={ami:.4f}")

        nmis.append(nmi)
        aris.append(ari)
        amis.append(ami)

        if not args.no_export and args.export_fold in (None, fold):
            # Same weights that produced this fold's test result.
            fold_latents[fold] = encode_all(model, x_rna_all, x_atac_all, device, args.batch_size)

    print("=" * 60)
    print(f"Mean over {args.folds} folds: NMI={np.mean(nmis):.4f} ARI={np.mean(aris):.4f} AMI={np.mean(amis):.4f}")

    if not args.no_export:
        export_npz(args, rna, label_key, n_classes, fold_latents, aris, nmis)


if __name__ == "__main__":
    main()

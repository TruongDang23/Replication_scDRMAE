# Xuất kết quả gom cụm ra `.npz` để vẽ UMAP so sánh với scMCC

> **Đối tượng đọc:** AI coding agent (hoặc người) đang làm việc trong repo của một
> phương pháp baseline (scMDC, Seurat WNN, …). File này **tự đủ**: mọi code cần
> copy đều nằm ở đây, không cần clone repo scMCC.

## 0. Bối cảnh

Repo scMCC có script `umap_present.py`. Script nhận vào nhiều file `.npz`, mỗi file
là kết quả của **một phương pháp trên một dataset**, rồi vẽ hình gồm nhiều cột,
mỗi phương pháp một cột:

- hàng trên: UMAP của `emb`, tô màu theo `y_pred` (nhãn cụm được ghép màu với cell type bằng Hungarian matching);
- hàng dưới: cùng toạ độ UMAP đó, tô màu theo `y_true`.

Trước khi vẽ, script **căn hàng các file theo barcode** (`cell_ids`), chỉ giữ phần
giao. Ngoài ra nó **báo lỗi nếu `y_true` của hai file khác nhau trên cùng một barcode**.

**Nhiệm vụ trong repo baseline:** sau bước gom cụm cuối cùng của method, ghi ra
đúng 1 file `{METHOD}_{DATASET}.npz` theo contract ở mục 1. **Không** sửa logic
train / gom cụm của method.

## 1. Contract của file `.npz`

| Khoá       | Shape  | Kiểu      | Ý nghĩa |
|------------|--------|-----------|---------|
| `emb`      | (N, d) | float32   | Ma trận latent mà **bước gom cụm cuối cùng nhận vào**. d tuỳ method, d ≥ 2, không được có NaN/Inf. **Không** lưu toạ độ UMAP 2D (script sẽ tự chạy UMAP). |
| `cell_ids` | (N,)   | str       | Barcode gốc lấy từ `obs_names` của file `.h5ad`, **giữ nguyên định dạng**, không được trùng. |
| `y_true`   | (N,)   | str       | **Tên** cell type lấy từ cột `obs['Group']` của cùng file `.h5ad`. Không lưu mã số 0..K-1. |
| `y_pred`   | (N,)   | int       | Nhãn cụm cuối cùng của method. |
| `method`   | scalar | str       | Tên hiển thị trên hình, ví dụ `"scMDC"`, `"Seurat WNN"`. |
| `ari`      | scalar | float     | ARI của **chính** `y_pred` này so với `y_true`. |
| `nmi`      | scalar | float     | NMI của **chính** `y_pred` này so với `y_true`. |
| *(tuỳ chọn)* | —    | —         | Có thể thêm khoá phụ qua `**extra`, ví dụ `emb_source="z_fused"` hoặc `epoch=...`. |

**Ràng buộc quan trọng nhất:** hàng thứ `i` của `emb`, `cell_ids`, `y_true`,
`y_pred` phải là **cùng một tế bào**.

Dữ liệu phía scMCC dùng để đối chiếu:
- `cell_ids` là `obs_names` của file RNA `.h5ad`, đã giao với ATAC và lọc `filter_cells(min_counts=1)` trên từng omics.
- `y_true = rna.obs['Group'].astype(str)`.

Baseline nên đọc **cùng hai file `.h5ad`** này.

## 2. Code để copy: `npz_export.py`

Phần lưu/đọc chỉ cần `numpy`. Riêng `npz_from_csv` cần thêm `pandas`. Copy nguyên
file sau vào repo baseline:

```python
"""Xuất kết quả của một phương pháp ra định dạng .npz chung để vẽ UMAP cùng scMCC.

Contract: emb (N,d) float32 | cell_ids (N,) str | y_true (N,) str |
          y_pred (N,) int | method str | ari, nmi float
"""

import numpy as np


def save_embedding(out_path, emb, cell_ids, y_true, y_pred,
                   method, ari=None, nmi=None, **extra):
    """Ghi file .npz theo contract ở docstring module."""
    emb = np.asarray(emb, dtype=np.float32)
    cell_ids = np.asarray(cell_ids, dtype=object)
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred)

    n = emb.shape[0]
    for name, arr in (('cell_ids', cell_ids), ('y_true', y_true), ('y_pred', y_pred)):
        if len(arr) != n:
            raise ValueError(f"{name} có {len(arr)} phần tử nhưng emb có {n} hàng "
                             f"— thứ tự tế bào đã lệch, kiểm tra lại loader.")
    if len(np.unique(cell_ids)) != n:
        raise ValueError("cell_ids bị trùng — không căn hàng được giữa các phương pháp.")

    np.savez_compressed(
        out_path,
        emb=emb,
        cell_ids=cell_ids,
        y_true=y_true,
        y_pred=y_pred,
        method=str(method),
        ari=np.float64(ari if ari is not None else np.nan),
        nmi=np.float64(nmi if nmi is not None else np.nan),
        **extra,
    )
    print(f"[export] {out_path}  emb={emb.shape}  "
          f"n_types={len(np.unique(y_true))}  n_clusters={len(np.unique(y_pred))}  "
          f"ARI={ari}  NMI={nmi}")


def load_embedding(path):
    """Đọc ngược file .npz -> dict, ép kiểu về str/float cho tiện dùng."""
    d = np.load(path, allow_pickle=True)
    return {
        'emb': d['emb'],
        'cell_ids': d['cell_ids'].astype(str),
        'y_true': d['y_true'].astype(str),
        'y_pred': d['y_pred'],
        'method': str(d['method']),
        'ari': float(d['ari']),
        'nmi': float(d['nmi']),
    }


def npz_from_csv(csv_path, out_path, method, y_true=None, cell_id_col=None,
                 pred_col='cluster', ari=None, nmi=None):
    """Chuyển output của baseline (CSV: index = barcode, các cột = chiều latent)
    sang .npz cùng contract.

    :param y_true: Series/dict ánh xạ barcode → cell type. Bỏ trống nếu CSV đã
        có cột 'cell_type'.
    :param pred_col: tên cột chứa nhãn cụm trong CSV; nếu không có thì y_pred
        được gán -1.
    """
    import pandas as pd

    df = pd.read_csv(csv_path, index_col=0 if cell_id_col is None else cell_id_col)
    y_pred = df.pop(pred_col).to_numpy() if pred_col in df.columns else np.full(len(df), -1)
    if 'cell_type' in df.columns:
        labels = df.pop('cell_type').astype(str).to_numpy()
    else:
        if y_true is None:
            raise ValueError("CSV không có cột 'cell_type' và cũng không truyền y_true.")
        labels = pd.Series(y_true).reindex(df.index).astype(str).to_numpy()

    save_embedding(out_path, df.to_numpy(), df.index.to_numpy().astype(str),
                   labels, y_pred, method=method, ari=ari, nmi=nmi)
```

> **Không copy** hàm `extract_final_state` trong `scMCC/export.py`. Hàm đó gọi
> `model.encodeBatch` và `model.kmeans_loss`, là các API riêng của scMCC.

## 3. Các bước tích hợp vào một baseline viết bằng Python

### Bước 1: Xác định 3 thứ trong code của baseline (đọc code, chưa sửa gì)

1. **Bước gom cụm cuối cùng** của method, tức chỗ sinh ra nhãn được dùng để báo ARI/NMI.
   Đó có thể là `KMeans.fit_predict`, `argmax` của soft assignment, `sc.tl.leiden`, …
2. **Ma trận đầu vào của bước đó.** Ma trận này chính là `emb`. Quy tắc:
   *`emb` = đúng ma trận mà bước gom cụm cuối cùng nhận vào.*
   - Method có hai encoder nhưng gom cụm trên latent đã fusion hoặc concat → lấy latent đã fusion/concat.
   - Method gom cụm trên graph (Leiden/Louvain) → lấy ma trận dùng để dựng graph, và ghi rõ nguồn bằng `emb_source=...`.
3. **AnnData/DataFrame chứa barcode và nhãn**, ở trạng thái **sau mọi bước lọc tế bào** của baseline.

Ghi lại `file:dòng` và shape của từng thứ. Nếu bước nào mơ hồ, ví dụ method có
nhiều latent hoặc nhiều lần gom cụm, **hỏi lại người dùng trước khi sửa**.

### Bước 2: Đảm bảo thứ tự hàng khớp nhau

- Khi encode toàn bộ dữ liệu để lấy `emb`, dùng `DataLoader(..., shuffle=False, drop_last=False)`.
- Nếu buộc phải dùng loader có shuffle, cho `Dataset.__getitem__` trả thêm `index`,
  gom lại rồi sắp xếp: `order = np.argsort(idx_all); Z = Z[order]; y_pred = y_pred[order]`.
- `cell_ids` và `y_true` phải lấy từ **cùng AnnData, cùng thứ tự** với dữ liệu đã đưa vào loader.

### Bước 3: Lấy `y_true` dạng tên

```python
y_true = adata.obs['Group'].astype(str).to_numpy()
# Nếu baseline chỉ giữ mã số: y_true = adata.obs['Group'].cat.categories[codes]
```

### Bước 4: Gọi `save_embedding` ngay sau bước gom cụm cuối cùng

```python
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from npz_export import save_embedding

METHOD, DATASET, OUT_DIR = "scMDC", "PBMC", "output"   # ← sửa cho từng baseline

Z = ...        # (N, d). Nếu là torch tensor: Z.detach().cpu().numpy()
y_pred = ...   # (N,). Nếu là chuỗi (vd: leiden): np.asarray(y_pred).astype(int)
y_true = adata.obs['Group'].astype(str).to_numpy()

save_embedding(
    out_path   = f"{OUT_DIR}/{METHOD}_{DATASET}.npz",
    emb        = Z,
    cell_ids   = adata.obs_names.to_numpy().astype(str),
    y_true     = y_true,
    y_pred     = y_pred,
    method     = METHOD,
    ari        = adjusted_rand_score(y_true, y_pred),
    nmi        = normalized_mutual_info_score(y_true, y_pred),
    emb_source = "<tên biến/tầng sinh ra Z>",   # tuỳ chọn, để truy vết
)
```

Lấy kết quả **cuối cùng hay tốt nhất**: dùng đúng kết quả mà bảng ARI/NMI trong
bài báo đang báo cáo cho method đó. scMCC xuất trạng thái **cuối cùng**
(không phải best checkpoint). Nếu baseline có cơ chế best/early-stopping khác,
ghi rõ bằng `**extra`, ví dụ `checkpoint="best"`.

### Bước 5: Chạy kiểm tra (mục 6)

## 4. Baseline viết bằng R

Xuất CSV với **index = barcode**, các cột là các chiều latent, thêm hai cột
`cluster` (int) và `cell_type` (tên). Ví dụ với Seurat (đổi tên reduction và tên
cột cho đúng baseline):

```r
emb <- Embeddings(obj, reduction = "<tên reduction mà bước gom cụm dùng>")  # N x d, rownames = barcode
df  <- data.frame(emb,
                  cluster   = as.integer(Idents(obj)),
                  cell_type = as.character(obj$Group),
                  check.names = FALSE)
stopifnot(identical(rownames(df), colnames(obj)))   # barcode giữ nguyên thứ tự
write.csv(df, "SeuratWNN_PBMC.csv", row.names = TRUE)
```

Sau đó chuyển sang `.npz` bằng Python:

```python
from npz_export import npz_from_csv
npz_from_csv("SeuratWNN_PBMC.csv", "output/SeuratWNN_PBMC.npz", method="Seurat WNN",
             ari=..., nmi=...)   # điền số mà method báo cáo, hoặc để trống rồi tính ở mục 6
```

Nếu `y_pred` toàn `-1` (CSV không có cột `cluster`), panel dự đoán sẽ vô nghĩa.
Bắt buộc phải có cột `cluster`.

## 5. Bẫy thường gặp

| Triệu chứng khi chạy `umap_present.py` / `check_npz.py` | Nguyên nhân | Cách sửa |
|---|---|---|
| `Giao của các tập cell_ids rỗng` | Barcode bị đổi: index reset về 0..N-1, thêm/bớt hậu tố `-1`, R đổi `-` thành `.` | Lấy lại `obs_names` gốc, không để R `make.names` biến đổi barcode |
| `[align] X: N -> M cells` với M nhỏ hơn nhiều | Baseline lọc tế bào khác scMCC | Dùng cùng file `.h5ad`; chấp nhận được nếu chỉ lệch ít |
| `y_true của X khác của scMCC` | Lưu mã số thay vì tên, hoặc lấy sai cột nhãn | `obs['Group'].astype(str)` |
| Hình hàng trên loang lổ, ARI tính lại ≈ 0 dù method báo cáo cao | `emb`/`y_pred` lệch thứ tự so với `cell_ids` (thường do shuffle) | Xem mục 3, bước 2 |
| ARI/NMI trong file khác số tính lại | Lưu ARI của checkpoint best nhưng `y_pred` lại là của final, hoặc ngược lại | Tính ARI/NMI từ chính `y_pred` được lưu |
| UMAP lỗi NaN | `emb` có NaN/Inf | Sửa ở nguồn, không thay NaN bằng 0 |
| UMAP trông như một đường cong hoặc một đám | Lưu nhầm toạ độ UMAP/t-SNE 2D thay vì latent | Lưu latent gốc |

## 6. Kiểm tra trước khi đưa file vào scMCC: `check_npz.py`

Cần `numpy` và `scikit-learn`. Cần thêm file `scMCC_{DATASET}.npz` làm chuẩn đối chiếu.

```python
"""Kiểm tra một file .npz có ghép được với file .npz của scMCC hay không.

    python check_npz.py --ref scMCC_PBMC.npz --cand scMDC_PBMC.npz
"""
import argparse
import sys

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Console Windows mặc định cp1252 → print tiếng Việt sẽ lỗi.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

REQUIRED = ("emb", "cell_ids", "y_true", "y_pred", "method", "ari", "nmi")


def load(path):
    d = np.load(path, allow_pickle=True)
    missing = [k for k in REQUIRED if k not in d.files]
    assert not missing, f"{path}: thiếu khoá {missing}"
    return {k: d[k] for k in d.files}


ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True, help="File .npz của scMCC (chuẩn đối chiếu)")
ap.add_argument("--cand", required=True, help="File .npz cần kiểm tra")
args = ap.parse_args()
ref, cand = load(args.ref), load(args.cand)

# 1. Hình dạng & kiểu dữ liệu
emb = cand["emb"]
n = emb.shape[0]
assert emb.ndim == 2 and emb.shape[1] >= 2, f"emb phải là (N, d>=2), đang là {emb.shape}"
assert np.isfinite(emb).all(), "emb chứa NaN/Inf — UMAP sẽ lỗi"
for k in ("cell_ids", "y_true", "y_pred"):
    assert len(cand[k]) == n, f"{k} có {len(cand[k])} phần tử, emb có {n} hàng"
ids = cand["cell_ids"].astype(str)
assert len(np.unique(ids)) == n, "cell_ids bị trùng"
assert np.issubdtype(cand["y_pred"].dtype, np.integer), f"y_pred phải là int, đang là {cand['y_pred'].dtype}"
assert not (cand["y_pred"] == -1).all(), "y_pred toàn -1 — chưa có nhãn cụm"

print(f"method={cand['method']}  N={n}  d={emb.shape[1]}  "
      f"K_pred={len(np.unique(cand['y_pred']))}  K_true={len(np.unique(cand['y_true']))}")

# 2. Căn hàng với scMCC theo barcode
ref_ids = ref["cell_ids"].astype(str)
common, i_ref, i_cand = np.intersect1d(ref_ids, ids, return_indices=True)
print(f"barcode chung với ref: {len(common)} "
      f"(= {len(common) / len(ref_ids):.1%} của ref, {len(common) / n:.1%} của cand)")
assert len(common) > 0, "Không có barcode chung — barcode bị đổi định dạng?"
if len(common) < 0.9 * len(ref_ids):
    print("[WARN] mất >10% tế bào so với scMCC — kiểm tra bước lọc tế bào / định dạng barcode")

# 3. Ground truth phải trùng khớp trên phần giao
bad = (ref["y_true"].astype(str)[i_ref] != cand["y_true"].astype(str)[i_cand]).sum()
assert bad == 0, f"{bad} tế bào có y_true khác scMCC — sai cột nhãn hoặc đang lưu mã số thay vì tên"

# 4. ARI/NMI lưu trong file phải khớp với chính y_pred
yt, yp = cand["y_true"].astype(str), cand["y_pred"]
ari, nmi = adjusted_rand_score(yt, yp), normalized_mutual_info_score(yt, yp)
print(f"ARI tính lại={ari:.4f} (trong file {float(cand['ari']):.4f}) | "
      f"NMI tính lại={nmi:.4f} (trong file {float(cand['nmi']):.4f})")
if not (np.isclose(ari, float(cand["ari"]), atol=1e-3) and np.isclose(nmi, float(cand["nmi"]), atol=1e-3)):
    print("[WARN] ARI/NMI trong file không khớp với y_pred — có thể đang lưu số của checkpoint khác")

print("OK")
```

**Điều kiện đạt:**
- Script in ra `OK`, không có `AssertionError`.
- Không có `[WARN]`, hoặc mọi `[WARN]` đều đã được giải thích.
- `ARI tính lại` gần với số mà method báo cáo trong bài.

## 7. Đưa file vào hình

Copy `{METHOD}_{DATASET}.npz` vào thư mục output của scMCC rồi chạy trong repo scMCC:

```bash
python umap_present.py \
    --inputs output/scMCC_PBMC.npz output/scMDC_PBMC.npz output/SeuratWNN_PBMC.npz \
    --out figures/Fig_UMAP_PBMC --per_row 4
```

Thứ tự trong `--inputs` là thứ tự các cột trên hình. Bảng màu cell type lấy theo
file đầu tiên, nên đặt scMCC đầu tiên.

## 8. Prompt mẫu khi giao cho AI agent trong repo baseline

```text
Đọc EXPORT_NPZ_GUIDE.md. Trong repo này, method là <TÊN METHOD>, dataset là <PBMC>,
dữ liệu đầu vào là <đường dẫn RNA.h5ad> và <đường dẫn ATAC.h5ad>.

1. Làm mục 3 - Bước 1: chỉ đọc code, báo lại file:dòng + shape của
   (a) bước gom cụm cuối cùng, (b) ma trận đầu vào của bước đó, (c) nơi giữ barcode/nhãn.
   Dừng lại chờ tôi xác nhận.
2. Sau khi tôi xác nhận: tạo npz_export.py (copy nguyên mục 2), gọi save_embedding
   ngay sau bước gom cụm cuối cùng theo mục 3 - Bước 4. Không đổi logic train/gom cụm.
3. Tạo check_npz.py (copy nguyên mục 6), chạy với --ref <đường dẫn scMCC_PBMC.npz>
   và dán toàn bộ output.
```

import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits
from torch.nn.functional import mse_loss as mse
from torch.nn.functional import normalize

class scDRMAE(torch.nn.Module):
    def __init__(
        self,
        num_genes,
        num_ATAC,
        hidden_size=128,
        dropout=0,
        masked_data_weight=.75,
        mask_loss_weight=0.7,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.num_ATAC = num_ATAC
        self.masked_data_weight = masked_data_weight
        self.mask_loss_weight = mask_loss_weight

        self.encoder = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_genes, 256),
            nn.LayerNorm(256),
            nn.Mish(inplace=True),
            nn.Linear(256, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Mish(inplace=True),
            nn.Linear(hidden_size, hidden_size)
        )
        self.encoder1 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_ATAC, 256),
            nn.LayerNorm(256),
            nn.Mish(inplace=True),
            nn.Linear(256, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Mish(inplace=True),
            nn.Linear(hidden_size, hidden_size)
        )

        self.mask_predictor = nn.Linear(hidden_size, num_genes)
        self.mask_predictor1 = nn.Linear(hidden_size, num_ATAC)
        self.decoder = nn.Linear(
            in_features=hidden_size+num_genes, out_features=num_genes)
        self.decoder1 = nn.Linear(
            in_features=hidden_size+num_ATAC, out_features=num_ATAC)
        self.trans_enc = nn.TransformerEncoderLayer(d_model= 2*hidden_size, nhead=1, dim_feedforward=256)
        self.extract_layers = nn.TransformerEncoder(self.trans_enc, num_layers=1)
    def cal_latent(self, z):
        sum_y = torch.sum(torch.square(z), dim=1)
        num = -2.0 * torch.matmul(z, z.t()) + torch.reshape(sum_y, [-1, 1]) + sum_y
        num = num / 1
        num = torch.pow(1.0 + num, -(1 + 1.0) / 2.0)
        zerodiag_num = num - torch.diag(torch.diag(num))
        latent_p = (zerodiag_num.t() / torch.sum(zerodiag_num, dim=1)).t()
        return num, latent_p
    def forward_mask(self, x,x1):
        latent = self.encoder(x)
        predicted_mask = self.mask_predictor(latent)
        reconstruction = self.decoder(
            torch.cat([latent, predicted_mask], dim=1))
        latent1 = self.encoder1(x1)
        predicted_mask1 = self.mask_predictor1(latent1)
        reconstruction1 = self.decoder1(
            torch.cat([latent1, predicted_mask1], dim=1))

        h00 = self.extract_layers(torch.cat((latent, latent1), 1).unsqueeze(1))
        h00=h00.squeeze(1)
        h00 = torch.cat([latent, h00], dim=-1)
        num, lq = self.cal_latent(h00)
        return h00, predicted_mask, reconstruction, predicted_mask1, reconstruction1,num, lq
        # return h00, predicted_mask, reconstruction, predicted_mask1, reconstruction1
    def kldloss(self, p, q):
            c1 = -torch.sum(p * torch.log(q), dim=-1)
            c2 = -torch.sum(p * torch.log(p), dim=-1)
            return torch.mean(c1 - c2)
    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()
    def loss_mask(self,x, y, mask,x1, y1, mask1,epoch,epochs):
        fea, predicted_mask, reconstruction, predicted_mask1, reconstruction1,num,lq = self.forward_mask(x,x1)
        ##RNA
        w_nums = mask * self.masked_data_weight + (1 - mask) * (1 - self.masked_data_weight)
        reconstruction_loss = (1-self.mask_loss_weight) * torch.mul(
            w_nums, mse(reconstruction, y, reduction='none'))

        mask_loss = self.mask_loss_weight * \
            bce_logits(predicted_mask, mask, reduction="mean")
        reconstruction_loss = reconstruction_loss.mean()
        ##ATAC
        w_nums1 = mask1 * self.masked_data_weight + (1 - mask1) * (1 - self.masked_data_weight)
        reconstruction_loss1 = (1-self.mask_loss_weight) * torch.mul(
            w_nums1, mse(reconstruction1, y1, reduction='none'))

        mask_loss1 = self.mask_loss_weight * \
            bce_logits(predicted_mask1, mask1, reduction="mean")
        reconstruction_loss1 = reconstruction_loss1.mean()

        lpbatch = self.target_distribution(lq)
        lqbatch = lq + torch.diag(torch.diag(num))
        lpbatch = lpbatch + torch.diag(torch.diag(num))
        kl_loss = self.kldloss(lpbatch, lqbatch) 
        if epoch+1 >= epochs * 0.5:
                loss = reconstruction_loss + mask_loss +reconstruction_loss1+mask_loss1 +0.000001*kl_loss
        else:
                loss = reconstruction_loss + mask_loss +reconstruction_loss1+mask_loss1
        return fea, loss

    def feature(self, x,x1):
        fea, predicted_mask, reconstruction, predicted_mask1, reconstruction1,_,_ = self.forward_mask(x,x1)
        return fea
    def chabu(self, x,x1):
        fea, predicted_mask, reconstruction, predicted_mask1, reconstruction1,_,_ = self.forward_mask(x,x1)
        return reconstruction,reconstruction1

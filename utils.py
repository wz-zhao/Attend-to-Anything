#---------------------------------------
# Copyright (c) Hyperbolic Saliency Lab.
# All rights reserved.
#---------------------------------------

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


import lorentz as L
import torch.distributed as dist
def lorentz_sanitize(z, curv, eps=1e-4, max_val=1e6):

    z = z.to(torch.float32)

    z = torch.nan_to_num(z, nan=0.0, posinf=max_val, neginf=-max_val)


    t_min = (1.0 / torch.sqrt(curv)).to(z.dtype) + eps

    z0 = z[..., 0].clamp_min(t_min)
    z = torch.cat([z0.unsqueeze(-1), z[..., 1:]], dim=-1)
    z = z.clamp(-max_val, max_val)
    return z


def lorentz_dist_global_to_origin(z: torch.Tensor, curv: torch.Tensor, eps: float = 1e-8):
    x_time = torch.sqrt(1.0 / curv + torch.sum(z**2, dim=-1, keepdim=True))
    val = torch.clamp((curv**0.5) * x_time, min=1.0 + eps)
    return torch.acosh(val) / (curv**0.5)

def lorentz_diag_dist(x: torch.Tensor, y: torch.Tensor, curv: torch.Tensor):
    with torch.autocast(device_type=x.device.type, dtype=torch.float32):
        dmat = L.pairwise_dist(x.to(torch.float32), y.to(torch.float32), curv)
    return torch.diag(dmat).view(-1, 1).to(dtype=x.dtype)


class PixelLorentzMap(nn.Module):
    def __init__(self, channels: int, curv: float = 1.0, learn_curv: bool = False, max_norm: float = 8.0):
        super().__init__()
        self.curv = nn.Parameter(torch.tensor(curv).log(), requires_grad=learn_curv)
        self.max_norm = max_norm
        self.alpha = nn.Parameter(torch.tensor(channels**-0.5).log())
        self._curv_minmax = {"max": math.log(curv * 10), "min": math.log(curv / 10)}

    def current_curv(self):
        if self.curv.requires_grad:
            self.curv.data = torch.clamp(self.curv.data, min=self._curv_minmax["min"], max=self._curv_minmax["max"])
        return self.curv.exp()

    def _flatten(self, x):
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(B * H * W, C), (B, C, H, W)

    def _unflatten(self, x, shape):
        B, C, H, W = shape
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def to_hyp(self, x_euc: torch.Tensor):
        curv = self.current_curv()
        self.alpha.data = torch.clamp(self.alpha.data, max=0.0)
        x, shp = self._flatten(x_euc)
        x = x * self.alpha.exp()
        n = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        n_clip = n.clamp_max(self.max_norm)
        x = x * (n_clip / n)
        with torch.autocast(device_type=x_euc.device.type, dtype=torch.float32):
            z = L.exp_map0(x.to(torch.float32), curv)
        return self._unflatten(z.to(dtype=x_euc.dtype), shp)

    def to_tan(self, z_hyp: torch.Tensor):
        curv = self.current_curv()
        z, shp = self._flatten(z_hyp)
        with torch.autocast(device_type=z_hyp.device.type, dtype=torch.float32):
            x = L.log_map0(z.to(torch.float32), curv)
        return self._unflatten(x.to(dtype=z_hyp.dtype), shp)

class GlobalLorentzMap(nn.Module):
    def __init__(self, dim: int, curv: float = 1.0, learn_curv: bool = False, max_norm: float = 8.0):
        super().__init__()
        self.curv = nn.Parameter(torch.tensor(curv).log(), requires_grad=learn_curv)
        self.alpha = nn.Parameter(torch.tensor(dim**-0.5).log())
        self.max_norm = max_norm
        self._curv_minmax = {"max": math.log(curv * 10), "min": math.log(curv / 10)}

    def current_curv(self):
        if self.curv.requires_grad:
            self.curv.data = torch.clamp(self.curv.data, min=self._curv_minmax["min"], max=self._curv_minmax["max"])
        return self.curv.exp()

    def to_hyp(self, x_tan: torch.Tensor):
        curv = self.current_curv()
        self.alpha.data = torch.clamp(self.alpha.data, max=0.0)
        x = x_tan * self.alpha.exp()
        n = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        n_clip = n.clamp_max(self.max_norm)
        x = x * (n_clip / n)
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            z = L.exp_map0(x.to(torch.float32), curv)
        return z.to(dtype=x_tan.dtype)


class HyperbolicConditionGate(nn.Module):
    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, channels), nn.Sigmoid())
    def forward(self, z_cond, curv):
        r = lorentz_dist_global_to_origin(z_cond, curv)
        return self.mlp(r)[:, :, None, None]

class HyperbolicScaleGatingLorentz(nn.Module):
    def __init__(self, text_dim, num_scales, hyp_dim, curv=1.0, learn_curv=False):
        super().__init__()
        self.map = GlobalLorentzMap(dim=hyp_dim, curv=curv, learn_curv=learn_curv)
        self.scale_mlp = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, 128), nn.GELU(), nn.Linear(128, num_scales))
        self.scale_anchors_tan = nn.Parameter(torch.randn(num_scales, hyp_dim) * (hyp_dim**-0.5))

    def forward(self, text_emb, z_cond):
        base_weights = F.softplus(self.scale_mlp(text_emb))
        if z_cond is None: return base_weights, None
        curv = self.map.current_curv()
        anchors = self.map.to_hyp(self.scale_anchors_tan)
        with torch.autocast(device_type=z_cond.device.type, dtype=torch.float32):
            d = L.pairwise_dist(z_cond.to(torch.float32), anchors.to(torch.float32), curv)
        w_hyp = F.softmax(-d.to(dtype=text_emb.dtype), dim=-1)
        return base_weights * (1.0 + w_hyp), w_hyp

class HyperbolicMultiScaleSpatialOperator(nn.Module):
    def __init__(self, channels, num_scales=3, curv=1.0, learn_curv=False):
        super().__init__()
        self.map = PixelLorentzMap(channels, curv, learn_curv)
        self.cond_gate = HyperbolicConditionGate(channels)
        self.op0 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False), nn.Conv2d(channels, channels, 1), nn.SiLU())
        self.op1 = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False), nn.Conv2d(channels, channels, 1), nn.SiLU())
        self.op2_proj = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.SiLU())
        self.mix = nn.Conv2d(channels, channels, 1)

    def forward(self, x_euc, z_cond=None):
        curv = self.map.current_curv()
        z_pix = self.map.to_hyp(x_euc)
        z_pix = lorentz_sanitize(z_pix, curv)  

        if z_cond is not None:
            gate = self.cond_gate(z_cond, curv).to(dtype=x_euc.dtype)
            gate = 0.5 * torch.tanh(gate)  

            x_tan = self.map.to_tan(z_pix)
            max_norm = 10.0
            n = x_tan.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            x_tan = x_tan * (max_norm / n).clamp(max=1.0)

            x_tan = x_tan * (1.0 + gate)

            z_pix = self.map.to_hyp(x_tan)
            z_pix = lorentz_sanitize(z_pix, curv) 

        x_tan = self.map.to_tan(z_pix)

        B, C, H, W = x_tan.shape
        return torch.stack([self.op0(x_tan), self.op1(x_tan), self.op2_proj(x_tan).expand(-1, -1, H, W)], dim=1)

    def fuse(self, Y, weights):
        return self.mix((Y * weights[:, :, None, None, None]).sum(dim=1))

class HyperbolicMultiScaleUpsampleLorentz(nn.Module):
    def __init__(self, in_channels, out_channels, text_dim=768, num_scales=3, hyp_dim=128, curv=1.0, learn_curv=False):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.norm = nn.GroupNorm(8, out_channels)
        self.style_mlp = nn.Sequential(nn.SiLU(), nn.Linear(text_dim, 2 * out_channels))
        self.scale_gate = HyperbolicScaleGatingLorentz(text_dim, num_scales, hyp_dim, curv, learn_curv)
        self.ms_op = HyperbolicMultiScaleSpatialOperator(out_channels, num_scales, curv, learn_curv)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, text_emb, z_cond):
        x = self.norm(self.up_conv(x))
        style = self.style_mlp(text_emb)
        gamma, beta = style.chunk(2, dim=1)
        x = x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        Y = self.ms_op(x, z_cond)
        scale_weights, _ = self.scale_gate(text_emb, z_cond)
        y_fused = self.ms_op.fuse(Y, scale_weights)
        
        lam = 0.25
        if z_cond is not None:
            r = lorentz_dist_global_to_origin(z_cond, self.scale_gate.map.current_curv())
            lam = (0.1 + 0.4 * r.clamp(0, 1)).view(x.size(0), 1, 1, 1).to(dtype=x.dtype)
        return self.act(x + lam * y_fused)

class HyperbolicMultiScaleBiasHeadLorentzJointAmp(nn.Module):
    def __init__(self, in_channels, nclass=1, text_dim=768, num_gaussians=6, num_scales=3, hyp_dim=128, curv=1.0, learn_curv=False):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.hyp_dim = hyp_dim
        self.final_conv = nn.Conv2d(in_channels, nclass, 3, padding=1)
        self.shape_predictor = nn.Sequential(nn.Linear(text_dim, 128), nn.ReLU(), nn.Linear(128, 4 * num_gaussians))
        self.img_pool = nn.AdaptiveAvgPool2d(1)
        self.img_to_hyp_tan = nn.Linear(in_channels, hyp_dim, bias=False)
        self.global_map = GlobalLorentzMap(dim=hyp_dim, curv=curv, learn_curv=learn_curv)
        self.amp_mlp = nn.Sequential(nn.LayerNorm(2 * hyp_dim + 3), nn.Linear(2 * hyp_dim + 3, 256), nn.GELU(), nn.Linear(256, num_gaussians))
        self.gauss_anchors_tan = nn.Parameter(torch.randn(num_gaussians, hyp_dim) * (hyp_dim**-0.5))
        self.ms_op = HyperbolicMultiScaleSpatialOperator(nclass, num_scales, curv, learn_curv)
        self.scale_gate = HyperbolicScaleGatingLorentz(text_dim, num_scales, hyp_dim, curv, learn_curv)
        self._grid_cache = {}

    def _get_unit_grid(self, H, W, device):
                 
        if (device, H, W) not in self._grid_cache:
            y = torch.linspace(0, 1, H, device=device)
            x = torch.linspace(0, 1, W, device=device)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            self._grid_cache[(device, H, W)] = (xx[None, None], yy[None, None])
        return self._grid_cache[(device, H, W)]

    def forward(self, x, text_emb, z_cond=None):
        B, _, H, W = x.shape
        curv = self.global_map.current_curv()
        spatial_out = self.final_conv(x)
        G = self.num_gaussians
        shp = self.shape_predictor(text_emb).view(B, G, 4)

        cx = torch.sigmoid(shp[..., 0])                
        cy = torch.sigmoid(shp[..., 1])                
        sx = torch.sigmoid(shp[..., 2]) * 0.5 + 0.05   
        sy = torch.sigmoid(shp[..., 3]) * 0.5 + 0.05   


        z_img = self.global_map.to_hyp(self.img_to_hyp_tan(self.img_pool(x).flatten(1)))


        if z_cond is None:
            z_cond = self.global_map.to_hyp(
                torch.zeros_like(self.img_to_hyp_tan.weight[0]).unsqueeze(0).expand(B, -1)
            )

        d_it = lorentz_diag_dist(z_img, z_cond, curv)            
        r_img = lorentz_dist_global_to_origin(z_img, curv)      
        r_cond = lorentz_dist_global_to_origin(z_cond, curv)       

        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            joint = torch.cat(
                [L.log_map0(z_img, curv), L.log_map0(z_cond, curv), d_it, r_img, r_cond],
                dim=-1
            )

        amp = torch.sigmoid(self.amp_mlp(joint.to(x.dtype))).view(B, G, 1, 1)  

        xx, yy = self._get_unit_grid(H, W, x.device)            
        g = torch.exp(
            -(
                (xx - cx[..., None, None]) ** 2 / (2 * sx[..., None, None] ** 2) +
                (yy - cy[..., None, None]) ** 2 / (2 * sy[..., None, None] ** 2)
            )
        ) 

        
        if g.dim() == 5 and g.size(2) == 1:
            g = g.squeeze(2)  

        if g.size(0) == G and g.size(1) == B:
            g = g.permute(1, 0, 2, 3).contiguous()

   
        if amp.size(0) == G and amp.size(1) == B:
            amp = amp.permute(1, 0, 2, 3).contiguous()

        assert g.shape[0] == B and g.shape[1] == G, f"g shape wrong: {g.shape}, expected [B,G,H,W]"
        assert amp.shape[0] == B and amp.shape[1] == G, f"amp shape wrong: {amp.shape}, expected [B,G,1,1]"

        bias_map = (g * amp).sum(dim=1, keepdim=True)             

        Y = self.ms_op(spatial_out, z_cond=z_cond)
        scale_weights, _ = self.scale_gate(text_emb, z_cond)
        struct_bias = self.ms_op.fuse(Y, scale_weights)

        lam_struct = (0.2 + 0.5 * r_cond.clamp(0, 1)).view(B, 1, 1, 1)

        return spatial_out + 0.5 * bias_map + lam_struct * struct_bias


class HyperbolicZeroShotSaliency(nn.Module):

    def __init__(
        self,
        visual_encoder: nn.Module,
        text_encoder: nn.Module,
        decoder: HyperbolicMultiScaleUpsampleLorentz, 
        bias_head: HyperbolicMultiScaleBiasHeadLorentzJointAmp, 
        embed_dim: int = 128,
        curv_init: float = 1.0,
        entail_weight: float = 1.0,
    ):
        super().__init__()
        self.visual = visual_encoder
        self.textual = text_encoder
        self.decoder = decoder
        self.bias_head = bias_head
        self.embed_dim = embed_dim
        self.entail_weight = entail_weight
        
        self.curv = nn.Parameter(torch.tensor(curv_init).log())
        self.visual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self._curv_minmax = {"max": math.log(curv_init * 10), "min": math.log(curv_init / 10)}
        

        self.visual_proj = nn.Linear(visual_encoder.num_features, embed_dim, bias=False)
        self.textual_proj = nn.Linear(text_encoder.width, embed_dim, bias=False)
        

        self.none_anchor_tan = nn.Parameter(torch.zeros(1, embed_dim))
        self._rank = dist.get_rank() if dist.is_available() else 0

    @property
    def device(self): return self.curv.device

    def _update_curv(self):
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        return self.curv.exp()

    def encode_hyp(self, feats, alpha, curv):
        alpha.data = torch.clamp(alpha.data, max=0.0)
        feats = feats * alpha.exp()
        with torch.autocast(self.device.type, dtype=torch.float32):
            z = L.exp_map0(feats, curv)
        return z

    def forward(self, images, text_tokens=None, anchor_tokens=None, gt_saliency=None):

        _curv = self._update_curv()
        B = images.shape[0]


        spatial_feats = self.visual(images) 
        img_global = torch.mean(spatial_feats[-1], dim=(2,3)) 
        z_img = self.encode_hyp(self.visual_proj(img_global), self.visual_alpha, _curv)

        z_none = self.encode_hyp(self.none_anchor_tan, self.textual_alpha, _curv) 
        
        if text_tokens is not None:
            txt_emb = self.textual(text_tokens)
            z_txt = self.encode_hyp(self.textual_proj(txt_emb), self.textual_alpha, _curv)
        else:
            z_txt = z_none.expand(B, -1) 

        if anchor_tokens is not None:
            anc_emb = self.textual(anchor_tokens)
            z_anc = self.encode_hyp(self.textual_proj(anc_emb), self.textual_alpha, _curv)
        else:
            z_anc = z_none.expand(B, -1)
       
        text_emb_tan = self.textual_proj(self.textual(text_tokens)) if text_tokens is not None else torch.zeros_like(z_txt)

        
        dec_feat = self.decoder(spatial_feats[-1], text_emb_tan, z_cond=z_txt) 
        pred_map = self.bias_head(dec_feat, text_emb_tan, z_cond=z_txt)

        loss_dict = {"pred_map": pred_map}
        
        if gt_saliency is not None:
          
            task_loss = F.binary_cross_entropy_with_logits(pred_map, gt_saliency)
            
          
            all_z_img = torch.cat(dist.gather_across_processes(z_img), dim=0)
            all_z_txt = torch.cat(dist.gather_across_processes(z_txt), dim=0)
            
            with torch.autocast(self.device.type, dtype=torch.float32):
                logits = -L.pairwise_dist(z_img, all_z_txt, _curv) * self.curv.exp() # scale
            
            labels = torch.arange(B, device=self.device) + B * self._rank
            contrast_loss = F.cross_entropy(logits, labels)

       
            angle_ti = L.oxy_angle(z_txt, z_img, _curv)
            aperture_t = L.half_aperture(z_txt, _curv)
            loss_ti = torch.clamp(angle_ti - 0.8 * aperture_t, min=0).mean()

          
            angle_at = L.oxy_angle(z_anc, z_txt, _curv)
            aperture_a = L.half_aperture(z_anc, _curv)
            loss_at = torch.clamp(angle_at - 1.0 * aperture_a, min=0).mean()

            angle_ai = L.oxy_angle(z_anc, z_img, _curv)
            loss_ai = torch.clamp(angle_ai - 1.0 * aperture_a, min=0).mean()

            entailment_loss = (loss_ti + loss_at + loss_ai) / 3.0

            total_loss = task_loss + 0.1 * contrast_loss + self.entail_weight * entailment_loss
            
            loss_dict.update({
                "loss": total_loss,
                "loss_task": task_loss,
                "loss_contrast": contrast_loss,
                "loss_entail": entailment_loss
            })

        return loss_dict
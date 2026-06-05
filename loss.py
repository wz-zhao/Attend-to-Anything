import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import vgg19
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal as Norm
import cv2

def combined_loss(y_true, y_pred):
    kld_loss = kldiv(y_pred, y_true)
    cc_loss = cc(y_pred, y_true)
    
    sim_loss = similarity(y_pred, y_true)
    l1_loss = F.l1_loss(y_pred, y_true)    
    
    total_loss = kld_loss - cc_loss - sim_loss
    # total_loss = kld_loss - cc_loss - sim_loss
    return total_loss,kld_loss,cc_loss,sim_loss




def kldiv(s_map: torch.Tensor, gt: torch.Tensor, eps: float = 1e-12, debug: bool = False) -> torch.Tensor:

    assert s_map.shape == gt.shape, f"shape mismatch: {s_map.shape} vs {gt.shape}"

    added_channel = False
    if s_map.dim() == 3:          
        s_map = s_map.unsqueeze(1)
        gt    = gt.unsqueeze(1)
        added_channel = True
    elif s_map.dim() == 4:       
        pass
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {s_map.shape}")

    B, C, H, W = s_map.shape

    s_sum = s_map.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    g_sum = gt.sum(dim=(2, 3), keepdim=True).clamp_min(eps)

    s_norm = s_map / s_sum
    g_norm = gt   / g_sum

    if debug:
        print(f"s_map: {s_map.shape}, gt: {gt.shape}")
        print(f"s_sum: {s_sum.shape}, g_sum: {g_sum.shape}")
        print(f"s_norm: {s_norm.shape}, g_norm: {g_norm.shape}")


    kld_per_pix = g_norm * (g_norm.clamp_min(eps).log() - s_norm.clamp_min(eps).log())
    kld_per_chan = kld_per_pix.sum(dim=(2, 3))   
    kld_per_sample = kld_per_chan.sum(dim=1)    
    kld_mean = kld_per_sample.mean()            


    return kld_mean


def normalize_map(s_map):
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    min_s_map = torch.min(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)
    max_s_map = torch.max(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)

    norm_s_map = (s_map - min_s_map) / (max_s_map - min_s_map * 1.0)
    return norm_s_map


def similarity(s_map: torch.Tensor, gt: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:

    assert s_map.shape == gt.shape, f"shape mismatch: {s_map.shape} vs {gt.shape}"


    if s_map.dim() == 3:
        s_map = s_map.unsqueeze(1)
        gt    = gt.unsqueeze(1)
    elif s_map.dim() != 4:
        raise ValueError(f"Expected 3D or 4D tensor, got {s_map.shape}")


    s_map = s_map.clamp_min(0)
    gt    = gt.clamp_min(0)

    s_sum = s_map.sum(dim=(2, 3), keepdim=True).clamp_min(eps)  
    g_sum = gt.sum(dim=(2, 3), keepdim=True).clamp_min(eps)    
    s_norm = s_map / s_sum
    g_norm = gt   / g_sum

    sim_map = torch.minimum(s_norm, g_norm)     
    sim_per_chan = sim_map.sum(dim=(2, 3))    
    sim_per_sample = sim_per_chan.sum(dim=1)   
    sim_mean = sim_per_sample.mean()           

    return sim_mean  

def cc(s_map: torch.Tensor, gt: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:

    assert s_map.shape == gt.shape, f"shape mismatch: {s_map.shape} vs {gt.shape}"

    if s_map.dim() == 3:
        s_map = s_map.unsqueeze(1)
        gt    = gt.unsqueeze(1)
    elif s_map.dim() != 4:
        raise ValueError(f"Expected 3D or 4D tensor, got {s_map.shape}")


    mean_s = s_map.mean(dim=(2, 3), keepdim=True)
    mean_g = gt.mean(dim=(2, 3), keepdim=True)
    xs = s_map - mean_s
    yg = gt - mean_g


    var_s = (xs * xs).sum(dim=(2, 3), keepdim=True).clamp_min(eps)  
    var_g = (yg * yg).sum(dim=(2, 3), keepdim=True).clamp_min(eps)  
    cov   = (xs * yg).sum(dim=(2, 3), keepdim=True)                  


    cc_map = (cov / (var_s.sqrt() * var_g.sqrt())).squeeze(-1).squeeze(-1)  

   
    return cc_map.mean()  



import torch
import torch.nn.functional as F

def nss_loss(s_map, gt):
   
    if s_map.dim() == 4 and s_map.size(1) == 1:
        s_map = s_map.squeeze(1)
    if gt.dim() == 4 and gt.size(1) == 1:
        gt = gt.squeeze(1)

   
    gt = gt.to(device=s_map.device, dtype=s_map.dtype)
    if s_map.size() != gt.size():
        s_map = F.interpolate(
            s_map.unsqueeze(1),
            size=(gt.size(1), gt.size(2)),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)

    assert s_map.size() == gt.size(), f"s_map {tuple(s_map.size())} != gt {tuple(gt.size())}"

    batch_size, h, w = s_map.size(0), s_map.size(1), s_map.size(2)

    s_flat = s_map.view(batch_size, -1)
    mean_s = s_flat.mean(dim=1, keepdim=True).view(batch_size, 1, 1)
    std_s = s_flat.std(dim=1, keepdim=True).view(batch_size, 1, 1)

    eps = 2.2204e-16
    s_norm = (s_map - mean_s) / (std_s + eps)

    num = torch.sum((s_norm * gt).view(batch_size, -1), dim=1)
    den = torch.sum(gt.view(batch_size, -1), dim=1).clamp_min(1.0) 

    return torch.mean(num / den)


import time
import numpy as np
import torch
from skimage.transform import resize

def AUC_Judd_fast_exact(saliency_map, fixation_map, jitter=True):
   
    def to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().to("cpu").numpy()
        return np.asarray(x)


    saliency_map = to_numpy(saliency_map)
    fixation_map = to_numpy(fixation_map) > 0

    if not np.any(fixation_map):
        print('No fixations to predict')
        return np.nan

    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='reflect')

    
    if jitter:
        random_values = np.random.rand(*saliency_map.shape).astype(np.float64)
        saliency_map = saliency_map.astype(np.float64) + random_values * 1e-7
    else:
        saliency_map = saliency_map.astype(np.float64, copy=False)
    smin = np.min(saliency_map)
    smax = np.max(saliency_map)
    if smax == smin:
        saliency_map = np.zeros_like(saliency_map, dtype=np.float64)
    else:
        saliency_map = (saliency_map - smin) / (smax - smin)

    S = saliency_map.ravel()
    F = fixation_map.ravel()

    S_fix = S[F]         
    n_fix = int(len(S_fix))
    n_pixels = int(len(S))

    if n_fix == 0:
        return np.nan

    thresholds = np.sort(S_fix)[::-1]
    S_sorted_asc = np.sort(S) 
    left = np.searchsorted(S_sorted_asc, thresholds, side='left')  
    above_th = (n_pixels - left).astype(np.float64)

    tp = np.zeros(n_fix + 2, dtype=np.float64)
    fp = np.zeros(n_fix + 2, dtype=np.float64)
    tp[0], tp[-1] = 0.0, 1.0
    fp[0], fp[-1] = 0.0, 1.0

    k1 = np.arange(1, n_fix + 1, dtype=np.float64)  # (k+1)
    tp[1:-1] = k1 / float(n_fix)
    fp[1:-1] = (above_th - k1) / float(n_pixels - n_fix)

    return float(np.trapz(tp, fp))


def auc_judd(saliency_maps, fixation_maps, jitter=True, reduce="mean"):
   

    def ensure_bhw(x):
        if torch.is_tensor(x):
            if x.ndim == 2:
                x = x.unsqueeze(0)
            elif x.ndim == 4 and x.shape[1] == 1:
                x = x.squeeze(1)
        else:
            x = np.asarray(x)
            if x.ndim == 2:
                x = x[None, ...]
            elif x.ndim == 4 and x.shape[1] == 1:
                x = np.squeeze(x, axis=1)
        return x

    sm = ensure_bhw(saliency_maps)
    fm = ensure_bhw(fixation_maps)
    
    u = np.unique(fixation_maps.cpu().numpy())


    B = sm.shape[0]
    out = []
    for b in range(B):
        out.append(AUC_Judd_fast_exact(sm[b], fm[b], jitter=jitter))

    if reduce == "none":
        return out
    if reduce == "mean":
        return float(np.nanmean(out))
    raise ValueError(f"Unknown reduce={reduce}, use 'mean' or 'none'.")


def auc_shuff(s_map, gt, other_map, splits=100):

    s_map_batch = _to_numpy_batch(s_map)
    gt_batch = _to_numpy_batch(gt)
    other_batch = _to_numpy_batch(other_map)
    
    batch_size = s_map_batch.shape[0]
    scores = []

    for i in range(batch_size):
        curr_s = s_map_batch[i]
        curr_g = gt_batch[i]
        curr_other = other_batch[i] 
        curr_s = _normalize_map_numpy(curr_s)

        num_fixations = np.sum(curr_g > 0)
        if num_fixations == 0:
            scores.append(np.nan)
            continue
        x, y = np.where(curr_other > 0)
        other_map_fixs = []
        w = curr_other.shape[1]
        for r, c in zip(x, y):
            other_map_fixs.append(r * w + c)
        
        ind = len(other_map_fixs)
        if ind == 0:
            scores.append(np.nan)
            continue

        aucs = []
        for _ in range(splits):
            chosen_indices = np.random.choice(ind, int(num_fixations), replace=True)
            r_sal_vals = []
            for idx in chosen_indices:
                r, c = idx // w, idx % w
                if r < curr_s.shape[0] and c < curr_s.shape[1]:
                    r_sal_vals.append(curr_s[r, c])
            
            r_sal_vals = np.array(r_sal_vals)
            
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            tp_list = [0.0]
            fp_list = [0.0]

            for thresh in thresholds:
                temp = np.zeros_like(curr_s)
                temp[curr_s >= thresh] = 1.0
                num_overlap = np.sum(np.logical_and(temp == 1.0, curr_g > 0))
                tp = num_overlap / (num_fixations * 1.0)
                
                fp = len(np.where(r_sal_vals > thresh)[0]) / (len(r_sal_vals) * 1.0 + 1e-12)
                
                tp_list.append(tp)
                fp_list.append(fp)
            
            tp_list.append(1.0)
            fp_list.append(1.0)
            
            pairs = sorted(zip(fp_list, tp_list), key=lambda x: x[0])
            fp_sorted = [p[0] for p in pairs]
            tp_sorted = [p[1] for p in pairs]
            
            aucs.append(np.trapz(tp_sorted, fp_sorted))
        
        scores.append(np.mean(aucs))

    return np.nanmean(scores)


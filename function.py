import torch
import torch.nn as nn
from tqdm import tqdm
import lorentz as L 
from loss import combined_loss, auc_judd, nss_loss
from itertools import zip_longest
import torch.nn.functional as F
import clip
import cv2
from PIL import Image
import numpy as np


def blur_cv2_torch(x: torch.Tensor, k_size: int = 11) -> torch.Tensor:

    device = x.device
    x_cpu = x.detach().float().cpu()
    if x_cpu.dim() == 2:
        x4 = x_cpu.unsqueeze(0).unsqueeze(0)    
    elif x_cpu.dim() == 3:
        x4 = x_cpu.unsqueeze(0)                 
    elif x_cpu.dim() == 4:
        x4 = x_cpu
    else:
        raise ValueError(f"Unsupported shape: {tuple(x_cpu.shape)}")

    n, c, h, w = x4.shape
    x_np = x4.numpy()  

    out = np.empty_like(x_np, dtype=np.float32)
    for i in range(n):
        for j in range(c):
            out[i, j] = cv2.GaussianBlur(x_np[i, j], (k_size, k_size), 0)

    y = torch.from_numpy(out) 
    if x.dim() == 2:
        y = y[0, 0]
    elif x.dim() == 3:
        y = y[0]
    return y.to(device=device)


def _get_prompt_embed(prompt_embeds: dict, dataset_name: str, batch_size: int, device):

    if prompt_embeds is None:
        return None
    if dataset_name not in prompt_embeds:
        dataset_name = "None"
    emb = prompt_embeds[dataset_name]
    if emb.device != device:
        emb = emb.to(device)
    return emb.expand(batch_size, -1)


def train_one_epoch_alternating(
    net: nn.Module,
    optimizer,
    train_loader_image,
    train_loader_video,
    device,
    epoch,
    clip_model=None,      
    audio_backbone=None,    
    prompt_embeds: dict = None,     
    amp_dtype: torch.dtype = torch.bfloat16, 
):
    net.train()
    if audio_backbone is not None:
        audio_backbone.eval()


    epoch_loss_img = 0.0
    epoch_loss_vid = 0.0
    n_img = 0
    n_vid = 0

    total_pairs = max(len(train_loader_image), len(train_loader_video))
    
    pbar = tqdm(
        zip_longest(train_loader_image, train_loader_video, fillvalue=None),
        total=total_pairs,
        desc=f"Epoch {epoch} [Train ALT]",
        unit="pair"
    )

    for batch_pair in pbar:
        batch_img, batch_vid = batch_pair
        if batch_img is not None:
            i = batch_img
            imgs = i["image"].to(device, non_blocking=True).squeeze(0)
            masks = i["saliency"].to(device, non_blocking=True).squeeze(0)
            dataset_name = i["label"][0]
            current_batch_size = imgs.shape[0]


            text_emb = _get_prompt_embed(prompt_embeds, dataset_name, current_batch_size, device)

            optimizer.zero_grad(set_to_none=True)


            with torch.autocast(device_type="npu", dtype=amp_dtype):
                pred_logits, hyp = net(imgs, text_emb=text_emb, dynamic=False)
                loss_task, kld, cc, sim = combined_loss(y_pred=pred_logits, y_true=masks)


            z_img = hyp["z_img"].float()
            z_txt = hyp["z_txt"].float()
            z_anc = hyp["z_anc"].float()
            curv  = hyp["curv"]

            # --- contrast ---
            if text_emb is not None:
                dist = L.pairwise_dist(z_img, z_txt, curv)
                loss_contrast = torch.relu(dist.diag()).mean()
            else:
                loss_contrast = torch.tensor(0.0, device=pred_logits.device)

            # --- entailment: Anchor(None) ⊃ Text ⊃ Image ---
            if text_emb is not None:
                angle_ti = L.oxy_angle(z_txt, z_img, curv)
                aperture_t = L.half_aperture(z_txt, curv)
                loss_ti = torch.clamp(angle_ti - 0.8 * aperture_t, min=0).mean()

                angle_at = L.oxy_angle(z_anc, z_txt, curv)
                aperture_a = L.half_aperture(z_anc, curv)
                loss_at = torch.clamp(angle_at - 1.0 * aperture_a, min=0).mean()

                loss_entail = 0.5 * (loss_ti + loss_at)
            else:
                loss_entail = torch.tensor(0.0, device=pred_logits.device)


            lambda_contrast = 0.4
            lambda_entail   = 1.0
            loss = loss_task + lambda_contrast * loss_contrast + lambda_entail * loss_entail

            loss.backward()
            optimizer.step()

            epoch_loss_img += float(loss.detach().cpu())
            n_img += 1

  
        if batch_vid is not None:
            i = batch_vid
            imgs = i["image"].to(device, non_blocking=True).squeeze(0)
            masks = i["saliency"].to(device, non_blocking=True).squeeze(0)
            dataset_name = i["label"][0]
            current_num_frames = imgs.shape[0]

            audio_emb = None
            if "audio" in i and audio_backbone is not None:
                # batch['audio']: (1, T, 1, 96000) -> squeeze(0) -> (T, 1, 96000)
                raw_audio = i["audio"].to(device, non_blocking=True).squeeze(0)
                with torch.no_grad():
                    audio_emb = audio_backbone(raw_audio)


            current_batch_size = imgs.shape[0]
            text_emb = _get_prompt_embed(prompt_embeds, dataset_name, current_batch_size, device)
            optimizer.zero_grad(set_to_none=True)

   
            with torch.autocast(device_type="npu", dtype=amp_dtype):
                pred_logits, hyp = net(imgs, text_emb=text_emb,audio_emb = audio_emb, dynamic=True)
                loss_task, kld, cc, sim = combined_loss(y_pred=pred_logits, y_true=masks)


            z_img = hyp["z_img"].float()
            z_txt = hyp["z_txt"].float()
            z_anc = hyp["z_anc"].float()
            curv  = hyp["curv"]
            
            
       
            if text_emb is not None:
                dist = L.pairwise_dist(z_img, z_txt, curv)
                loss_contrast = torch.relu(dist.diag()).mean()
            else:
                loss_contrast = torch.tensor(0.0, device=pred_logits.device)


            if text_emb is not None:
                angle_ti = L.oxy_angle(z_txt, z_img, curv)
                aperture_t = L.half_aperture(z_txt, curv)
                loss_ti = torch.clamp(angle_ti - 0.8 * aperture_t, min=0).mean()

                angle_at = L.oxy_angle(z_anc, z_txt, curv)
                aperture_a = L.half_aperture(z_anc, curv)
                loss_at = torch.clamp(angle_at - 1.0 * aperture_a, min=0).mean()
                loss_entail = 0.5 * (loss_ti + loss_at)
            else:
                loss_entail = torch.tensor(0.0, device=pred_logits.device)

        
            lambda_contrast = 0.4
            lambda_entail   = 1.0
            loss = loss_task + lambda_contrast * loss_contrast + lambda_entail * loss_entail

            loss.backward()
            optimizer.step()

            epoch_loss_vid += float(loss.detach().cpu())
            n_vid += 1

        last_loss = 0.0
        if batch_vid is not None:
            last_loss = float(loss.detach().cpu())
        elif batch_img is not None:
            last_loss = float(loss.detach().cpu())

        pbar.set_postfix(loss=f"{last_loss:.4f}")


    avg_img_loss = epoch_loss_img / max(1, n_img)
    avg_vid_loss = epoch_loss_vid / max(1, n_vid)

    return avg_img_loss, avg_vid_loss





def train_one_epoch(
    net: nn.Module,
    optimizer,
    train_loader,
    device,
    epoch,
    clip_model=None, 
    audio_backbone=None,
    dynamic: bool = False,
    prompt_embeds: dict = None,
    amp_dtype: torch.dtype = torch.bfloat16, 
):
    net.train()
    if audio_backbone is not None:
        audio_backbone.eval()

    epoch_loss = 0.0
    epoch_kld, epoch_cc, epoch_sim = 0.0, 0.0, 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", unit="batch")
    for batch in pbar:
        imgs = batch["image"].to(device, non_blocking=True).squeeze(0)
        masks = batch["saliency"].to(device, non_blocking=True).squeeze(0)
        
        current_num_frames = imgs.shape[0]

        audio_emb = None
        if "audio" in batch and audio_backbone is not None:
            # batch['audio']: (1, T, 1, 96000) -> squeeze(0) -> (T, 1, 96000)
            raw_audio = batch["audio"].to(device, non_blocking=True).squeeze(0)
            with torch.no_grad():
                audio_emb = audio_backbone(raw_audio)

                
        dataset_name = batch["label"][0]
        B = masks.shape[0]
        text_emb = _get_prompt_embed(prompt_embeds, dataset_name, B, device)

        optimizer.zero_grad(set_to_none=True)
        audio_emb = None


        with torch.autocast(device_type="npu", dtype=amp_dtype):
            pred_logits, hyp = net(imgs, text_emb=text_emb,audio_emb = audio_emb, dynamic=dynamic)

            loss_task, kld, cc, sim = combined_loss(y_pred=pred_logits, y_true=masks)


        z_img = hyp["z_img"].float()
        z_txt = hyp["z_txt"].float()
        z_anc = hyp["z_anc"].float()
        curv = hyp["curv"]  

        if text_emb is not None:
            dist = L.pairwise_dist(z_img, z_txt, curv)
            loss_contrast = torch.relu(dist.diag()).mean()

            angle_ti = L.oxy_angle(z_txt, z_img, curv)
            aperture_t = L.half_aperture(z_txt, curv)
            loss_ti = torch.clamp(angle_ti - 0.8 * aperture_t, min=0).mean()

            angle_at = L.oxy_angle(z_anc, z_txt, curv)
            aperture_a = L.half_aperture(z_anc, curv)
            loss_at = torch.clamp(angle_at - 1.0 * aperture_a, min=0).mean()

            loss_entail = 0.5 * (loss_ti + loss_at)
        else:
            loss_contrast = torch.tensor(0.0, device=device)
            loss_entail = torch.tensor(0.0, device=device)

        loss = loss_task + 0.4 * loss_contrast + 1.0 * loss_entail
        loss.backward()
        optimizer.step()

        epoch_loss += float(loss.detach().cpu())
        epoch_kld += float(kld.detach().cpu())
        epoch_cc += float(cc.detach().cpu())
        epoch_sim += float(sim.detach().cpu())

        pbar.set_postfix(
            loss=f"{float(loss.detach().cpu()):.4f}",
            cc=f"{float(cc.detach().cpu()):.4f}",
        )

    avg_loss = epoch_loss / max(1, len(train_loader))
    return avg_loss




@torch.no_grad()
def validate_saliency(
    net: nn.Module,
    val_loader,
    device,
    epoch,
    clip_model=None,
    audio_backbone=None,
    prompt_text=None,
    dynamic: bool = False,
    prompt_embeds: dict = None,
    dataset_name: str = None,
    amp_dtype: torch.dtype = torch.bfloat16,
):
    net.eval()

    epoch_loss = 0.0
    epoch_cc, epoch_sim, epoch_kld, epoch_nss, epoch_auc = 0.0, 0.0, 0.0, 0.0, 0.0

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", unit="batch", leave=False)
    for batch in pbar:
        imgs = batch["image"].to(device, non_blocking=True).squeeze(0)
        masks = batch["saliency"].to(device, non_blocking=True).squeeze(0) 
      
        current_num_frames = imgs.shape[0]

        audio_emb = None
        if "audio" in batch and audio_backbone is not None:
            raw_audio = batch["audio"].to(device, non_blocking=True).squeeze(0)
            with torch.no_grad():
                audio_emb = audio_backbone(raw_audio)

        dname = dataset_name if dataset_name is not None else batch["label"][0]

        
        B = masks.shape[0]
        text_emb = _get_prompt_embed(prompt_embeds, dname, B, device)
        
        with torch.autocast(device_type="npu", dtype=amp_dtype):
            
            pred_logits, _ = net(imgs, text_emb=text_emb, audio_emb = audio_emb, dynamic=dynamic)
            
            gt_h, gt_w = masks.shape[-2], masks.shape[-1]

            if pred_logits.shape[-2:] != (gt_h, gt_w):
                pred_logits = F.interpolate(
                    pred_logits, size=(gt_h, gt_w),
                    mode="bilinear", align_corners=False
                )
            if dynamic:
                pred_logits = blur_cv2_torch(pred_logits, k_size=11) 


            if isinstance(batch["fixation"], dict):
                batch["fixation"] = batch["fixation"]["fixmap"].squeeze(0)
            else:
                batch["fixation"] = batch["fixation"].squeeze(0)
 
            loss, kld, cc, sim = combined_loss(y_pred=pred_logits, y_true=masks)
            # auc = auc_judd(pred_logits,batch["fixation"])
            # nss = nss_loss(pred_logits,batch["fixation"])
            
        
        epoch_loss += float(loss.detach().cpu())
        epoch_cc += float(cc.detach().cpu())
        epoch_sim += float(sim.detach().cpu())
        epoch_kld += float(kld.detach().cpu())
        epoch_nss += float(nss.detach().cpu())
        
        # if isinstance(auc, torch.Tensor):
        #     epoch_auc += auc.detach().float().cpu().item()
        # else:
        #     epoch_auc += float(auc)

        pbar.set_postfix(
            loss=f"{float(loss.detach().cpu()):.4f}",
            cc=f"{float(cc.detach().cpu()):.4f}",
            sim=f"{float(sim.detach().cpu()):.4f}",
        )

    avg_loss = epoch_loss / max(1, len(val_loader))
    avg_cc = epoch_cc / max(1, len(val_loader))
    avg_sim = epoch_sim / max(1, len(val_loader))
    avg_kld = epoch_kld / max(1, len(val_loader))
    avg_nss = epoch_nss / max(1, len(val_loader))
    avg_auc = epoch_auc / max(1, len(val_loader))
    
    return avg_loss, avg_cc, avg_sim, avg_nss, avg_kld, avg_auc





def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _to_str_path(x):
    import os as _os
    if x is None:
        return ""
    if isinstance(x, _os.PathLike):
        return _os.fspath(x)
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return ""
        return _to_str_path(x[0])
    return str(x)


def _normalize_frame_paths(frame_paths):
    if isinstance(frame_paths, list) and len(frame_paths) == 1 and isinstance(frame_paths[0], (list, tuple)):
        return [_to_str_path(p) for p in frame_paths[0]]
    if isinstance(frame_paths, (list, tuple)):
        return [_to_str_path(p) for p in frame_paths]
    return [_to_str_path(frame_paths)]


def _img_path_to_pred_path(img_path: str) -> str:
    img_path = img_path.replace("\\", "/")
    if "/images/" in img_path:
        return img_path.replace("/images/", "/pred/", 1)
    return img_path



def _save_1ch_png_prob(
    pred_1hw: torch.Tensor,
    save_path: str,
    normalize: bool = True,
):


    x = pred_1hw.detach().float()

    if x.dim() == 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.dim() != 2:
        raise ValueError(f"Expect (1,H,W) or (H,W), got {tuple(x.shape)}")


    if normalize:
        minv = x.min()
        maxv = x.max()
        if (maxv - minv) > 1e-8:
            x = (x - minv) / (maxv - minv)
        else:
            x = torch.zeros_like(x)
    x = torch.round(x * 255.0 + 0.5).clamp(0, 255).to(torch.uint8)

    arr = x.cpu().numpy()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(arr, mode="L").save(save_path)




# -------------------------
# Save function
# -------------------------
@torch.no_grad()
def save_saliency(
    net: nn.Module,
    val_loader,
    device,
    epoch,
    clip_model=None,
    audio_backbone=None,
    prompt_text=None,
    dynamic: bool = False,
    prompt_embeds: dict = None,
    dataset_name: str = None,
    amp_dtype: torch.dtype = torch.bfloat16,
):
    net.eval()

    if amp_dtype not in (torch.float16, torch.bfloat16):
        amp_dtype = torch.bfloat16

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val-Save]", unit="batch", leave=False)

    for batch in pbar:
        imgs = batch["image"].to(device, non_blocking=True).squeeze(0)
   
        dname = dataset_name if dataset_name is not None else (
            batch["label"][0] if isinstance(batch["label"], (list, tuple)) else batch["label"]
        )
        B = imgs.shape[0]
        text_emb = _get_prompt_embed(prompt_embeds, dname, B, device)
        
        with torch.autocast(device_type="npu", dtype=amp_dtype):
            
            pred_logits, _ = net(imgs, text_emb=text_emb, audio_emb = None,dynamic=True)
            gt_h, gt_w = int(360), int(640)

            if pred_logits.shape[-2:] != (gt_h, gt_w):
                pred_logits = F.interpolate(
                    pred_logits, size=(gt_h, gt_w),
                    mode="bilinear", align_corners=False
                )
        if dynamic:
                pred_logits = blur_cv2_torch(pred_logits, k_size=11)
                
                
        frame_paths = _normalize_frame_paths(batch["frame_paths"])
        for t in range(B):
            img_path = _to_str_path(frame_paths[t])
            save_path = _img_path_to_pred_path(img_path)

            _save_1ch_png_prob(pred_logits[t], save_path)

    return {}



@torch.no_grad()
def save_saliency_image(
    net: nn.Module,
    val_loader,
    device,
    epoch,
    clip_model=None,
    audio_backbone=None,
    prompt_text=None,
    dynamic: bool = False,
    prompt_embeds: dict = None,
    dataset_name: str = None,
    amp_dtype: torch.dtype = torch.bfloat16,
    save_predictions: bool = True,  
    save_dir: str = "./predictions_image"  
):
    net.eval()
    
    if save_predictions and save_dir:
        os.makedirs(save_dir, exist_ok=True)

    epoch_loss = 0.0
    epoch_cc, epoch_sim, epoch_kld, epoch_nss, epoch_auc = 0.0, 0.0, 0.0, 0.0, 0.0

    if amp_dtype not in (torch.float16, torch.bfloat16):
        amp_dtype = torch.bfloat16

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val & Save]", unit="batch", leave=False)
    
    for batch in pbar:
        imgs = batch["image"].to(device, non_blocking=True).squeeze(0)     
        masks = batch["saliency"].to(device, non_blocking=True).squeeze(0) 
        
        audio_emb = None
        if "audio" in batch and audio_backbone is not None:
            raw_audio = batch["audio"].to(device, non_blocking=True).squeeze(0)
            with torch.no_grad():
                audio_emb = audio_backbone(raw_audio)

        dname = dataset_name if dataset_name is not None else (
            batch["label"][0] if isinstance(batch["label"], (list, tuple)) else batch["label"]
        )
        B = masks.shape[0] 
        text_emb = _get_prompt_embed(prompt_embeds, dname, B, device)

        with torch.autocast(device_type="npu", dtype=amp_dtype):
            pred_logits, _ = net(imgs, text_emb=text_emb, audio_emb=audio_emb, dynamic=dynamic)
            
            gt_h, gt_w = masks.shape[-2], masks.shape[-1]

            if pred_logits.shape[-2:] != (gt_h, gt_w):
                pred_logits = F.interpolate(
                    pred_logits, size=(gt_h, gt_w),
                    mode="bilinear", align_corners=False
                )
            
            if dynamic:
                pred_logits = blur_cv2_torch(pred_logits, k_size=11) 

            if save_predictions:
                if "im_path" in batch:
                    frame_paths = _normalize_frame_paths(batch["im_path"])
                 
                    
                    for t in range(B):
                        img_path_str = _to_str_path(frame_paths[t])
                        file_name = os.path.basename(img_path_str)
                        file_name_no_ext = os.path.splitext(file_name)[0]
                        save_name = file_name_no_ext + ".png"
                        save_path = os.path.join(save_dir, str(dname), save_name)

                        _save_1ch_png_prob(pred_logits[t], save_path)
                        
            if isinstance(batch["fixation"], dict):
                batch["fixation"] = batch["fixation"]["fixmap"].squeeze(0)
            else:
                batch["fixation"] = batch["fixation"].squeeze(0)

            loss, kld, cc, sim = combined_loss(y_pred=pred_logits, y_true=masks)
            
            # auc = auc_judd(pred_logits, batch["fixation"])
            # nss = nss_loss(pred_logits, batch["fixation"])
        
        epoch_loss += float(loss.detach().cpu())
        epoch_cc += float(cc.detach().cpu())
        epoch_sim += float(sim.detach().cpu())
        epoch_kld += float(kld.detach().cpu())
        
        pbar.set_postfix(
            loss=f"{float(loss.detach().cpu()):.4f}",
            cc=f"{float(cc.detach().cpu()):.4f}",
            sim=f"{float(sim.detach().cpu()):.4f}",
        )


    num_batches = max(1, len(val_loader))
    avg_loss = epoch_loss / num_batches
    avg_cc = epoch_cc / num_batches
    avg_sim = epoch_sim / num_batches
    avg_kld = epoch_kld / num_batches
    avg_nss = epoch_nss / num_batches
    avg_auc = epoch_auc / num_batches
    
    return avg_loss, avg_cc, avg_sim, avg_nss, avg_kld, avg_auc
import sys
import os
import argparse
import random
import time
import numpy as np
from audio_backbone import Wav2CLIPAudioEncoder
import torch
import torch.distributed as dist
import torch.optim as optim

import sys
import os
import datetime

import torch.nn as nn

from ptflops import get_model_complexity_info

try:
    import torch_npu  
    _HAS_NPU = True
except Exception:
    _HAS_NPU = hasattr(torch, "npu")

from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from dataset_video import DHF1KDataset, Hollywood_UCFDataset, LEDOVDataset
from dataloader_all import (
    get_train_loader_image,
    get_val_loaders_image,
    get_train_loader_video,
    get_val_loader_video
)
from function import train_one_epoch, validate_saliency, train_one_epoch_alternating, save_saliency, save_saliency_image
from dpt import DPT
from LORA import apply_lora_dino_backbone,apply_hyper_lora_dino_backbone
from loss import combined_loss
from prompts import get_default_prompt_texts
import training_utils as tu

import clip
import torchaudio
import torchaudio.functional as F


import warnings, traceback, os

def _warn_with_stack(message, category, filename, lineno, file=None, line=None):
    msg = str(message)
    if "npu autocast" in msg and "target dtype is not supported" in msg:
        print(f"\n[WARN] {filename}:{lineno}: {category.__name__}: {message}\n")
        traceback.print_stack(limit=25)


_original_spectrogram = F.spectrogram

def _npu_safe_spectrogram(waveform, pad, window, n_fft, hop_length, win_length, power, normalized, center=True, pad_mode="reflect", onesided=True, return_complex=None):
    original_shape = waveform.shape
    if waveform.ndim > 2:
        waveform = waveform.reshape(-1, original_shape[-1])
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 4

    spec_f = torch.stft(
        waveform, 
        n_fft, 
        hop_length, 
        win_length, 
        window, 
        center, 
        pad_mode, 
        normalized, 
        onesided, 
        return_complex=True 
    )

    if power is not None:
        spec_view = torch.view_as_real(spec_f) 
        pow_spec = spec_view.pow(2).sum(-1) 
        
        if power == 2.0:
            result = pow_spec
        else:
            result = (pow_spec + 1e-6).pow(power / 2.0)
    else:
        result = spec_f


    if len(original_shape) > 2:
        new_shape = original_shape[:-1] + result.shape[-2:]
        result = result.reshape(new_shape)
            
    return result


print("[INFO] Patching torchaudio.functional.spectrogram for NPU compatibility (with 3D input support)...")
torchaudio.functional.spectrogram = _npu_safe_spectrogram

def _set_cpu_thread_env():
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def reduce_tensor(t: torch.Tensor) -> torch.Tensor:
    t = t.to(dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return t


def setup_distributed_npu():

    if "RANK" not in os.environ or os.environ.get("WORLD_SIZE", "1") == "1":
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

    backend = "hccl" 
    dist.init_process_group(backend=backend)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    torch.npu.set_device(local_rank)
    return local_rank


def freeze_all_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def set_backbone_trainable_best_strategy(backbone: nn.Module):
    print("[Backbone Freeze] BEST strategy")
    freeze_all_params(backbone)

    blocks = getattr(backbone, "blocks", None)
    num_blocks = len(blocks)

    if num_blocks == 24:
        unfreeze_indices = list(range(16, 24))
    elif num_blocks == 12:
        unfreeze_indices = list(range(8, 12))
    else:
        stage_size = num_blocks // 4
        unfreeze_indices = list(range(2 * stage_size, num_blocks))

    for idx in unfreeze_indices:
        for p in blocks[idx].parameters():
            p.requires_grad = True

    if hasattr(backbone, "norm"):
        for p in backbone.norm.parameters():
            p.requires_grad = True

def infer_dinov3_variant(sd):
    if "cls_token" in sd:
        dim = sd["cls_token"].shape[-1]
    else:
        dim = None

    return {
        384: "vits16",
        768: "vitb16",
        1024: "vitl16"
    }.get(dim, None)


def load_dinov3_backbone(repo_dir, dino_ckpt, dino_size_arg):
    state_dict = torch.load(dino_ckpt, map_location="cpu")
    inferred = infer_dinov3_variant(state_dict)

    user_map = {"s": "vits16", "b": "vitb16", "l": "vitl16"}
    arch = inferred if inferred else user_map[dino_size_arg]

    backbone = torch.hub.load(
        repo_dir,
        f"dinov3_{arch}",
        source="local",
        pretrained=False,
        use_moe=False,
        text_dim=768
    )
    backbone.load_state_dict(state_dict, strict=False)
    return backbone


def load_checkpoint_compat(model, resume_path):
    if resume_path is None or (not os.path.exists(resume_path)):
        return

    if dist.get_rank() == 0:
        print(f"Loading weights from: {resume_path}")

    checkpoint = torch.load(resume_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    if dist.get_rank() == 0:
        print(f"Weights loaded. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")


@torch.no_grad()
def build_prompt_embeds(clip_model, prompt_texts: dict, device: torch.device):
    clip_model.eval()
    prompt_embeds = {}
    for k, text in prompt_texts.items():
        tokens = clip.tokenize([text]).to(device)
        emb = clip_model.encode_text(tokens)
        prompt_embeds[k] = emb
    return prompt_embeds

def freeze_lora_params(model: nn.Module, keywords=("lora", "lora_", "lora_a", "lora_b")):
    num = 0
    for name, p in model.named_parameters():
        lname = name.lower()
        if any(k in lname for k in keywords):
            if p.requires_grad:
                p.requires_grad = False
                num += 1
    if dist.get_rank() == 0:
        print(f"[Freeze LoRA] frozen {num} LoRA parameter tensors.")



class Logger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a", buffering=1) 

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def build_grouped_optimizer(model, base_lr, weight_decay):

    head_params = []
    lora_params = []
    router_params = []
    
    trainable_param_names = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        trainable_param_names.append(name)
        
        if "router" in name:
            router_params.append(p)
        elif "lora" in name: 
            lora_params.append(p)
        else:
            head_params.append(p)

    if dist.get_rank() == 0:
        print(f"[Optimizer] Detected {len(trainable_param_names)} trainable tensors.")
        print(f"   - Head/Motion params: {len(head_params)}")
        print(f"   - LoRA params: {len(lora_params)}")
        print(f"   - Router params: {len(router_params)}")

    param_groups = [
        {"params": head_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": lora_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": router_params, "lr": base_lr , "weight_decay": weight_decay * 0.1}, 
    ]
    
    optimizer = optim.AdamW(param_groups)
    return optimizer


def main(args):
    tu.set_cpu_thread_env()

    local_rank = tu.setup_distributed_npu()
    device = torch.device(f"npu:{local_rank}")


    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


    if dist.get_rank() == 0:
        print(f"[DDP] world_size={dist.get_world_size()} backend={dist.get_backend()}")
    print(f"[rank {dist.get_rank()}] local_rank={local_rank} npu_device={torch.npu.current_device()} device={device}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    save_root = args.save_root
    ckpt_dir = os.path.join(save_root, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    backbone = tu.load_dinov3_backbone(args.repo_dir, args.dino_ckpt, args.dino_size)
    model = DPT(nclass=1, backbone=backbone)
    
    apply_lora_dino_backbone(
        model,
        r=32, alpha=64, dropout=0.05,
        last_n_blocks=24,  
        target=("attn.qkv", "attn.proj"), 
    )
    
#     apply_lora_dino_backbone(
#         model,
#         r=16, alpha=32, dropout=0.05,
#         last_n_blocks=8,  
#         target=("attn.qkv", "attn.proj"),
    
    
    print("Configuring training strategy: Frozen Backbone + MoE LoRA...")
    trainable_keywords = ['motion_module', 'head', 'lora', 'router']
    
    trainable_count = 0
    total_count = 0
    
    for name, p in model.named_parameters():
        p.requires_grad = False 
        if any(k in name for k in trainable_keywords):
            p.requires_grad = True
            trainable_count += p.numel()
        
        total_count += p.numel()

    if dist.get_rank() == 0:
        print(f"[Strategy] Total Params: {total_count/1e6:.2f}M")
        print(f"[Strategy] Trainable Params: {trainable_count/1e6:.2f}M (Ratio: {trainable_count/total_count:.2%})")
        print(f"[Strategy] Active Modules: {trainable_keywords}")




    if args.resume_path is not None and os.path.exists(args.resume_path):
        if dist.get_rank() == 0:
            print(f"Loading weights from: {args.resume_path}")

        checkpoint = torch.load(args.resume_path, map_location='cpu')

        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v  
            
        msg = model.load_state_dict(new_state_dict, strict=False)
        if dist.get_rank() == 0:
            print(f"Weights loaded. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
    
        if dist.get_rank() == 0:
            print(f"Weights loaded.")
            print(f"  Missing keys ({len(msg.missing_keys)}):")
            for k in msg.missing_keys:
                print(f"    - {k}")

            print(f"  Unexpected keys ({len(msg.unexpected_keys)}):")
            for k in msg.unexpected_keys:
                print(f"    - {k}")




    if dist.get_rank() == 0:
        print("\nParameters to be trained:")
        total_trainable_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
                total_trainable_params += param.numel()
        print(f"\nTotal trainable parameters: {total_trainable_params}\n")

    model.to(device)

    print("Loading CLIP for Prompt-MoE...")
    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model.eval()
    text_dim = 768 
    for p in clip_model.parameters():
        p.requires_grad = False
        

    
    prompt_texts = get_default_prompt_texts()

    prompt_embeds = tu.build_prompt_embeds(clip_model, prompt_texts, device)

    optimizer = tu.build_grouped_optimizer(model, base_lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10,
        eta_min=1e-5,
        last_epoch=-1
    )


    optimizer.zero_grad()

    ####################################################IMAGE##########################################################################

    train_loader_image, train_sampler_image = get_train_loader_image(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
    )
    val_loader_image = get_val_loaders_image(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
    )

    ####################################################VIDEO##########################################################################
    torch.autograd.set_detect_anomaly(False)

    train_loader_video, train_sampler_video = get_train_loader_video(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
    )
    val_loader_video = get_val_loader_video(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
    )


    log_file_path = os.path.join(save_root, "train_val_log.txt")
    if dist.get_rank() == 0:
        if not os.path.exists(log_file_path):
            with open(log_file_path, "w") as f:
                f.write("Epoch\tDataset\tTrain_Loss\tVal_Loss\tVal_CC\tVal_SIM\n")

    best_val_cc = -1.0
    best_val_cc_UCF = -1.0
    best_val_cc_SALICON = -1.0

    amp_dtype = torch.bfloat16
    imagebind = Wav2CLIPAudioEncoder(device=device, freeze=False)
    imagebind.eval()
    
    
    viz = None
    for epoch in range(1, args.epochs + 1):

#         train_loss_image = train_one_epoch(
#              model, optimizer, train_loader_image, device, epoch,
#              clip_model=clip_model , dynamic=False,prompt_embeds=prompt_embeds, amp_dtype=amp_dtype)
        
#         train_loss_video = train_one_epoch(
#              model, optimizer, train_loader_video, device, epoch,
#              clip_model=clip_model, audio_backbone=imagebind , dynamic=True,  prompt_embeds=prompt_embeds, amp_dtype=amp_dtype)

        
        train_loss_image, train_loss_video = train_one_epoch_alternating(
            model,
            optimizer,
            train_loader_image,
            train_loader_video,
            device,
            epoch,
            clip_model=clip_model,
            audio_backbone=imagebind,
            prompt_embeds=prompt_embeds,
            amp_dtype=amp_dtype
        )

        train_loss_image = tu.reduce_tensor(torch.tensor(train_loss_image).to(device)).item()
        train_loss_video = tu.reduce_tensor(torch.tensor(train_loss_video).to(device)).item()
        scheduler.step()

        if dist.get_rank() == 0:
            print(f"Epoch {epoch} | Train Loss Image: {train_loss_image:.4f}")
        if dist.get_rank() == 0:
            print(f"Epoch {epoch} | Train Loss Video: {train_loss_video:.4f}")

        if epoch % 2 == 0 and epoch != 0:
            with torch.no_grad():
                # for dataset_name, loader in val_loader_video.items():
                #     # print(f"Validating on {dataset_name}...")
                #     current_prompt = prompt_texts.get(dataset_name, "Default saliency prompt")
                #     # print(current_prompt)
                #     val_loss = save_saliency(
                #         model, loader, device, epoch,
                #         clip_model=clip_model, prompt_text=current_prompt, dynamic=True, prompt_embeds=prompt_embeds,
                #         dataset_name=dataset_name,
                #         amp_dtype=amp_dtype,)
            ####################################################VIDEO##########################################################################
                for dataset_name, loader in val_loader_video.items():
                    print(f"Validating on {dataset_name}...")
                    current_prompt = prompt_texts.get(dataset_name, "Default saliency prompt")
                    # print(current_prompt)
                    val_loss, val_cc, val_sim, val_nss, val_kld, val_auc = validate_saliency(
                        model, loader, device, epoch,
                        clip_model=clip_model,audio_backbone=imagebind, prompt_text=current_prompt, dynamic=True, prompt_embeds=prompt_embeds,
                        dataset_name=dataset_name,
                        amp_dtype=amp_dtype,
                    )
           
                    val_loss = tu.reduce_tensor(torch.tensor(val_loss).to(device)).item()
                    val_cc = tu.reduce_tensor(torch.tensor(val_cc).to(device)).item()
                    val_sim = tu.reduce_tensor(torch.tensor(val_sim).to(device)).item()
                    # val_nss = reduce_tensor(torch.tensor(val_nss).to(device)).item()
                    # val_auc = reduce_tensor(torch.tensor(val_auc).to(device)).item()
                    val_kld = tu.reduce_tensor(torch.tensor(val_kld).to(device)).item()
                    
                    if dist.get_rank() == 0:
                        print(f'################### {dataset_name} ###################')
                        print(f" Val Loss: {val_loss:.4f} | CC: {val_cc:.4f} | SIM: {val_sim:.4f} | KLD: {val_kld:.4f} | NSS: {val_nss:.4f}")
#                         print(f" AUC: {val_auc:.4f}")


                        with open(log_file_path, "a") as f:
                                                                             
                            log_str = f"{epoch}\t{dataset_name}\t{train_loss_video:.4f}\t{val_loss:.4f}\t{val_cc:.4f}\t{val_sim:.4f}\n"
                            f.write(log_str)
                            
                    if dataset_name == "DHF1K":
                        if val_cc > best_val_cc:
                            best_val_cc = val_cc
                            best_path = os.path.join(ckpt_dir, f"best_model_cc_DHF1K{val_cc:.4f}.pth")
                            torch.save(model.state_dict(), best_path)
                            print(f"\n[Save] New best checkpoint saved to: {best_path} (DHF_CC: {val_cc:.4f})\n")
                    if dataset_name == "UCF":
                        if val_cc > best_val_cc_UCF:
                            best_val_cc_UCF = val_cc
                            best_path = os.path.join(ckpt_dir, f"best_model_cc_UCF{val_cc:.4f}.pth")
                            torch.save(model.state_dict(), best_path)
                            print(f"\n[Save] New best checkpoint saved to: {best_path} (UCF_CC: {val_cc:.4f})\n")
                ###################################################IMAGE##########################################################################
                for dataset_name, loader in val_loader_image.items():
                    print(f"Validating on {dataset_name}...")
                    current_prompt = prompt_texts.get(dataset_name, "Default saliency prompt")
                    
                           
                    val_loss, val_cc, val_sim, val_nss, val_kld, val_auc = validate_saliency(
                        model, loader, device, epoch,
                        clip_model=clip_model, prompt_text=current_prompt, dynamic=False,
                        prompt_embeds=prompt_embeds,
                        dataset_name=dataset_name,
                        amp_dtype=amp_dtype,
                    )

                    val_loss = tu.reduce_tensor(torch.tensor(val_loss).to(device)).item()
                    val_cc = tu.reduce_tensor(torch.tensor(val_cc).to(device)).item()
                    val_sim = tu.reduce_tensor(torch.tensor(val_sim).to(device)).item()
                    # val_nss = reduce_tensor(torch.tensor(val_nss).to(device)).item()
                    # val_auc = reduce_tensor(torch.tensor(val_auc).to(device)).item()
                    val_kld = tu.reduce_tensor(torch.tensor(val_kld).to(device)).item()
                    
                    if dist.get_rank() == 0:
                        print(f'################### {dataset_name} ###################')
                        print(f" Val Loss: {val_loss:.4f} | CC: {val_cc:.4f} | SIM: {val_sim:.4f} | KLD: {val_kld:.4f} | NSS: {val_nss:.4f}")
                        print(f" AUC: {val_auc:.4f}")
                    
                        
                    if dataset_name == "SALICON":
                        if val_cc > best_val_cc_SALICON:
                            best_val_cc_SALICON = val_cc
                            best_path = os.path.join(ckpt_dir, f"best_model_cc_SALICON{val_cc:.4f}.pth")
                            torch.save(model.state_dict(), best_path)
                            print(f"\n[Save] New best checkpoint saved to: {best_path} (SALICON_CC: {val_cc:.4f})\n")


                # if dist.get_rank() == 0:
                #     with open(log_file_path, "a") as f:
                #                 log_str = f"{epoch}\t{dataset_name}\t{train_loss_video:.4f}\t{val_loss:.4f}\t{val_cc:.4f}\t{val_sim:.4f}\n"
                #                 f.write(log_str)

    print("=" * 60)
    print(f"Training finished. Best Validation CC = {best_val_cc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SegDINO for Saliency Detection")
    parser.add_argument("--save_root", type=str, default="./runs")
    parser.add_argument("--repo_dir", type=str, default="./dinov3")
    parser.add_argument("--data_root", type=str, default=os.environ.get("A2_DATA_ROOT", "datasets"),
                        help="Root directory containing all image and video datasets.")
    # parser.add_argument("--dino_ckpt", type=str, default="./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    # parser.add_argument("--dino_ckpt", type=str, default="./web_pth/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    parser.add_argument("--dino_ckpt", type=str, default="./web_pth/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--dino_size", type=str, default="l", choices=["b", "s", "l"],
                        help="s->vits16, b->vitb16, l->vitl16")
    parser.add_argument("--input_h", type=int, default=448)
    parser.add_argument("--input_w", type=int, default=448)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_path", type=str, default=None, help="Optional checkpoint path to resume from")
    args = parser.parse_args()
    main(args)

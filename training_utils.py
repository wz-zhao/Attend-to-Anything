import os

import clip
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim


def set_cpu_thread_env():
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

    dist.init_process_group(backend="hccl")

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
    if blocks is None:
        raise AttributeError("backbone has no blocks attribute")

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
    dim = sd["cls_token"].shape[-1] if "cls_token" in sd else None
    return {384: "vits16", 768: "vitb16", 1024: "vitl16"}.get(dim, None)


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
        text_dim=768,
    )
    backbone.load_state_dict(state_dict, strict=False)
    return backbone


def load_checkpoint_compat(model, resume_path):
    if resume_path is None or not os.path.exists(resume_path):
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
        if any(k in lname for k in keywords) and p.requires_grad:
            p.requires_grad = False
            num += 1
    if dist.get_rank() == 0:
        print(f"[Freeze LoRA] frozen {num} LoRA parameter tensors.")


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
        {"params": router_params, "lr": base_lr, "weight_decay": weight_decay * 0.1},
    ]
    return optim.AdamW(param_groups)

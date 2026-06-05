import math
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from utils import HyperbolicMultiScaleUpsampleLorentz,HyperbolicMultiScaleBiasHeadLorentzJointAmp
from Video_NSA import StreamFormerTemporalBlock
import math
import lorentz as L
import torch.distributed as dist

class LoRALinear(nn.Module):

    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.05):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.bias = base.bias
        self.weight = base.weight 

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def apply_lora_dino_backbone(
    model: nn.Module,
    r=16,
    alpha=32,
    dropout=0.05,
    last_n_blocks: int | None = 4, 
    target=("attn.qkv", "attn.proj"),
):
  
    max_bid = -1
    for name, _ in model.named_modules():
        if "backbone.blocks." in name:
            parts = name.split(".")
            for j in range(len(parts) - 2):
                if parts[j] == "backbone" and parts[j+1] == "blocks" and parts[j+2].isdigit():
                    max_bid = max(max_bid, int(parts[j+2]))
                    break

    def get_bid(name: str):
        parts = name.split(".")
        for j in range(len(parts) - 2):
            if parts[j] == "backbone" and parts[j+1] == "blocks" and parts[j+2].isdigit():
                return int(parts[j+2])
        return None

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not name.startswith("backbone.blocks."):
            continue
        if not any(t in name for t in target):
            continue

        bid = get_bid(name)
        if last_n_blocks is not None and bid is not None and max_bid >= 0:
            if bid < (max_bid - last_n_blocks + 1):
                continue 
        parent = model
        *path, leaf = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, leaf, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        replaced += 1

    print(f"[LoRA] replaced {replaced} Linear layers (target={target}, last_n_blocks={last_n_blocks})")
    return model


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributed as dist



class HyperbolicRouter(nn.Module):
   
    def __init__(self, text_in_dim=768, hyp_dim=128, num_experts=4, curv_init=1.0):
        super().__init__()
        self.hyp_dim = hyp_dim
        self.device_buffer = nn.Parameter(torch.empty(0)) 
        self.curv = nn.Parameter(torch.tensor(curv_init).log())
        self._curv_minmax = {"max": math.log(curv_init * 10), "min": math.log(curv_init / 10)}
        self.textual_proj = nn.Linear(text_in_dim, hyp_dim, bias=False)
        self.textual_alpha = nn.Parameter(torch.tensor(hyp_dim ** -0.5).log())
        self.none_anchor_tan = nn.Parameter(torch.zeros(1, hyp_dim))
        self.expert_centroids_tan = nn.Parameter(torch.randn(num_experts, hyp_dim) * 0.2)
        self.current_weights = None
        self.logit_scale = nn.Parameter(torch.tensor(5.0).log())  
        self.logit_scale_min = math.log(1.0)
        self.logit_scale_max = math.log(10.0)

    def _update_curv(self):
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        return self.curv.exp()

    def encode_hyp(self, feats_tan, alpha_log, curv):
     
        alpha_log.data = torch.clamp(alpha_log.data, max=0.0)
        feats = feats_tan * alpha_log.exp()
        with torch.autocast(self.device_buffer.device.type, dtype=torch.float32):
            z = L.exp_map0(feats, curv)
        return z

    def forward(self, text_emb):
       
        curv = self._update_curv()
        B = text_emb.shape[0]


        text_tan = self.textual_proj(text_emb)
        z_txt = self.encode_hyp(text_tan, self.textual_alpha, curv)


        z_none_single = self.encode_hyp(self.none_anchor_tan, self.textual_alpha, curv)
   
        z_anc = z_none_single.expand(B, -1)

  
        z_experts = L.exp_map0(self.expert_centroids_tan, curv) 

 
        z_txt_exp = z_txt.unsqueeze(1)
        z_exp_exp = z_experts.unsqueeze(0)

       
        interaction = -z_txt_exp[..., 0] * z_exp_exp[..., 0] + \
                       torch.sum(z_txt_exp[..., 1:] * z_exp_exp[..., 1:], dim=-1)
        



        self.logit_scale.data.clamp_(self.logit_scale_min, self.logit_scale_max)
        logits = interaction * self.logit_scale.exp()
        weights = F.softmax(logits, dim=-1)

        topk = 4
        vals, idx = torch.topk(weights, k=topk, dim=-1)  

        self.current_weights = weights

        return z_txt, z_anc, weights, curv



class HyperbolicMoELoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, router: nn.Module, num_experts=4, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base
        self.router = router 
       
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.bias = base.bias
        self.weight = base.weight

        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()


        self.lora_A_experts = nn.Parameter(torch.empty(num_experts, self.in_features, r))
        self.lora_B_experts = nn.Parameter(torch.empty(num_experts, r, self.out_features))

        self.reset_parameters()


        for p in self.base.parameters():
            p.requires_grad_(False)

    def reset_parameters(self):
        for k in range(self.lora_A_experts.shape[0]):
            nn.init.kaiming_uniform_(self.lora_A_experts[k], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_experts[k])

    def forward(self, x):

        base_out = self.base(x)


        weights = getattr(self.router, 'current_weights', None)
        if weights is None:
         
            return base_out

       
        x_d = self.dropout(x) 


        xa = torch.einsum('bnd,kdr->bnkr', x_d, self.lora_A_experts)

       
        xb = torch.einsum('bnkr,kro->bnko', xa, self.lora_B_experts)


        w_expanded = weights.unsqueeze(1).unsqueeze(-1)
        lora_out = (xb * w_expanded).sum(dim=2) 

        return base_out + lora_out * self.scaling


def apply_hyper_lora_dino_backbone(
    model: nn.Module, 
    router: nn.Module, 
    num_experts=4, 
    r=8, 
    alpha=32, 
    dropout=0.05, 
    last_n_blocks=8, 
    target=("attn.qkv", "attn.proj")
):
   
    max_bid = -1
    for name, _ in model.named_modules():
      
        if "blocks." in name:
            parts = name.split(".")
            try:
              
                idx = parts.index("blocks")
                if idx + 1 < len(parts) and parts[idx+1].isdigit():
                    max_bid = max(max_bid, int(parts[idx+1]))
            except ValueError:
                continue
                
    print(f"[LoRA Setup] Detected max block ID: {max_bid}")


    def get_bid(name):
        parts = name.split(".")
        try:
            if "blocks" in parts:
                idx = parts.index("blocks")
                return int(parts[idx+1])
        except (ValueError, IndexError):
            return None
        return None


    replaced = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear): 
            continue
            

        if "blocks." not in name:
            continue
            
        if not any(t in name for t in target): 
            continue

        bid = get_bid(name)
        if bid is None: 
            continue
        
        if last_n_blocks is not None and max_bid >= 0:

            if bid < (max_bid - last_n_blocks + 1): 
                continue


        parent = model
        path = name.split(".")
 
        for p in path[:-1]: 
            parent = getattr(parent, p)
        leaf = path[-1]
        

        new_layer = HyperbolicMoELoRALinear(
            base=module, 
            router=router, 
            num_experts=num_experts, 
            r=r, 
            alpha=alpha, 
            dropout=dropout
        )

        setattr(parent, leaf, new_layer)
        replaced += 1
        
  
        if replaced == 1:
            print(f"[LoRA Info] First layer replaced: {name}")

    if replaced == 0:
        print(f"[LoRA Error] 0 layers replaced! Check your 'target' list: {target}")
        print("Example layer names in model:", list(dict(model.named_modules()).keys())[:5])
    else:
        print(f"[LoRA Success] Replaced {replaced} Linear layers with MoE-LoRA (K={num_experts}).")
    
    return model
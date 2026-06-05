import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

                     
class TextConditionedRouter(nn.Module):
    def __init__(self, text_dim=768, num_experts=8):
        super().__init__()
                              
        self.mapper = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_experts)
        )

    def forward(self, text_embedding):
        # text_embedding: [B, text_dim]
        logits = self.mapper(text_embedding)
        return logits

                            
class LoRAExpert(nn.Module):
    """
    使用 LoRA 结构的专家。
    相比全连接 Adapter，参数量减少 90% 以上，但拟合能力极强。
    """
    def __init__(self, dim, r=16, alpha=32):                 
        super().__init__()
        self.lora_down = nn.Linear(dim, r, bias=False)
        self.act = nn.GELU()
        self.lora_up = nn.Linear(r, dim, bias=False)
        self.scaling = alpha / r
        
                                
        nn.init.zeros_(self.lora_up.weight)
                            
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)

    def forward(self, x, H, W, text_emb=None):
                                         
        return self.lora_up(self.act(self.lora_down(x))) * self.scaling

                              
class AdaLNModulator(nn.Module):
    """
    SOTA 标配：使用文本特征动态预测 Scale (gamma) 和 Shift (beta)。
    这比简单的加权求和更深入地改变了特征的分布。
    """
    def __init__(self, dim, text_dim):
        super().__init__()
        self.siLU = nn.SiLU()
        self.linear = nn.Linear(text_dim, 2 * dim)                  
        
                                      
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, text_emb):
        # text_emb: [B, text_dim] -> [B, 2*dim]
                                          
        emb = self.linear(self.siLU(text_emb)).unsqueeze(1)
        gamma, beta = emb.chunk(2, dim=-1)
        
                                      
        return x * (1 + gamma) + beta

                                
class BiasExpert(nn.Module):
    def __init__(self, dim, text_dim=768):
        super().__init__()
        self.param_predictor = nn.Sequential(
            nn.Linear(text_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2) 
        )
        self.out_proj = nn.Linear(1, dim)

    def generate_gaussian(self, H, W, sigma, device):
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dist = xx**2 + yy**2
        return torch.exp(-dist / (2 * (sigma**2 + 1e-6)))

    def forward(self, x, H, W, text_emb):
        B = text_emb.shape[0]
        N = x.shape[1]
        num_extra = N - (H * W)
        
        params = self.param_predictor(text_emb)
        sigma = torch.sigmoid(params[:, 0]) * 5.0 + 0.1
        amp = torch.sigmoid(params[:, 1])
        
        bias_maps = []
        for i in range(B):
            g = self.generate_gaussian(H, W, sigma[i], text_emb.device)
            bias_maps.append(g * amp[i])
        
        bias = torch.stack(bias_maps).unsqueeze(-1).view(B, -1, 1) 
        bias_feat = self.out_proj(bias)
        
        if num_extra > 0:
            zeros = torch.zeros(B, num_extra, bias_feat.shape[-1], device=x.device, dtype=bias_feat.dtype)
            bias_feat = torch.cat([zeros, bias_feat], dim=1)
            
        return bias_feat

                           
class MoEAdapterLayer(nn.Module):
    def __init__(self, dim, text_dim=768, num_experts=12):            
        super().__init__()
        self.router = TextConditionedRouter(text_dim, num_experts)
        
                                  
        self.adaLN = AdaLNModulator(dim, text_dim)
        
        self.experts = nn.ModuleList()
        
                            
                                
                                           
                                                       
        for _ in range(num_experts - 1):
            self.experts.append(LoRAExpert(dim, r=16))
            
                                           
        self.experts.append(BiasExpert(dim, text_dim))
        
        self.forward_count = 0
        
    def forward(self, x, H, W, text_emb):
                 
      
        router_logits = self.router(text_emb) 
#         self.forward_count += 1
#         if self.forward_count % 100 == 0:
                                 
#             print("\n[MoE Router] Forward step:", self.forward_count)
#             print("Router logits:", router_logits.detach().cpu())
                                     
#             print("Router probs:", torch.softmax(router_logits, dim=-1).detach().cpu())
        
                                
        top_k = 4 
        top_k_vals, top_k_indices = torch.topk(router_logits, top_k, dim=-1)
        top_k_probs = F.softmax(top_k_vals, dim=-1)
        
                
        weights = torch.zeros_like(router_logits)
        weights.scatter_(1, top_k_indices, top_k_probs)
        
                                  
        moe_out = 0
        for i, expert in enumerate(self.experts):
            w_i = weights[:, i]
            if w_i.sum() == 0: continue
            
            expert_out = expert(x, H, W, text_emb)
            moe_out += w_i.view(-1, 1, 1) * expert_out
            
                                     
                    
                                              
                                                
        moe_out = self.adaLN(moe_out, text_emb)
        
                 
        return x + moe_out
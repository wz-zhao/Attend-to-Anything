import torch
import torch.nn as nn
import torchaudio.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

import torch
import torch.nn as nn
import wav2clip
import librosa
import numpy as np
import torchaudio.transforms as T

class Wav2CLIPAudioEncoder(nn.Module):
    def __init__(self, device='cuda', freeze=True):
        super().__init__()
        print(">>> Loading Wav2CLIP Model (Audio-Visual Aligned)...")
        
                 
                                                     
        self.model = wav2clip.get_model()
        
                  
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        else:
            self.model.train()
            
        self.model.to(device)
        self.device = device

    def forward(self, x_audio):
      
        
                 
        if x_audio.dim() == 3:
            x_audio = x_audio.squeeze(1) # (B, 1, L) -> (B, L)
            
                             
                                           
                                        
        
                                                  
                                   
        if not hasattr(self, 'resampler'):
            self.resampler = T.Resample(48000, 16000).to(self.device)
            
        x_16k = self.resampler(x_audio)
        
                 
                                 
        embeddings = self.model(x_16k)
        
        # Output shape: (B, 512)
        return embeddings.float()


class SelectiveAudioVisualFusion(nn.Module):
    def __init__(self, visual_dim, audio_dim=1024, hidden_dim=256, dropout=0.1):
        super().__init__()
        
                
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        
                                  
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        
                                       
        self.film_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2)          
        )
        
                                             
                                         
                                                      
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)

        # 4. Selector (Gate)
        self.relevance_selector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
                                 
        nn.init.constant_(self.relevance_selector[-2].bias, -1.0)

                
        self.out_proj = nn.Linear(hidden_dim, visual_dim)
        self.norm = nn.LayerNorm(visual_dim)                    

    def forward(self, x_visual, x_audio):
        T, C, H, W = x_visual.shape
        
                      
        x_vis_perm = x_visual.permute(0, 2, 3, 1) # (T, H, W, C)
        x_vis_flat = x_vis_perm.flatten(1, 2)     # (T, N, C)
        x_aud_seq = x_audio.unsqueeze(1)          # (T, 1, 1024)

        q = self.visual_proj(x_vis_flat) # (T, N, Hid)
        k = self.audio_proj(x_aud_seq)   # (T, 1, Hid)
        v = k

                         
        vis_global = q.mean(dim=1)
        aud_global = k.squeeze(1)
        global_feat = torch.cat([vis_global, aud_global], dim=-1)
        alpha = self.relevance_selector(global_feat).unsqueeze(1) # (T, 1, 1)

        # ... Attention ...
        attn_audio, _ = self.cross_attn(query=q, key=k, value=v)
        
                         
        film_params = self.film_generator(attn_audio)
        raw_gamma, beta = torch.chunk(film_params, 2, dim=-1)
        
                          
                               
        gamma = torch.tanh(raw_gamma)
        
                       
        # Visual * (1 + alpha * gamma) + (alpha * beta)
                                                       
                                                 
        
        fused = q * (1 + alpha * gamma) + (alpha * beta)
        
                    
        out = self.out_proj(fused)
        out = self.norm(out + x_vis_flat)
        out = out.view(T, H, W, C).permute(0, 3, 1, 2).contiguous()

        return out, alpha
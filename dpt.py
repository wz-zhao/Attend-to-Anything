import torch
import math
import torch.nn as nn
import lorentz as L
from easydict import EasyDict
import torch.distributed as dist
import torch.nn.functional as F
from Video_NSA import StreamFormerTemporalBlock
from audio_backbone import SelectiveAudioVisualFusion
from LORA import HyperbolicRouter,apply_hyper_lora_dino_backbone
from utils import HyperbolicMultiScaleUpsampleLorentz,HyperbolicMultiScaleBiasHeadLorentzJointAmp



class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=True, out_channels=[96, 192, 384, 768], 
                 hyp_embed_dim=128, curv_init=1.0, text_in_dim=768, audio_in_dim=512):
        super().__init__()

        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels, out_channels=oc, kernel_size=1) for oc in out_channels]
        )
        self.av_fusion = SelectiveAudioVisualFusion(
            visual_dim=out_channels[3],
            audio_dim=audio_in_dim,      
            hidden_dim=256,
        )


        fp_kwargs = dict(
            enable_fp_drift=True,
            fp_dt=1.0,
            fp_enable_diffusion=True,
            fp_enable_assimilation=True,
            fp_couple_features=True,

            fp_detach_A=True,
            fp_attn_temperature=2.0,
            fp_alpha_init=0.0,   
            fp_alpha_max=0.5,   

            keep_feature_temporal_attn=True,
        )
        
        streamformer_kwargs = dict(
            num_frames=64,
            num_heads=8,
            attn_drop=0.0,
            proj_drop=0.0,
            qkv_bias=True,
            ffn_drop=0.0,
            mlp_ratio=4.0,
            **fp_kwargs,
        )
        
        def make_stage_block(c):
            return StreamFormerTemporalBlock(in_channels=c, **streamformer_kwargs)

        self.motion_modules = nn.ModuleList([
            make_stage_block(out_channels[0]),
            make_stage_block(out_channels[1]),
            make_stage_block(out_channels[2]),
            make_stage_block(out_channels[3]),
            make_stage_block(256),
        ])

        self.embed_dim = hyp_embed_dim
        self.curv = nn.Parameter(torch.tensor(curv_init).log())
        self._curv_minmax = {"max": math.log(curv_init * 10), "min": math.log(curv_init / 10)}
        self.visual_alpha = nn.Parameter(torch.tensor(hyp_embed_dim ** -0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(hyp_embed_dim ** -0.5).log())
        C4 = out_channels[3]
        self.visual_proj = nn.Linear(C4, hyp_embed_dim, bias=False)
        self.textual_proj = nn.Linear(text_in_dim, hyp_embed_dim, bias=False)
        self.none_anchor_tan = nn.Parameter(torch.zeros(1, hyp_embed_dim))

        self._rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        self.decoder_stage1 = HyperbolicMultiScaleUpsampleLorentz(features, 128, text_dim=text_in_dim, hyp_dim=hyp_embed_dim, curv=curv_init, learn_curv=False)
        self.decoder_stage2 = HyperbolicMultiScaleUpsampleLorentz(128, 64,  text_dim=text_in_dim, hyp_dim=hyp_embed_dim, curv=curv_init, learn_curv=False)
        self.decoder_stage3 = HyperbolicMultiScaleUpsampleLorentz(64,  32,  text_dim=text_in_dim, hyp_dim=hyp_embed_dim, curv=curv_init, learn_curv=False)

        self.decoder_out = HyperbolicMultiScaleBiasHeadLorentzJointAmp(32, nclass=nclass, text_dim=text_in_dim, hyp_dim=hyp_embed_dim, curv=curv_init, learn_curv=False)

        C4 = out_channels[3]  
        self.start_proj = nn.Sequential(
            nn.Conv2d(C4, features, kernel_size=1, bias=False),
            nn.GroupNorm(32, features) if use_bn else nn.Identity(),
            nn.SiLU(),
        )

     
        C1, C2, C3 = out_channels[0], out_channels[1], out_channels[2]


        self.fuse1 = nn.Sequential(
            nn.Conv2d(128 + C3, 128, kernel_size=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
    

        self.fuse2 = nn.Sequential(
            nn.Conv2d(64 + C2, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
   

        self.fuse3 = nn.Sequential(
            nn.Conv2d(32 + C1, 32, kernel_size=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.se1 = SELayer(channel=128, reduction=16)
        self.se2 = SELayer(channel=64, reduction=8)
        self.se3 = SELayer(channel=32, reduction=4)
        
    @property
    def device(self):
        return self.curv.device

    def _update_curv(self):
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        return self.curv.exp()

    def encode_hyp(self, feats_tan, alpha_log, curv):
        """tangent -> hyperbolic"""
        alpha_log.data = torch.clamp(alpha_log.data, max=0.0)
        feats = feats_tan * alpha_log.exp()
        with torch.autocast(self.device.type, dtype=torch.float32):
            z = L.exp_map0(feats, curv)
        return z


      
    def forward(self, out_features, patch_h, patch_w, dynamic,text_emb=None,audio_emb = None):
 
        _curv = self._update_curv()
        outs = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)

            if dynamic and i > 1:
                F_, C, H, W = x.shape
    
                x_motion = x.unsqueeze(0).permute(0, 2, 1, 3, 4)  
                x_motion = self.motion_modules[i](x_motion)
   
                x = x_motion.permute(0, 2, 1, 3, 4).squeeze(0)
            outs.append(x)

        layer_1, layer_2, layer_3, layer_4 = outs 
        if dynamic and audio_emb is not None:
            layer_4, alpha = self.av_fusion(layer_4, audio_emb)
   

        img_global = torch.mean(layer_4, dim=(2, 3))        # (B, C4)
        img_tan = self.visual_proj(img_global)              # (B, embed_dim)
        z_img = self.encode_hyp(img_tan, self.visual_alpha, _curv)

        # none anchor
        z_none = self.encode_hyp(self.none_anchor_tan, self.textual_alpha, _curv)
        B = layer_4.shape[0]

        text_tan = self.textual_proj(text_emb)         # (B, hyp_dim)
        z_txt = self.encode_hyp(text_tan, self.textual_alpha, _curv)    
        z_anc = z_none.expand(B, -1)

        x = self.start_proj(layer_4)

        d1 = self.decoder_stage1(x, text_emb, z_txt)
        skip1 = layer_3
        if skip1.shape[-2:] != d1.shape[-2:]:
            skip1 = F.interpolate(skip1, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.fuse1(torch.cat([d1, skip1], dim=1))
        d1 = self.se1(d1)

        d2 = self.decoder_stage2(d1, text_emb, z_txt)
        skip2 = layer_2
        if skip2.shape[-2:] != d2.shape[-2:]:
            skip2 = F.interpolate(skip2, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.fuse2(torch.cat([d2, skip2], dim=1))
        d2 = self.se2(d2)
   

        d3 = self.decoder_stage3(d2, text_emb, z_txt)
        skip3 = layer_1
        if skip3.shape[-2:] != d3.shape[-2:]:
            skip3 = F.interpolate(skip3, size=d3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.fuse3(torch.cat([d3, skip3], dim=1))
        d3 = self.se3(d3)
       
        pred_map = self.decoder_out(d3, text_emb, z_txt)  # logits
        return pred_map, z_img, z_txt, z_anc, _curv
       


class DPT(nn.Module):
    def __init__(self, nclass=1, features=256, out_channels=[96, 192, 384, 768], use_bn=True, backbone=None):
        super().__init__()

        # self.router = HyperbolicRouter(text_in_dim=768, hyp_dim=128, num_experts=4)

        self.backbone = backbone

        if self.backbone.embed_dim == 1024:
            self.encoder_size = "large"
        elif self.backbone.embed_dim == 768:
            self.encoder_size = "base"
        elif self.backbone.embed_dim == 384:
            self.encoder_size = "small"
        else:
            raise ValueError(f"Error backbone embed_dim: {self.backbone.embed_dim}")

        print(f"[DPT Info] : '{self.encoder_size}' (embed_dim: {self.backbone.embed_dim})")

        self.intermediate_layer_idx = {
            "small": [2, 5, 8, 11],
            "base":  [2, 5, 8, 11],
            "large": [5, 11, 17, 23],
        }

        head_out_channels = out_channels
        self.head = DPTHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=head_out_channels)

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x, text_emb=None,audio_emb=None, dynamic=False):
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16

        features = self.backbone.get_intermediate_layers(
            x, 
            n=self.intermediate_layer_idx[self.encoder_size],
            text_emb=text_emb
        )

        out, z_img, z_txt, z_anc, curv = self.head(features, patch_h, patch_w, dynamic,text_emb,audio_emb)
        out = F.interpolate(out, (x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=True)
        out = torch.sigmoid(out)
        return out, {"z_img": z_img, "z_txt": z_txt, "z_anc": z_anc, "curv": curv}



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

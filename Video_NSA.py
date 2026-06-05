import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None, return_attn=False, attn_temperature=1.0):
        B_, L, C = x.shape
        h = self.num_heads
        d = C // h

        qkv = self.qkv(x).reshape(B_, L, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_temperature != 1.0:
            attn = attn / float(attn_temperature)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, L, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        if return_attn:
            return out, attn
        return out



class LightFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class StreamFormerTemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        num_frames=16,
        num_heads=8,
        attn_drop=0.0,
        proj_drop=0.0,
        qkv_bias=True,
        ffn_drop=0.0,
        mlp_ratio=4.0,
        init_scale_temporal=1e-3,

        # FP options
        enable_fp_drift=True,
        fp_dt=1.0,
        fp_enable_diffusion=True,
        fp_enable_assimilation=True,
        fp_couple_features=True,
        fp_detach_A=True,        
        fp_attn_temperature=2.0, 
        fp_alpha_init=0.0,        
        fp_alpha_max=1.0,         

        keep_feature_temporal_attn=True,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.enable_fp_drift = enable_fp_drift
        self.fp_dt = float(fp_dt)
        self.fp_enable_diffusion = fp_enable_diffusion
        self.fp_enable_assimilation = fp_enable_assimilation
        self.fp_couple_features = fp_couple_features
        self.keep_feature_temporal_attn = keep_feature_temporal_attn

        self.fp_detach_A = fp_detach_A
        self.fp_attn_temperature = fp_attn_temperature
        self.fp_alpha_max = float(fp_alpha_max)

        # temporal
        self.temporal_norm = nn.LayerNorm(in_channels)
        self.temporal_attn = MultiheadSelfAttention(
            dim=in_channels, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.temporal_dense = nn.Linear(in_channels, in_channels)
        self.temporal_attention_scale = nn.Parameter(init_scale_temporal * torch.ones(1))

        self.temporal_ffn_norm = nn.LayerNorm(in_channels)
        self.temporal_ffn = LightFFN(in_channels, mlp_ratio=mlp_ratio, drop=ffn_drop)
        self.temporal_ffn_scale = nn.Parameter(init_scale_temporal * torch.ones(1))


        self.spatial_mixer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
            nn.GELU(),
        )


        self.time_embedding = nn.Parameter(torch.zeros(1, num_frames, in_channels))
        nn.init.trunc_normal_(self.time_embedding, std=0.02)


        self.sal_obs_head = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)

        self.nu_head = nn.Linear(in_channels, 1)    
        self.lam_head = nn.Linear(in_channels, 1)  

        self.fp_alpha_logit = nn.Parameter(torch.tensor(float(fp_alpha_init)))
        self._eps = 1e-6

    def _get_time_embed(self, T, device):
        if T == self.num_frames:
            return self.time_embedding.to(device)
        te = F.interpolate(
            self.time_embedding.transpose(1, 2), size=T, mode="linear", align_corners=False
        ).transpose(1, 2)
        return te.to(device)

    @staticmethod
    def _spatial_softmax_density(logit_1chw):
        N, _, H, W = logit_1chw.shape
        return torch.softmax(logit_1chw.flatten(2), dim=-1).view(N, 1, H, W)

    def _normalize_density_per_frame(self, u_bthw):
        B, T, H, W = u_bthw.shape
        u = torch.clamp(u_bthw, min=0.0)
        z = u.view(B, T, -1).sum(dim=-1, keepdim=True) + self._eps
        u = (u.view(B, T, -1) / z).view(B, T, H, W)
        return u

    def _tokenwise_safe_norm(self, u_tokens):

        u = torch.clamp(u_tokens, min=0.0)
        z = u.sum(dim=0, keepdim=True) + self._eps  
        return u / z

    def forward(self, x, return_density: bool = False, **kwargs):

        B, C, T, H, W = x.shape
        device = x.device
        x_bt = x.permute(0, 2, 3, 4, 1).contiguous()
        x_bt = x_bt + self._get_time_embed(T, device)[:, :, None, None, :]

        u_seq = None


        x_temp_feat = x_bt.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, C)
        x_temp_feat_norm = self.temporal_norm(x_temp_feat)

        feat_attn_out, attn_w = self.temporal_attn(
            x_temp_feat_norm,
            return_attn=True,
            attn_temperature=self.fp_attn_temperature
        )


        if self.enable_fp_drift:
            ft_all = x_bt.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W).contiguous()
            logit = self.sal_obs_head(ft_all)
            u_obs = self._spatial_softmax_density(logit).view(B, T, 1, H, W)
            u_tokens = u_obs.squeeze(2).permute(0, 2, 3, 1).reshape(B * H * W, T, 1)  # (BHW,T,1)


            A = attn_w.mean(dim=1)
            if self.fp_detach_A:
                A = A.detach()


            u_drift = torch.bmm(A, u_tokens)
            u_tokens = u_tokens + self.fp_dt * (u_drift - u_tokens)


            if self.fp_enable_diffusion and T >= 3:
                nu = F.softplus(self.nu_head(x_temp_feat_norm))  # (BHW,T,1)
                lap = u_tokens[:, :-2] - 2.0 * u_tokens[:, 1:-1] + u_tokens[:, 2:]
                u_tokens[:, 1:-1] = u_tokens[:, 1:-1] + self.fp_dt * nu[:, 1:-1] * lap


            if self.fp_enable_assimilation:
                lam = torch.sigmoid(self.lam_head(x_temp_feat_norm))  # (BHW,T,1)
                u_obs_tokens = u_obs.squeeze(2).permute(0, 2, 3, 1).reshape(B * H * W, T, 1)
                u_tokens = u_tokens + self.fp_dt * lam * (u_obs_tokens - u_tokens)


            u_tokens = self._tokenwise_safe_norm(u_tokens)


            u = u_tokens.view(B, H, W, T).permute(0, 3, 1, 2).contiguous()  
            u = self._normalize_density_per_frame(u)
            u_seq = u.unsqueeze(2)

  
            if self.fp_couple_features:
                alpha = self.fp_alpha_max * torch.sigmoid(self.fp_alpha_logit)  
                gate = u_seq.permute(0, 1, 3, 4, 2).contiguous()  

                x_bt = x_bt + alpha * gate * x_bt

                x_temp_feat = x_bt.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, C)

        # =========================================================
        # Feature mixing (temporal attn + ffn)
        # =========================================================
        if self.keep_feature_temporal_attn:
            x_attn = self.temporal_dense(feat_attn_out)
            x_temp = x_temp_feat + self.temporal_attention_scale * x_attn

            x_norm2 = self.temporal_ffn_norm(x_temp)
            x_ffn = self.temporal_ffn(x_norm2)
            x_temp = x_temp + self.temporal_ffn_scale * x_ffn

            x_bt = x_temp.view(B, H, W, T, C).permute(0, 3, 1, 2, 4).contiguous()


        x_btc = x_bt.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W).contiguous()
        x_btc = x_btc + self.spatial_mixer(x_btc)
        x_bt = x_btc.view(B, T, C, H, W).permute(0, 1, 3, 4, 2).contiguous()
        x_out = x_bt.permute(0, 4, 1, 2, 3).contiguous()

        if return_density:
            return x_out, u_seq
        return x_out

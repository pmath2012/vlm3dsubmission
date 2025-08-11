import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from MedSyn.src.train_super_res import Unet3D, GaussianDiffusion
from MedSyn.src.utils import * 

from einops import rearrange
from text_to_latent_dataloader import GLOBAL_STD, GLOBAL_MEAN

import kornia

def edge_map(volume_tensor):
    """
    Calculates the Sobel edge map for a 5D volume by processing it as a batch of 2D slices.
    """
    # Get the original 5D shape
    # B: Batch, C: Channel, D: Depth, H: Height, W: Width
    B, C, D, H, W = volume_tensor.shape

    # 1. Reshape the volume to look like a batch of 2D images
    # We merge the Batch and Depth dimensions together
    # New shape: (B * D, C, H, W)
    slices_2d = volume_tensor.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

    # 2. Apply the 2D Sobel filter
    # The output will be a 4D tensor of edge maps
    edge_maps_2d = kornia.filters.sobel(slices_2d)

    # 3. Reshape the edge maps back to the original 5D volume format
    # First, view it as (B, D, C, H, W)
    # Then, permute the dimensions back to (B, C, D, H, W)
    final_edge_volume = edge_maps_2d.view(B, D, C, H, W).permute(0, 2, 1, 3, 4)

    return final_edge_volume

def edge_loss(pred, target):
    """
    Calculates the L1 loss between the Sobel edge maps of the prediction and target.
    """
    pred_edge = edge_map(pred)
    target_edge = edge_map(target)

    # Use L1 loss for robustness
    loss = torch.nn.functional.l1_loss(pred_edge, target_edge)
    return loss

class LatentOnlyUNet3D(Unet3D):
    """
    UNet3D variant for latent-only training (no low-res input concatenation).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace first conv to accept only latent channels (no *2 for SR concat)
        init_dim = self.init_conv0.out_channels
        init_kernel_size = self.init_conv0.kernel_size[0]
        init_padding = init_kernel_size // 2
        self.init_conv0 = nn.Conv3d(
            self.channels,  # latent channels only
            init_dim,
            (init_kernel_size, init_kernel_size, init_kernel_size),
            padding=(init_padding, init_padding, init_padding)
        )

    def exists(self, val):
        return val is not None
    
    def forward(
            self,
            x,
            time,
            indexes=None,
            cond=None,
            null_cond_prob=0.,
            focus_present_mask=None,
            prob_focus_present=0.
            # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        assert not (self.has_cond and not self.exists(cond)), 'cond must be passed in if cond_dim specified'

        x = self.init_conv0(x)

        r = x.clone()
        t = self.time_mlp(time) if self.exists(self.time_mlp) else None

        # classifier free guidance

        h = []

        for idx, (block1, block2, spatial_attn, cross_attn, temporal_conv, downsample) in enumerate(self.downs):
            x = block1(x, t)
            x = block2(x, t)
            h.append(x)
            x = downsample(x)
            x = spatial_attn(x)
            x = cross_attn(x, kv=cond)
            x = temporal_conv(x, t)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_conv(x, t)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, cross_attn, temporal_conv, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = upsample(x)
            x = block2(x, t)
            x = spatial_attn(x)
            x = cross_attn(x, kv=cond)
            x = temporal_conv(x, t)

        x = torch.cat((x, r), dim=1)
        return self.final_conv0(x)

class LatentDiffusion(GaussianDiffusion):
    def __init__(self, unet_kwargs, diffusion_kwargs):
        unet = LatentOnlyUNet3D(**unet_kwargs)
        # for name, module in unet.named_modules():
            # if hasattr(module, "context_dim"):
                # print("***"*50)
                # print(name, module.context_dim)
                # print("***"*50)
        super().__init__(denoise_fn=unet, **diffusion_kwargs)
        self.log_var_l2 = nn.Parameter(torch.tensor(0.0))     
        self.log_var_edge = nn.Parameter(torch.tensor(0.0))   

    def p_losses(self, x_start, t, indexes=None, cond=None, noise=None, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        x_noisy, _noise = get_z_t(x_start, t)
        

        # print("--"*10, "\n\nCond in p_losses : ",cond.shape, "\n","--"*10) 
        # assume text embeddings extracted from CheXBERT
        if cond is not None:
            if np.random.choice(10, 1) == 0:
                cond = cond*0
            # print("--"*10, "\n\nCond after None check : ", cond.shape, "\n","--"*10) 
            
        x_recon = self.denoise_fn(x_noisy, t*self.num_timesteps, indexes=indexes, cond=cond, **kwargs)

        if self.loss_type == 'l1':
            loss = F.l1_loss(x_start, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(x_start, x_recon)
        elif self.loss_type == 'edge+l2':
            l2 = F.mse_loss(x_start, x_recon, reduction='mean')
            edge = edge_loss(x_start, x_recon)

            loss = (
                torch.exp(-self.log_var_l2) * l2 +
                torch.exp(-self.log_var_edge) * edge +
                self.log_var_l2 + self.log_var_edge
            )
            return loss
        else:
            raise NotImplementedError()
        return loss

    def forward(self, x, *args, **kwargs):
        b, device, img_size = x.shape[0], x.device, self.image_size
        t = torch.rand((b,), device=device).float()
        return self.p_losses(x, t, *args, **kwargs)

    def p_mean_variance(self, x, t, clip_denoised: bool, indexes=None, cond=None, cond_scale=1.):

        x_recon = self.denoise_fn.forward_with_cond_scale(x, t, indexes=indexes, cond=cond, cond_scale=cond_scale)

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        # model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        model_mean, posterior_variance = get_z_t_via_z_tp1(x_recon, x, (t - 1) * 1.0 / (self.num_timesteps - 1.0),
                                                           (t * 1.0) / (self.num_timesteps - 1.0))
        return model_mean, posterior_variance
    
    @torch.inference_mode()
    def p_sample(self, x, t, indexes=None, cond=None, cond_scale=1., clip_denoised=True):
        device = x.device
        B, _, F, _, _ = x.shape  # assume x is [B, C, F, H, W]; change if different

        # ensure t is [B] LongTensor on the same device
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=device, dtype=torch.long)
        t = t.to(device).long().view(-1)
        if t.numel() == 1 and B > 1:
            t = t.expand(B)

        model_mean, model_variance = self.p_mean_variance(
            x=x, t=t, indexes=indexes, clip_denoised=clip_denoised,
            cond=cond, cond_scale=cond_scale
        )

        noise = torch.randn_like(x)

        # no noise when t == 0  -> mask shape [B,1,F,1,1], broadcast-safe
        nonzero_mask = (t != 0).float().view(B, 1, 1, 1, 1).expand(B, 1, F, 1, 1)

        return model_mean + nonzero_mask * torch.sqrt(model_variance) * noise

    
    @torch.inference_mode()
    def p_sample_ddim(self, x,slice_id, cond, t, t_minus, clip_denoised: bool, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        x_recon = self.denoise_fn.forward_with_cond_scale(x, t, indexes=slice_id, cond=cond, cond_scale=1.0)

        # current prediction for x_0
        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s
        if t[0]<int(self.num_timesteps / self.ddim_timesteps):
            x = x_recon
        else:
            t_minus = torch.clip(t_minus,min=0.0)
            x = ddim_sample(x_recon, x, (t_minus * 1.0) / (self.num_timesteps), (t * 1.0) / (self.num_timesteps))
        return x


    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, cond_scale=1., use_ddim=True, clip_denoised=False, device="cpu"):
        bsz = shape[0]

        if use_ddim:
            time_steps = range(0, self.num_timesteps+1, int(self.num_timesteps/self.ddim_timesteps))
        else:
            time_steps = range(0, self.num_timesteps)

        img = torch.randn(shape, device=device)
        indexes = []
        for b in range(bsz):
            index = np.arange(self.num_frames)
            indexes.append(torch.from_numpy(index))
        indexes = torch.stack(indexes, dim=0).long().to(device)
        for i, t in enumerate((reversed(time_steps))):
            time = torch.full((bsz,), t, device=device, dtype=torch.float32)

            if use_ddim:
                time_minus = time - int(self.num_timesteps / self.ddim_timesteps)
                img = self.p_sample_ddim(x=img, slice_id=indexes, cond=cond, t=time,
                                         t_minus=time_minus, clip_denoised=clip_denoised, index=len(time_steps) - i - 1)
            else:
                img = self.p_sample(img, time, indexes=indexes, cond=cond,
                                    cond_scale=cond_scale, clip_denoised=clip_denoised)

        return img

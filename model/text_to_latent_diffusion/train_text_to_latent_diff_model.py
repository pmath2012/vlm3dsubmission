import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append('/opt/app/model/text_to_latent_diffusion/MedSyn/src/')
sys.path.append('/opt/app/model/weights/bert_finetuned/')
sys.path.append('/opt/app/model/text_to_latent_diffusion/')

import torch
from torch.optim import AdamW
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar
from transformers import AutoTokenizer, AutoModel
from train_super_res import EMA
from text_to_latent_dataloader import get_dataloader
from text_to_latent_diffusion_model import LatentDiffusion
import imageio
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from copy import deepcopy


# ----------------------------
# GIF Visualization Utilities
# ----------------------------
@rank_zero_only
def volume_to_axial_gif(volume, save_path, batch_text=""):
    """
    Convert a 3D volume tensor into an axial-view animated GIF.
    """
    volume = volume.squeeze().cpu().numpy()
    if volume.ndim == 4:  # (C, D, H, W) -> take first channel
        volume = volume[0]

    # Normalize to 0–255
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    volume = (volume * 255).astype(np.uint8)

    frames = []
    for i in range(volume.shape[0]):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(volume[i], cmap='gray')
        ax.axis('off')
        if batch_text:
            ax.set_title(f"{batch_text} - Slice {i+1}/{volume.shape[0]}", fontsize=8)
        fig.canvas.draw()

        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(save_path, frames, duration=0.1)  # ~10 FPS

@rank_zero_only
def visualize_latent_comparison_gif(original, reconstructed, batch_text="", save_path_prefix=None):
    """
    Save two GIFs: one for original latent, one for reconstructed latent.
    """
    assert save_path_prefix is not None, "save_path_prefix must be provided"

    orig_path = f"{save_path_prefix}_original.gif"
    recon_path = f"{save_path_prefix}_reconstructed.gif"

    volume_to_axial_gif(original, orig_path, batch_text=batch_text + " (Original)")
    volume_to_axial_gif(reconstructed, recon_path, batch_text=batch_text + " (Reconstructed)")

    print(f"[Visualization] Saved GIFs:\n - {orig_path}\n - {recon_path}")


# ----------------------------
# Lightning Module
# ----------------------------

class LatentDiffusionLightningModule(LightningModule):
    def __init__(
        self,
        unet_kwargs,
        diffusion_kwargs,
        tokenizer_path,
        learning_rate=2e-4,
        vis_interval=1000,
        output_dir="checkpoints",
        ema_decay=0.995,
        step_start_ema=1000,
        update_ema_every=10
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["unet_kwargs", "diffusion_kwargs"])

        # Tokenizer (CheXbert or HF tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        # Text encoder (CheXbert fine-tuned model, frozen)
        self.text_encoder = AutoModel.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Latent Diffusion Model
        self.diffusion = LatentDiffusion(unet_kwargs, diffusion_kwargs)
        
        self.ema = EMA(ema_decay)
        self.ema_model = deepcopy(self.diffusion)

        for p in self.ema_model.parameters():
            p.requires_grad = False

        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.learning_rate = learning_rate
        self.vis_interval = vis_interval
        self.output_dir = output_dir

    def exists(self, val):
        return val is not None

    def get_text_embeddings(self, texts, pad_id=0., eps=1e-6):
        # Debug what comes from the dataloader
        # print("\n=== DEBUG: Raw texts from dataloader ===")
        # print(texts)
        # print("Type of texts:", type(texts))
        # if isinstance(texts, (list, tuple)):
            # print("Lengths:", [len(str(t)) if isinstance(t, str) else None for t in texts])
        # else:
            # print("Length:", len(str(texts)) if isinstance(texts, str) else None)

        # Tokenize
        tokens = self.tokenizer(
            texts,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        tokens = tokens.to(self.device)

        # Get embeddings
        with torch.no_grad():
            outputs = self.text_encoder(input_ids = tokens.input_ids,
                                       attention_mask = tokens.attention_mask,
                                       output_hidden_states=True)
            hidden_state = outputs.hidden_states[-1]
            return hidden_state

    def training_step(self, batch, batch_idx):
        latents = batch["latent"].float().to(self.device)
        raw_texts = batch["text"]

        # Encode text
        text_embeds = self.get_text_embeddings(raw_texts)

        # Loss
        # print("--"*10, "\n\nCond in unet : ", text_embeds.shape, "\n","--"*10) 
        loss = self.diffusion(latents, cond=text_embeds)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        if (self.global_step >= self.step_start_ema) and (self.global_step % self.update_ema_every == 0):
            self.ema.update_model_average(self.ema_model, self.diffusion)

        # Visualize periodically
        if (self.global_step % self.vis_interval == 0) and (batch_idx == 0):
            self._visualize(latents, text_embeds, raw_texts, device=self.device)

        return loss

    def _visualize(self, latents, text_embeds, raw_texts, device):
        self.ema_model.eval()
        with torch.no_grad():
            recon_latents = self.ema_model.p_sample_loop(
                cond=text_embeds,
                shape=latents.shape,
                device=device

            )
        vis_prefix = os.path.join(self.output_dir, f"vis_step{self.global_step}")
        visualize_latent_comparison_gif(latents[0], recon_latents[0], batch_text=raw_texts[0], save_path_prefix=vis_prefix)

    def configure_optimizers(self):
        return AdamW(self.diffusion.parameters(), lr=self.learning_rate)

    #def on_save_checkpoint(self, checkpoint):
    #    checkpoint["ema_state_dict"] = self.ema_model.state_dict()

    # def on_load_checkpoint(self, checkpoint):
    #     # Optional: Initialize ema_model if it wasn’t saved
    #     ema_keys_missing = not any(k.startswith("ema_model") for k in checkpoint.get("state_dict", {}))

    #     for key in self.state_dict().keys():
    #         if key.startswith("ema_model") and key not in checkpoint["state_dict"]:
    #             print(f"⚠️ Filling missing EMA key: {key}")
    #             checkpoint["state_dict"][key] = self.state_dict()[key]

    #     if ema_keys_missing:
    #         print("⚠️ EMA weights not found in checkpoint. Initializing ema_model from diffusion.")
    #         self.ema_model.load_state_dict(self.diffusion.state_dict())
    #     else:
    #         print("✅ EMA weights found in checkpoint.")

# ----------------------------
# Training Entry Point
# ----------------------------

def train(
    csv_path,
    tokenizer_path,
    unet_kwargs,
    diffusion_kwargs,
    output_dir="checkpoints",
    checkpoint=None,
    batch_size=1,
    num_epochs=1000,
    learning_rate=2e-4,
    log_interval=100,
    vis_interval=1000,
    num_workers=1,
    target_shape=(96, 96, 96),
    gpus=1,
    seed=42
):
    seed_everything(seed)

    dataloader = get_dataloader(
        csv_path=csv_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        target_shape=target_shape
    )

    model = LatentDiffusionLightningModule(
        unet_kwargs=unet_kwargs,
        diffusion_kwargs=diffusion_kwargs,
        tokenizer_path=tokenizer_path,
        learning_rate=learning_rate,
        vis_interval=vis_interval,
        output_dir=output_dir
    )

    logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, "runs"),
        name=datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    checkpoint_callback = ModelCheckpoint(dirpath=output_dir, filename="light_model_{epoch:02d}_step_{step}", save_top_k=1, every_n_train_steps=5000)

    trainer = Trainer(
        max_epochs=num_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=10)],
        log_every_n_steps=log_interval,
        accelerator="gpu" if gpus else "cpu",
        devices=gpus,
        precision=16 if torch.cuda.is_available() else 32,
    )
    if checkpoint is None:
        print("No checkpoint provided, starting training from scratch.")
        trainer.fit(model, dataloader)
    else:
        trainer.fit(model, dataloader, ckpt_path=checkpoint)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--checkpoint_path", required=False, default=None)
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    unet_kwargs = {
        "dim": 32,
        "channels": 1,
        "cond_dim": 768,
        "total_slices": 96,
        "use_bert_text_cond": True,
    }

    diffusion_kwargs = {
        "image_size": 128,
        "num_frames": 96,
        "text_use_bert_cls": True,
        "channels": 1,
        "ddim_timesteps": 75,
        "loss_type": "edge+l2",
    }

    train(
        csv_path=args.csv_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        unet_kwargs=unet_kwargs,
        diffusion_kwargs=diffusion_kwargs,
        checkpoint=args.checkpoint_path,
        gpus=args.gpus
    )


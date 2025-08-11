import torch
import os
import nibabel as nib
import numpy as np
from pathlib import Path
import sys
import json
from tqdm import tqdm
import zipfile
import shutil

import SimpleITK as sitk
# -----------------------------
# Import MedVAE (local)
# -----------------------------
from model.MedVAE.medvae.medvae_main import MVAE

from model.text_to_latent_diffusion.train_text_to_latent_diff_model import LatentDiffusionLightningModule 
from model.text_to_latent_diffusion.text_to_latent_dataloader import GLOBAL_MEAN, GLOBAL_STD

INPUT_DIR = Path("/input")
OUTPUT_DIR = Path("/output")
TMP_DIR = Path("/tmp")
MODEL_PATH = Path("/opt/app/")


# Final Output Parameters
FINAL_SPACING = (1.0, 1.0, 1.0)  # (x, y, z) mm
HU_MIN, HU_MAX = -1000.0, 1000.0


@torch.inference_mode()
def to_hu_tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Maps MedVAE decoder output to HU range [-1000, 1000] via:
    tanh -> (0,1) -> HU
    All ops are GPU friendly.
    """
    # Ensure float32 to avoid precision loss
    x = x.float()

    # Step 1: Tanh activation → (-1, 1)
    x = torch.tanh(x)

    # Step 2: Shift/scale → (0, 1)
    x = (x + 1.0) / 2.0

    # Step 3: Scale → (-1000, 1000)
    x = x * (HU_MAX - HU_MIN) + HU_MIN

    return x.clamp(HU_MIN, HU_MAX)


def save_mha_hu(decoded: torch.Tensor, out_path: Path):
    """
    decoded: [B,1,D,H,W] or [1,D,H,W] tensor from MedVAE decoder (unknown scale).
    Writes .mha as HU with given voxel spacing (W,H,D order in SITK comes via array depth).
    """
    hu = to_hu_tanh(decoded)  # [B,1,D,H,W]
    vol = hu[0,0].detach().cpu().numpy()  # (D,H,W)

    itk_img = sitk.GetImageFromArray(vol)  # interprets array as z,y,x -> (D,H,W)
    itk_img.SetSpacing(tuple(map(float, FINAL_SPACING)))  # (sx, sy, sz)
    itk_img.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(itk_img, str(out_path))

def main():
    try:
        input_json_path = next(INPUT_DIR.glob("*.json"))
        print(f"Found input JSON: {input_json_path}")
    except StopIteration:
        print(f"✗ No JSON file found in {INPUT_DIR}")
        sys.exit(1)

    with input_json_path.open("r") as f:
        prompts_data = json.load(f)

    if not isinstance(prompts_data, list):
        print("✗ Input JSON must be a list of objects.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # Load MedVAE decoder (offline)
    # -----------------------------
    medvae_weights_dir = os.path.join(MODEL_PATH, "model", "MedVAE")
    medvae = MVAE(
        model_name="medvae_4_1_3d",
        modality="ct",
        model_weights_dir=medvae_weights_dir  # This should load your offline ckpt
    )
    medvae.eval().to(device)

    # -----------------------------
    # Load text-to-latent diffusion model
    # -----------------------------
    diffusion_ckpt_path = os.path.join(MODEL_PATH, "model", "weights", "model_checkpoint.ckpt")
    tokenizer_path = os.path.join(MODEL_PATH, "model", "weights",  "bert_finetuned")
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
        "ddim_timesteps": 75
    }


    light_model = LatentDiffusionLightningModule.load_from_checkpoint(
        diffusion_ckpt_path,
        map_location=device,
        unet_kwargs=unet_kwargs,
        diffusion_kwargs=diffusion_kwargs,
        tokenizer_path=tokenizer_path,
    )
    light_model.eval().to(device)

    # -----------------------------
    # Generate latent from text
    # -----------------------------
    
    for item in tqdm(prompts_data, desc="Generating CTs"):
            prompt = item["report"]
            output_filename = Path(item["input_image_name"]).stem
            tqdm.write(f"\nProcessing: {output_filename}")

            tqdm.write("  → Stage 1: Encoding text")

            cond = light_model.get_text_embeddings([prompt])

            tqdm.write("  → Stage 2: generating latent from text")
            with torch.no_grad():
                recon_latents = light_model.ema_model.p_sample_loop(
                    cond=cond,
                    shape=(1,1,96,96,96),
                    device=cond.device,
                    use_ddim=True,
                    clip_denoised=False,
                )
            
            tqdm.write("  → Stage 3: Decoding latent to CT volume")
            rescaled_latents = recon_latents * GLOBAL_STD + GLOBAL_MEAN

            with torch.no_grad():
                decoded_latent = medvae.decode(rescaled_latents)
            
            save_mha_hu(decoded_latent, TMP_DIR / f"{output_filename}.mha")
            
            tqdm.write(f"  → Saved: {output_filename}.mha")
        
    # Define a temporary path for the zip file
    tmp_zip = TMP_DIR / "predictions.zip"
    tqdm.write(f"Creating zip file: {tmp_zip}")

    # Create the zip archive with DEFLATE compression
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for mha in sorted(TMP_DIR.glob("*.mha")):
            # The write method works similarly to tar.add
            zipf.write(mha, arcname=mha.name)  # flat, no nested folders

    tqdm.write(f"Zip file created: {tmp_zip}")

    # Move zip file to /output
    final_zip_path = OUTPUT_DIR / "predictions.zip"
    # If /output/output.zip exists, remove it first to avoid errors
    if final_zip_path.exists():
        final_zip_path.unlink()
    try:
        tmp_zip.replace(final_zip_path)      # fast move if same filesystem
    except OSError:
        shutil.move(str(tmp_zip), str(final_zip_path)) # cross-device safe

    tqdm.write(f"Moved zip file to: {final_zip_path}")
    tqdm.write("All CTs generated and saved successfully.")


if __name__ == "__main__":

    main()

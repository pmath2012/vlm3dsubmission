import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Transpose, Compose, EnsureChannelFirst, ResizeWithPadOrCrop, Resize, ToTensor, Transform

GLOBAL_MEAN = -20.029061
GLOBAL_STD = 13.203140

class Normalize(Transform):
    def __call__(self, data):
        data = data.astype(np.float32)
        return (data - GLOBAL_MEAN) / GLOBAL_STD

def get_latent_transforms(target_shape, crop_shape):
    """
    Returns MONAI array transforms for latents (already loaded into memory).
    shape: full pad size (D, H, W)
    crop_shape: crop size (D, H, W)
    """
    return Compose([
        EnsureChannelFirst(channel_dim="no_channel"),  # (1, H, W, D)
        Transpose((0, 3, 1, 2)),  # (1, D, H, W)
        Resize(spatial_size=(None, target_shape[1], target_shape[2]),  # (D, H, W)
            mode="trilinear",
            align_corners=True,
            anti_aliasing=True
        ),
        ResizeWithPadOrCrop(spatial_size=target_shape),
        Normalize(),
        ToTensor()
    ])

class TextLatentDataset(Dataset):
    def __init__(self, dataframe, transforms=None):
        self.df = dataframe
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load latent
        latent = np.load(row["latent_path"])
        if self.transforms:
            latent = self.transforms(latent)  # apply array transforms

        # Raw text only
        text = row["impressions"]

        return {
            "latent": latent.float(),
            "text": text
        }

def get_dataloader(csv_path, batch_size, shuffle, num_workers, target_shape):
    df = pd.read_csv(csv_path)
    transforms = get_latent_transforms(target_shape=target_shape, crop_shape=target_shape)
    dataset = TextLatentDataset(df, transforms=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


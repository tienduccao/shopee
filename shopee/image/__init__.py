
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import ImageInferenceDataset


def get_embeddings(
    data: pd.DataFrame, 
    images_path: str, 
    model: torch.nn.Module, 
    batch_size: int = 64,
    num_workers: int = 2,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute image embeddings from model

    Args:
        data: shopee dataframe (test/train)
        images_path: path to directory containing images
        model: the Pytorch model for computing image embeddings
        batch_size: batch_size used for computing embeddings
        num_workers: number of worker for the Pytorch Dataloader
        device: the Pytorch device ("cpu", "cuda", "cuda:0", etc)

    Returns:
        a Pytorch tensor with shape (N, C) containing image embeddings,
            where N is the number of images and C the embedding dimension
    """
    print("Computing image embeddings...")
    image_dataset = ImageInferenceDataset(
        data,
        images_path
    )
    image_loader = DataLoader(
        image_dataset, 
        batch_size, 
        num_workers=num_workers
    )
    
    model.eval()
    model.to(device)

    with torch.no_grad():
        image_embeddings = torch.cat([
            model(batch.to(device))
            for batch in tqdm(image_loader)
        ]).squeeze()

    print("image embeddings shape", image_embeddings.shape)

    return image_embeddings
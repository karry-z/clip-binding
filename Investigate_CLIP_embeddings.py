# Investigate CLIP embeddings

# Lib
import torch
from train import *
import pandas as pd
import numpy as np




# Config
class Config:
    model_name = "clip"
    dataset = "rel"
    lr = 1e-06
    weight_decay = 1e-05
    clip_model = "ViT-L/14"
    epochs = 20
    train_batch_size = 1
    eval_batch_size = 1
    gradient_accumulation_steps = 1
    evaluate_only = False
    context_length = 77
    emb_dim = 768
    attr_dropout = 0.2
    save_dir = "save"
    save_model = True
    seed = 0

config = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"





def get_distance_sample_mean(model, dataset, device):
    df = pd.DataFrame(columns=['aRb', 'bRa', 'aSb', 'cRb', 'aRc'])
    dataloader = DataLoader(dataset, batch_size=config.eval_batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        for i, (batch_img, texts, label) in enumerate(dataloader):
            logits_per_image = model(batch_img.to(device), texts)
            aRb = logits_per_image[:, 0].mean().item()
            bRa = logits_per_image[:, 1].mean().item()
            aSb = logits_per_image[:, 2].mean().item()
            cRb = logits_per_image[:, 3].mean().item()
            aRc = logits_per_image[:, 4].mean().item()
            df.loc[i] = [aRb, bRa, aSb, cRb, aRc]
    return df



if __name__ == '__main__':
    dataset = choose_dataset(config)
    model, optimizer = get_model(dataset["train"], config, device)
    for split in ["train", "val", "gen"]:
        df = get_distance_sample_mean(model, dataset[split], device)
        df.to_csv(f'clip_embeddings_distance_sample_mean_{split}.csv', index=False)
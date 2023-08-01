# Investigate CLIP embeddings

# Lib
import random
from PIL import Image
import types
from datasets.clevr_dataset import preprocess, RelDataset
import torch
from models import get_model

from torch.utils.data.dataloader import DataLoader
from utils import set_seed, save_predictions
import pandas as pd
import numpy as np

import os

import argparse



def cos_sim_img_txt_samples(model, dataset, device):
    df_logits = pd.DataFrame(columns=['aRb', 'bRa', 'aSb', 'cRb', 'aRc'])
    dataloader = DataLoader(dataset, batch_size=config.eval_batch_size, shuffle=False)
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for i, (batch_img, texts, label) in enumerate(dataloader):
            logits_per_image = model(batch_img.to(device), texts)
            aRb = logits_per_image[:, 0].item()
            bRa = logits_per_image[:, 1].item()
            aSb = logits_per_image[:, 2].item()
            cRb = logits_per_image[:, 3].item()
            aRc = logits_per_image[:, 4].item()
            df_logits.loc[i] = [aRb, bRa, aSb, cRb, aRc]
            true_labels.append(texts[0][0])
            pred_labels.append(texts[df_logits.loc[i].argmax()][0])
            print(f"Batch {i} done")
    return df_logits, true_labels, pred_labels



def get_true_pred_labels(model, dataset, config, device):
    dataloader = DataLoader(dataset, batch_size=config.eval_batch_size, shuffle=False)
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for i, (batch_img, texts, label) in enumerate(dataloader):
            logits_per_image,_,_ = model(batch_img.to(device), texts)
            true_labels.extend(texts[0])
            pred_labels.extend([texts[ctx][batch] for batch,ctx in enumerate(logits_per_image.argmax(axis=1))])
            print(f"Batch {i} done")
    return true_labels, pred_labels

def getitem(self, idx):
    img_path = self.img_dir + self.ims_labels[idx][0]
    image = Image.open(img_path)  # Image from PIL module
    image = preprocess(image)

    # for regular train/validation/testing
    subj, rel, obj = self.ims_labels[idx][1].strip().split()

    # Distractors have the following structure. If the true label is aRb,
    # the distractors are bRa, aSb, cRb, aRc, where S is the opposite relation to R
    # and c is an object not equal to a or b
    distractors = []
    distractors.append(f"{obj} {rel} {subj}")
    distractors.append(f"{subj} {self.rel_opposites[rel]} {obj}")
    # since there are always three nouns, this shouldn't make a difference.
    other_nouns = list(set(self.nouns).difference(set([subj, obj])))
    assert len(other_nouns) == 1
    other_noun = other_nouns[0]

    # other_noun = random.choice(other_nouns)
    distractors.append(f"{other_noun} {rel} {obj}")
    distractors.append(f"{subj} {rel} {other_noun}")
    texts = [self.ims_labels[idx][1]] + distractors

    # shuffle the texts and return the label of the correct text 
    indices = list(range(len(texts)))
    texts = [texts[i] for i in indices]
    label = indices.index(0)

    return image, texts, label

RelDataset.__getitem__ = getitem

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", help="name of the experiment", type=str, default="T5_Text"
    )
    parser.add_argument(
        "--dataset", help="name of the dataset", type=str, default="rel"
    )
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-06)
    parser.add_argument(
        "--weight_decay", help="weight decay", type=float, default=1e-05
    )
    parser.add_argument(
        "--clip_model", help="clip model type", type=str, default="ViT-L/14"
    )
    parser.add_argument(
        "--eval_batch_size", help="eval batch size", default=3, type=int
    )
    parser.add_argument(
        "--context_length",
        help="Sets the context length of the clip model. This is used only in CSP.",
        default=77,
        type=int,
    )
    parser.add_argument(
        "--emb_dim",
        help="Embedding dimension.",
        default=768,
        type=int,
    )
    parser.add_argument(
        "--attr_dropout",
        help="add dropout to attributes",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--model_path",
        help="path to the model",
        type=str,
        default='/user/work/pu22650/clip-binding-out/T5_FT_rel_5/final_model.pt'
    )
    parser.add_argument("--save_dir", help="save path", type=str, default="/user/work/pu22650/clip-binding-out/T5_FT_rel_5")
    parser.add_argument("--seed", help="seed value", default=0, type=int)
    config = parser.parse_args()

    # set the seed value
    set_seed(config.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset = {split: RelDataset(split) for split in ["train", "val", "gen"]}
    model, optimizer = get_model(dataset['train'], config, device)
    model = model.float()
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    for split in ["train", "val", "gen"]:
        # df, shape_true, shape_pred, rel_true, rel_pred, true, pred = cos_sim_img_txt_samples(model, dataset[split], device)
        # df.to_csv(os.path.join(save_dir, f'cos_sim_{split}.csv'), index=False)
        # # write shape (true, pred) and relation to csv, respectively
        # df_shape = pd.DataFrame(list(zip(shape_true, shape_pred)), columns=['true', 'pred'])
        # df_shape.to_csv(os.path.join(save_dir, f'shape_{split}.csv'), index=False)
        # df_rel = pd.DataFrame(list(zip(rel_true, rel_pred)), columns=['true', 'pred'])
        # df_rel.to_csv(os.path.join(save_dir, f'rel_{split}.csv'), index=False)
        true, pred = get_true_pred_labels(model, dataset[split], config, device)
        pd.DataFrame(list(zip(true, pred)), columns=['true', 'pred']).to_csv(os.path.join(config.save_dir, f'contexts_{split}.csv'), index=False)

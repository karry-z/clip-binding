import gc
import random
from PIL import Image
import types
from datasets.clevr_dataset import preprocess
import torch
from train import *
import pandas as pd
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch import nn
import torch.nn.functional as F
import wandb
wandb.login(key='94cf1e387f47487c4a61b27307ed870991ba63c1')
# Config
class Config:
    model_name = "clip"
    dataset = "rel"
    lr = 1e-07
    weight_decay = 1e-05
    clip_model = "ViT-L/14"
    epochs = 5
    train_batch_size = 32
    eval_batch_size = 64
    gradient_accumulation_steps = 1
    evaluate_only = False
    context_length = 77
    emb_dim = 768
    attr_dropout = 0.2
    save_dir = "/user/work/pu22650/clip-binding-out/compositional_objective_MLP"
    save_model = True
    seed = 0

config = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"
wandb.init(
    # set the wandb project where this run will be logged
    project="clip-binding",
    entity="karry-z",
    name="compositional_objective_MLP",
    # track hyperparameters and run metadata
    # config=dict
)

dataset = {split: RelDataset(split) for split in ["train", "val", "gen"]}
model, optimizer = get_model(dataset["train"], config, device)
model.mlp = torch.nn.Sequential(
    torch.nn.Linear(config.emb_dim*3, 300),
    torch.nn.ReLU(),
    torch.nn.Linear(300, 300),
    torch.nn.ReLU(),
    torch.nn.Linear(300, config.emb_dim),
) 
model.to(device)
model = model.to(torch.float32)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.lr,
    weight_decay=config.weight_decay,
    eps=1e-6,
)




mse = nn.MSELoss()


trainloader = DataLoader(dataset["train"], batch_size=config.train_batch_size, shuffle=False)
valloader = DataLoader(dataset["val"], batch_size=config.eval_batch_size, shuffle=False)
genloader = DataLoader(dataset["gen"], batch_size=config.eval_batch_size, shuffle=False)

def compute_threshold_acc_on_sentence_level_emb(emb_txt, emb_com, threshold=10):
    correct_l = [F.mse_loss(emb_txt[i, :], emb_com[i, :]).item() < threshold for i in range(emb_txt.shape[0])]
    acc = sum(correct_l) / len(correct_l)
    return acc

def compute_combined_embeddings(model, texts):
    sub_l = []
    rel_l = []
    obj_l = []
    for sample in texts:
        for batch in sample:
            sub, rel, obj = batch.split()
            sub_l.append(sub)
            rel_l.append(rel)
            obj_l.append(obj)
    texts_split = [sub_l, rel_l, obj_l]
    with torch.no_grad():
        emb_sub, emb_rel, emb_obj = model.compute_text_representations(texts_split).view(-1, 3, 768).permute(1, 0, 2)
    emb_com = model.mlp(torch.cat([emb_sub, emb_rel, emb_obj], dim=-1))
    return emb_com

best_val_acc = 0
for epoch in range(config.epochs):
    model.train()
    epoch_train_losses = []
    train_acc_l = [] 
    for id, (batch_img, texts, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        emb_com = compute_combined_embeddings(model, texts) # get combined embeddings
        torch.cuda.empty_cache()
        emb_txt = model.compute_text_representations(texts) # get text embeddings
        loss = mse(emb_txt, emb_com)
        epoch_train_losses.append(loss.item())
        train_acc_l.append(compute_threshold_acc_on_sentence_level_emb(emb_txt, emb_com))
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
    logger.info(f"Epoch {epoch} training Loss: {np.mean(epoch_train_losses):.2f}")
    logger.info(f"Epoch {epoch} thrs10 training acc *100: {100*np.mean(train_acc_l):.2f}")

    model.eval()
    val_acc_l = []
    epoch_val_losses = []
    torch.cuda.empty_cache()
    with torch.no_grad():
        for id, (batch_img, texts, labels) in enumerate(valloader):
            emb_com = compute_combined_embeddings(model, texts) # get combined embeddings
            emb_txt = model.compute_text_representations(texts) # get text embeddings
            loss = mse(emb_txt, emb_com)
            epoch_val_losses.append(loss.item())
            val_acc_l.append(compute_threshold_acc_on_sentence_level_emb(emb_txt, emb_com))
    logger.info(f"Epoch {epoch} Val Loss: {np.mean(epoch_val_losses):.2f}")
    logger.info(f"Epoch {epoch} thrs10 Val acc *100: {100*np.mean(val_acc_l):.2f}")
    
    torch.cuda.empty_cache()
    # save best model
    if config.save_model and np.mean(val_acc_l) > best_val_acc:
        best_val_acc = np.mean(val_acc_l)
        torch.save(model.state_dict(), os.path.join(config.save_dir, f"clip_compositional_best.pt"))

    torch.cuda.empty_cache()
    model.eval()
    gen_acc_l = []
    epoch_gen_losses = []
    with torch.no_grad():
        for id, (batch_img, texts, labels) in enumerate(genloader):
            emb_com = compute_combined_embeddings(model, texts) # get combined embeddings
            emb_txt = model.compute_text_representations(texts) # get text embeddings
            loss = mse(emb_txt, emb_com)
            epoch_gen_losses.append(loss.item())
            gen_acc_l.append(compute_threshold_acc_on_sentence_level_emb(emb_txt, emb_com))
    logger.info(f"Epoch {epoch} Gen Loss: {np.mean(epoch_gen_losses):.2f}")
    logger.info(f"Epoch {epoch} thrs10 Gen acc *100: {100*np.mean(gen_acc_l):.2f}")
    wandb.log({
        "train": {'loss':np.mean(epoch_train_losses), 'acc': np.mean(train_acc_l)},
        "val": {'loss':np.mean(epoch_val_losses), 'acc': np.mean(val_acc_l)},
        "gen": {'loss':np.mean(epoch_gen_losses), 'acc': np.mean(gen_acc_l)}
    })



# save model
if config.save_model:
    torch.save(model.state_dict(), os.path.join(config.save_dir, f"clip_compositional.pt"))
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

# Config
class Config:
    model_name = "clip"
    dataset = "rel"
    lr = 1e-06
    weight_decay = 1e-05
    clip_model = "ViT-L/14"
    epochs = 5
    train_batch_size = 4
    eval_batch_size = 8
    gradient_accumulation_steps = 1
    evaluate_only = False
    context_length = 77
    emb_dim = 768
    attr_dropout = 0.2
    save_dir = "/user/work/pu22650/clip-binding-out/compositional_objective"
    save_model = True
    seed = 0

config = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"


dataset = {split: RelDataset(split) for split in ["train", "val", "gen"]}
model, optimizer = get_model(dataset["train"], config, device)
model.eleLinearWeight = nn.Parameter(torch.empty(3).uniform_(-0.1, 0.1))
model.eleLinearSoftmax = nn.Softmax(dim=0)

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
    emb_sub, emb_rel, emb_obj = model.compute_text_representations(texts_split).view(-1, 3, 768).permute(1, 0, 2)
    model.eleLinearWeight = model.eleLinearSoftmax(model.eleLinearWeight)
    emb_com = model.eleLinearWeight[0] * emb_sub + model.eleLinearWeight[1] * emb_rel + model.eleLinearWeight[2] * emb_obj
    return emb_com

best_val_acc = 0
for epoch in range(config.epochs):
    model.train()
    epoch_train_losses = []
    train_acc_l = []
    for id, (batch_img, texts, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        emb_com = compute_combined_embeddings(model, texts) # get combined embeddings
        emb_txt = model.compute_text_representations(texts) # get text embeddings
        loss = mse(emb_txt, emb_com)
        epoch_train_losses.append(loss.item())
        cos_sim = F.cosine_similarity(emb_txt, emb_com, dim=-1)
        train_acc_l.append(torch.mean((cos_sim > 0.9).float()).item())
        loss.backward()
        optimizer.step()
        

    model.eval()
    val_acc_l = []
    for id, (batch_img, texts, labels) in enumerate(valloader):
        emb_com = compute_combined_embeddings(model, texts) # get combined embeddings
        emb_txt = model.compute_text_representations(texts) # get text embeddings
        cos_sim = F.cosine_similarity(emb_txt, emb_com, dim=-1)
        val_acc_l.append(torch.mean((cos_sim > 0.9).float()).item())
    logger.info(f"Epoch {epoch} Loss: {np.mean(epoch_train_losses):.2f}")
    logger.info(f"Epoch {epoch} Train Accuracy: {100*np.mean(train_acc_l):.2f}")
    logger.info(f"Epoch {epoch} Val Accuracy: {100*np.mean(val_acc_l):.2f}")
    # save best model
    if config.save_model and np.mean(val_acc_l) > best_val_acc:
        best_val_acc = np.mean(val_acc_l)
        torch.save(model.state_dict(), os.path.join(config.save_dir, f"clip_compositional_best.pt"))


model.eval()
gen_acc_l = []
for id, (batch_img, texts, labels) in enumerate(genloader):
    emb_com = compute_combined_embeddings(model, texts) # get combined embeddings
    emb_txt = model.compute_text_representations(texts) # get text embeddings
    cos_sim = F.cosine_similarity(emb_txt, emb_com, dim=-1)
    gen_acc_l.append(torch.mean((cos_sim > 0.9).float()).item())
    if id >1:
            break
logger.info(f"Epoch {epoch} Gen Accuracy: {100*np.mean(gen_acc_l):.2f}")



# save model
if config.save_model:
    torch.save(model.state_dict(), os.path.join(config.save_dir, f"clip_compositional.pt"))
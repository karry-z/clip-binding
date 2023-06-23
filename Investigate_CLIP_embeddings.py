# Investigate CLIP embeddings

# Lib
import torch
from train import *
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.colors as mcolors

# Config
class Config:
    model_name = "clip"
    dataset = "rel"
    lr = 1e-06
    weight_decay = 1e-05
    clip_model = "ViT-L/14"
    epochs = 20
    train_batch_size = 4
    eval_batch_size = 4
    gradient_accumulation_steps = 1
    evaluate_only = False
    context_length = 77
    emb_dim = 768
    attr_dropout = 0.2
    save_dir = "save"
    save_model = True
    seed = 0

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
dataset = choose_dataset(config)
model, optimizer = get_model(dataset["train"], config, device)

dataloader = DataLoader(dataset['val'], batch_size=config.eval_batch_size, shuffle=False)

# Inference
pos_mean = []
neg_mean = []

model.eval()
with torch.no_grad():
    for i, (batch_img, texts, labels) in enumerate(dataloader):
        logits_per_image = model(batch_img.to(device), texts)
        pos, neg = [],[]
        for raw in range(len(labels)):
            pos.append(logits_per_image[raw, labels[raw]])
            for col in range(5):
                if col != labels[raw]:
                    neg.append(logits_per_image[raw, col])
        pos = torch.stack(pos)
        neg = torch.stack(neg)
        # compute the mean value of pos and neg in each batch
        pos_mean.append(pos.mean().item())
        neg_mean.append(neg.mean().item())


# save pos_mean and neg_mean in csv
df = pd.DataFrame({'pos_mean': pos_mean, 'neg_mean': neg_mean})
df.to_csv('clip_embeddings_val.csv', index=False)


# def normal_dist(x, mu, sigma):
#     return np.exp(-(x-mu)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma)

# # df = pd.read_csv('clip_embeddings.csv')
# X = np.linspace(15, 25, 500)
# fig, ax = plt.subplots()
# ax.set_title("distribution of sample mean of CLIP embeddings")
# hist_neg = ax.hist(df['neg_mean'], bins='auto', density=True, label='neg hist', alpha = 0.3, color=mcolors.CSS4_COLORS['blue'])
# hist_pos = ax.hist(df['pos_mean'], bins='auto', density=True, label='pos hist', alpha = 0.3, color=mcolors.CSS4_COLORS['green'])
# neg_dist = stats.rv_histogram(hist_neg[:2])
# pos_dist = stats.rv_histogram(hist_pos[:2])

# ax.plot(X, normal_dist(X, neg_dist.mean(), neg_dist.std()), label='pdf of neg', color=mcolors.CSS4_COLORS['blue'])
# ax.vlines(neg_dist.mean(), 0, 3, colors=mcolors.CSS4_COLORS['darkblue'], label='neg mean', linestyle='--')
# print(neg_dist.mean(), neg_dist.std())

# ax.plot(X, normal_dist(X, pos_dist.mean(), pos_dist.std()), label='pdf of pos', color=mcolors.CSS4_COLORS['green'])
# ax.vlines(pos_dist.mean(), 0, 3, colors=mcolors.CSS4_COLORS['darkgreen'], label='pos mean', linestyle='--')
# print(pos_dist.mean(), pos_dist.std())
# ax.legend()
# ax.set_xlabel('sample mean')
# ax.set_ylabel('probability density')
# ax.set_xlim(19, 23)
# ax.set_ylim(0, 1)
# fig.show()

# stats.ttest_ind(a=df['neg_mean'], b=df['pos_mean']) 


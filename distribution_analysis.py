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
from transformers import AutoTokenizer, T5EncoderModel
import os


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


set_seed(config.seed)


def cos_sim_img_txt_samples(model, dataset, device):
    df_logits = pd.DataFrame(columns=['aRb', 'bRa', 'aSb', 'cRb', 'aRc'])
    dataloader = DataLoader(dataset, batch_size=config.eval_batch_size, shuffle=False)
    model.eval()
    shape_true = []
    shape_pred = []
    rel_true = []
    rel_pred = []
    with torch.no_grad():
        for i, (batch_img, texts, label) in enumerate(dataloader):
            logits_per_image = model(batch_img.to(device), texts)
            aRb = logits_per_image[:, 0].mean().item()
            bRa = logits_per_image[:, 1].mean().item()
            aSb = logits_per_image[:, 2].mean().item()
            cRb = logits_per_image[:, 3].mean().item()
            aRc = logits_per_image[:, 4].mean().item()
            df_logits.loc[i] = [aRb, bRa, aSb, cRb, aRc]
            real_l = texts[0][0].split()
            pred_l = texts[df_logits.loc[i].argmax()][0].split()
            shape_true.append(real_l[0])
            shape_true.append(real_l[2])
            shape_pred.append(pred_l[0])
            shape_pred.append(pred_l[2])
            rel_true.append(real_l[1])
            rel_pred.append(pred_l[1])
            
    return df_logits, shape_true, shape_pred, rel_true, rel_pred


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

# define get model function
def get_T5_FT_model(config, device):
    model, optimizer = get_model(None, config, device)
    
    model.t5 = T5EncoderModel.from_pretrained("t5-small")
    model.t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
    # add a mlp layer to map the t5 output to the same dimension as clip
    model.mlp = torch.nn.Sequential(
        torch.nn.Linear(512, 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, config.emb_dim),
    ) 
    model.to(device)
    ckpt=torch.load('/user/work/pu22650/clip-binding-out/T5_FT_rel_5/final_model.pt', map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model = model.float()
    # load mlp weights
    import types
    def forward_(self, batch_images, texts):
        texts = list(map(list, zip(*texts)))
        bsz = len(texts)
        num_captions = len(texts[0])

        t5_output = []
        for text in texts:
            encoded_input = self.t5_tokenizer(text, return_tensors='pt')
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            last_hidden_state = self.t5(**encoded_input).last_hidden_state
            pooled_output = torch.mean(last_hidden_state, dim=1)
            t5_output.append(self.mlp(pooled_output))
        t5_output = torch.concat(t5_output)
        text_features = t5_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view([bsz, num_captions, -1])

        batch_images = batch_images.to(self.device)
        batch_feat = self.encode_image(batch_images)
        

        batch_feat = batch_feat / batch_feat.norm(dim=-1, keepdim=True)
        batch_feat = batch_feat.unsqueeze(2)

        logits_per_image = self.clip_model.logit_scale.exp() * torch.bmm(
            text_features, batch_feat
        )
        logits_per_image = logits_per_image.squeeze(2)
        return logits_per_image
    
    model.forward = types.MethodType(forward_, model)
    return model

if __name__ == '__main__':
    dataset = {split: RelDataset(split) for split in ["train", "val", "gen"]}
    # TODO: change model
    model = get_T5_FT_model(config, device)
    # TODO: change path
    save_dir = "/user/work/pu22650/clip-binding-out/T5_FT_rel_5"
    for split in ["train", "val", "gen"]:
        df, shape_true, shape_pred, rel_true, rel_pred = cos_sim_img_txt_samples(model, dataset[split], device)
        df.to_csv(os.path.join(save_dir, f'cos_sim_{split}.csv'), index=False)
        # write shape (true, pred) and relation to csv, respectively
        df_shape = pd.DataFrame(list(zip(shape_true, shape_pred)), columns=['true', 'pred'])
        df_shape.to_csv(os.path.join(save_dir, f'shape_{split}.csv'), index=False)
        df_rel = pd.DataFrame(list(zip(rel_true, rel_pred)), columns=['true', 'pred'])
        df_rel.to_csv(os.path.join(save_dir, f'rel_{split}.csv'), index=False)



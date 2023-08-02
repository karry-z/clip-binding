import torch
from torch import nn
import clip

class CompositionalCLIP(nn.Module):
    r'''get emb_com, emb_txt and logits_per_image from CLIP
    '''
    def __init__(self, config, device):
        super().__init__()
        self.clip_model, self.preprocess = clip.load(config.clip_model, device=device)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(config.emb_dim*3, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, config.emb_dim),
        ) 
        self.mlp = self.mlp.to(device).to(self.clip_model.dtype)
        self.device = device


    def forward(self, batch_images, texts):
        texts = list(map(list, zip(*texts)))
        bsz = len(texts)
        num_captions = len(texts[0])

        emb_com = self.encode_text_samples_compositional(texts)
        emb_txt = self.encode_text_samples(texts)
        emb_img = self.encode_image(batch_images.to(self.device))

        emb_txt_norm = emb_txt / emb_txt.norm(dim=-1, keepdim=True)
        emb_txt_norm = emb_txt_norm.view([bsz, num_captions, -1])

        emb_img_norm = emb_img / emb_img.norm(dim=-1, keepdim=True)
        emb_img_norm = emb_img_norm.unsqueeze(2)

        logits_per_image = self.clip_model.logit_scale.exp() * torch.bmm(
            emb_txt_norm, emb_img_norm
        )
        logits_per_image = logits_per_image.squeeze(2)
        return logits_per_image, emb_com, emb_txt
    
    def combine_embeddings(self, emb_sub, emb_rel, emb_obj):
        return self.mlp(torch.cat([emb_sub, emb_rel, emb_obj], dim=-1))

    def encode_text_samples(self, texts):
        tokenized_text = [
            clip.tokenize(["a photo of" + t for t in _texts]) for _texts in texts
        ]
        tokenized_text = torch.stack(tokenized_text, dim=0)
        bsz, num_captions, padding_length = tokenized_text.shape
        batch_texts = tokenized_text.view([-1, padding_length]).to(self.device)
        text_features = self.clip_model.encode_text(batch_texts)
        return text_features
    
    def encode_text_samples_compositional(self, texts):
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
            emb_sub, emb_rel, emb_obj = self.encode_text_samples(texts_split).view(-1, 3, 768).permute(1, 0, 2)
        emb_com = self.combine_embeddings(emb_sub, emb_rel, emb_obj)
        return emb_com
    
    def encode_image(self, images):
        return self.clip_model.encode_image(images)
    


def get_model_(config, device):
    # load the model
    model = CompositionalCLIP(config, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        eps=1e-6,
    )
    if config.model_path:
        model.load_state_dict(torch.load(config.model_path, map_location=device), strict=False)
    return model, optimizer
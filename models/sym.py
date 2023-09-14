import torch
from torch import nn
import clip


class Sym(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.obj2sym = {
            "cube":[0, 0, 1],
            "sphere":[0, 1, 0],
            "cylinder":[1, 0, 0]
        }
        self.rel2sym = {
            "front":[0, 0, 0, 1],
            "behind":[0, 0 , 1, 0],
            "left":[0, 1, 0, 0],
            "right":[1, 0, 0, 0]
        }
    def forward(self, texts):
        sym = []
        for text_batch in texts:
            sym_batch = []
            for text in text_batch:
                a, r, b = text.split()
                sym_batch.append(self.obj2sym[a] + self.rel2sym[r] + self.obj2sym[b])
            sym.append(sym_batch)
        sym = torch.tensor(sym, dtype=torch.float32, device=self.device)
        sym = sym.view(-1, 10) 
        return sym

class MLP(nn.Module):
    def __init__(self, config, device) -> None:
        super().__init__()
        self.device = device
        self.h_0 = torch.nn.Sequential(
            torch.nn.Linear(config.emb_dim+10, config.neurons_per_layer),
            torch.nn.ReLU()
        )
        self.h_mid = [torch.nn.Sequential(torch.nn.Linear(config.neurons_per_layer, config.neurons_per_layer), torch.nn.ReLU()) for _ in range(config.hidden_layers-1)]
        self.out = torch.nn.Sequential(
            torch.nn.Linear(config.neurons_per_layer, config.emb_dim),
        )

    def forward(self, x):
        x = self.h_0(x)
        if self.h_mid != []:
            for h in self.h_mid:
                h = h.to(self.device).to(x.dtype)
                x = h(x)
        x = self.out(x)
        return x


class CLIP_Sym(nn.Module):
    def __init__(self, config, device) -> None:
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load(config.clip_model, device=device)
        self.sym = Sym(config, device)
        self.mlp = MLP(config, device)
        self.mlp = self.mlp.to(device).to(self.clip_model.dtype)
        # freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, batch_images, texts):
        texts = list(map(list, zip(*texts)))
        bsz = len(texts)
        num_captions = len(texts[0])

        sym_features = self.sym(texts).to(self.clip_model.dtype)
        clip_features = self.encode_clip_texts(texts)
        text_features = self.mlp(torch.cat([clip_features, sym_features], dim=-1))
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

    
    def encode_clip_texts(self, texts):
        tokenized_text = [
            clip.tokenize(["a photo of" + t for t in _texts]) for _texts in texts
        ]
        tokenized_text = torch.stack(tokenized_text, dim=0)
        bsz, num_captions, padding_length = tokenized_text.shape
        batch_texts = tokenized_text.view([-1, padding_length]).to(self.device)
        text_features = self.clip_model.encode_text(batch_texts)
        return text_features
    
    def encode_image(self, batch_images):
        return self.clip_model.encode_image(batch_images)


class Sym_MLP(nn.Module):
    def __init__(self, config, device) -> None:
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load(config.clip_model, device=device)
        self.sym = Sym(config, device)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(10, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 768)
        )
        self.mlp = self.mlp.to(device).to(self.clip_model.dtype)
        # freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, batch_images, texts):
        texts = list(map(list, zip(*texts)))
        bsz = len(texts)
        num_captions = len(texts[0])

        sym_features = self.sym(texts).to(self.clip_model.dtype)
        text_features = self.mlp(sym_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view([bsz, num_captions, -1])
        batch_images = batch_images.to(self.device)
        with torch.no_grad():
            batch_feat = self.encode_image(batch_images)
            batch_feat = batch_feat / batch_feat.norm(dim=-1, keepdim=True)
            batch_feat = batch_feat.unsqueeze(2)
        logits_per_image = self.clip_model.logit_scale.exp() * torch.bmm(
            text_features, batch_feat
        )
        logits_per_image = logits_per_image.squeeze(2)
        return logits_per_image

    
    def encode_clip_texts(self, texts):
        tokenized_text = [
            clip.tokenize(["a photo of" + t for t in _texts]) for _texts in texts
        ]
        tokenized_text = torch.stack(tokenized_text, dim=0)
        bsz, num_captions, padding_length = tokenized_text.shape
        batch_texts = tokenized_text.view([-1, padding_length]).to(self.device)
        text_features = self.clip_model.encode_text(batch_texts)
        return text_features
    
    def encode_image(self, batch_images):
        return self.clip_model.encode_image(batch_images)


def get_model_(config, device):
    # load the model
    if config.model_name == "MLP_1_100":
        config.hidden_layers = 1
        config.neurons_per_layer = 100
        model = CLIP_Sym(config, device)
    elif config.model_name == "MLP_2_100":
        config.hidden_layers = 2
        config.neurons_per_layer = 100
        model = CLIP_Sym(config, device)
    elif (config.model_name == "MLP_2_300"
        or config.model_name == "CLIP_Sym"):
        config.hidden_layers = 2
        config.neurons_per_layer = 300
        model = CLIP_Sym(config, device)
    elif config.model_name == "MLP_2_500":
        config.hidden_layers = 2
        config.neurons_per_layer = 500
        model = CLIP_Sym(config, device)
    elif config.model_name == "MLP_2_800":
        config.hidden_layers = 2
        config.neurons_per_layer = 800
        model = CLIP_Sym(config, device)
    elif config.model_name == "MLP_3_100":
        config.hidden_layers = 3
        config.neurons_per_layer = 100
        model = CLIP_Sym(config, device)
    elif config.model_name == "Sym_MLP":
        model = Sym_MLP(config, device)
    else:
        raise NotImplementedError(
            "Error: Unrecognized Model Name {:s}.".format(config.model_name)
        )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        eps=1e-6,
    )
    if config.model_path:
        model.load_state_dict(torch.load(config.model_path, map_location=device), strict=False)
    model = model.to(device)
    return model, optimizer
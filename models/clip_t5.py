import torch
from torch import nn
import clip
from transformers import AutoTokenizer, T5EncoderModel

class CLIP_T5_Base(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load(config.clip_model, device=device)
        self.t5 = T5EncoderModel.from_pretrained("t5-small")
        self.t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(512, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, config.emb_dim),
        ) 
        self.mlp = self.mlp.to(device).to(self.clip_model.dtype)
        # freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
        # freeze T5
        for param in self.t5.parameters():
            param.requires_grad = False
    def forward(self, batch_images, texts):
        assert False, "Not implemented"
    
    def encode_t5_texts(self, texts):
        t5_output = []
        for text in texts:
            encoded_input = self.t5_tokenizer(text, return_tensors='pt')
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            last_hidden_state = self.t5(**encoded_input).last_hidden_state
            pooled_output = torch.mean(last_hidden_state, dim=1)
            # map t5 from 512 to 768
            t5_output.append(self.mlp(pooled_output))
        t5_output = torch.concat(t5_output)
        return t5_output
    
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


class T5_Text(CLIP_T5_Base):
    def forward(self, batch_images, texts):
        texts = list(map(list, zip(*texts)))
        bsz = len(texts)
        num_captions = len(texts[0])

        text_features = self.encode_t5_texts(texts)
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


class CLIP_T5_Avg(CLIP_T5_Base):
    def forward(self, batch_images, texts):
        texts = list(map(list, zip(*texts)))
        bsz = len(texts)
        num_captions = len(texts[0])

        t5_features = self.encode_t5_texts(texts)
        clip_features = self.encode_clip_texts(texts)
        # average CLIP and T5 features
        text_features = (t5_features + clip_features) / 2
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
    

class CLIP_T5_Cat(CLIP_T5_Base):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.cat_mlp = torch.nn.Sequential(
            torch.nn.Linear(config.emb_dim*2, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, config.emb_dim),
        )
    def forward(self, batch_images, texts):
        texts = list(map(list, zip(*texts)))
        bsz = len(texts)
        num_captions = len(texts[0])

        t5_features = self.encode_t5_texts(texts)
        clip_features = self.encode_clip_texts(texts)
        # concatenate CLIP and T5 features
        text_features = self.cat_mlp(torch.cat([t5_features, clip_features], dim=-1))

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


def get_model_(config, device):
    # load the model
    if config.model_name == "T5_Text":
        model = T5_Text(config, device)
    elif config.model_name == "CLIP_T5_Avg":
        model = CLIP_T5_Avg(config, device)
    elif config.model_name == "CLIP_T5_Cat":
        model = CLIP_T5_Cat(config, device)
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
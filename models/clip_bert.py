import torch
from torch import nn
import clip
from transformers import BertTokenizer, BertModel


class CLIP_BERT_Base(nn.Module):
    def __init__(self, config, device) -> None:
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load(config.clip_model, device=device)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
        # freeze BERT
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, batch_images, texts):
        assert False, "Not implemented"
    
    def encode_bert_texts(self, texts):
        bert_output = []
        for text in texts:
            encoded_input = self.bert_tokenizer(text, return_tensors='pt')
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            bert_output.append(self.bert(**encoded_input).pooler_output)
        bert_output = torch.concat(bert_output)
        return bert_output
    
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
    



class BERT_Text(CLIP_BERT_Base):
    def forward(self, batch_images, texts):
        texts = list(map(list, zip(*texts)))
        bsz = len(texts)
        num_captions = len(texts[0])

        text_features = self.encode_bert_texts(texts)
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


class CLIP_BERT_Avg(CLIP_BERT_Base):
    def forward(self, batch_images, texts):
        texts = list(map(list, zip(*texts)))
        bsz = len(texts)
        num_captions = len(texts[0])

        bert_features = self.encode_bert_texts(texts)
        clip_features = self.encode_clip_texts(texts)
        # average CLIP and BERT features
        text_features = (bert_features + clip_features) / 2
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


class CLIP_BERT_Cat(CLIP_BERT_Base):
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

        bert_features = self.encode_bert_texts(texts)
        clip_features = self.encode_clip_texts(texts)
        # concatenate CLIP and BERT features
        text_features = self.cat_mlp(torch.cat([bert_features, clip_features], dim=-1))

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
    if config.model_name == "BERT_Text":
        model = BERT_Text(config, device)
    elif config.model_name == "CLIP_BERT_Avg":
        model = CLIP_BERT_Avg(config, device)
    elif config.model_name == "CLIP_BERT_Cat":
        model = CLIP_BERT_Cat(config, device)
    elif config.model_name == "BERT_FT":
        model = BERT_Text(config, device)
        for param in model.bert.parameters():
            param.requires_grad = True
    elif config.model_name == "CLIP_BERT_Cat_train_CLIP":
        model = CLIP_BERT_Cat(config, device)
        for param in model.clip_model.parameters():
            param.requires_grad = True
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
import torch
from torch import nn
import clip

class CLIP_CLASS(nn.Module):
    r'''get emb_com, emb_txt and logits_per_image from CLIP
    '''
    def __init__(self, config, device):
        super().__init__()
        self.clip_model, self.preprocess = clip.load(config.clip_model, device=device)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(config.emb_dim, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 24),
            torch.nn.Softmax(dim=-1)
        ) 
        self.classifier = self.classifier.to(device).to(self.clip_model.dtype)
        self.device = device
    
    def forward(self, batch_images, texts):
        if self.training:
            return self.train_forward(batch_images, texts)
        else:
            return self.test_forward(batch_images, texts)

    def train_forward(self, batch_images, texts):
        texts = list(map(list, zip(*texts)))
        bsz = len(texts)
        num_captions = len(texts[0])

        emb_txt = self.encode_text_samples(texts)
        emb_img = self.encode_image(batch_images.to(self.device))
        class_pred = self.classifier(emb_img)
        emb_txt = emb_txt / emb_txt.norm(dim=-1, keepdim=True)
        emb_txt = emb_txt.view([bsz, num_captions, -1])

        emb_img = emb_img / emb_img.norm(dim=-1, keepdim=True)
        emb_img = emb_img.unsqueeze(2)
        
        logits_per_image = self.clip_model.logit_scale.exp() * torch.bmm(
            emb_txt, emb_img
        )
        logits_per_image = logits_per_image.squeeze(2)
        return logits_per_image, class_pred
    
    def test_forward(self, batch_images, texts):
        texts = list(map(list, zip(*texts)))
        bsz = len(texts)
        num_captions = len(texts[0])

        emb_txt = self.encode_text_samples(texts)
        emb_img = self.encode_image(batch_images.to(self.device))
        emb_txt = emb_txt / emb_txt.norm(dim=-1, keepdim=True)
        emb_txt = emb_txt.view([bsz, num_captions, -1])

        emb_img = emb_img / emb_img.norm(dim=-1, keepdim=True)
        emb_img = emb_img.unsqueeze(2)
        
        logits_per_image = self.clip_model.logit_scale.exp() * torch.bmm(
            emb_txt, emb_img
        )
        logits_per_image = logits_per_image.squeeze(2)
        return logits_per_image


    def encode_text_samples(self, texts):
        tokenized_text = [
            clip.tokenize(["a photo of" + t for t in _texts]) for _texts in texts
        ]
        tokenized_text = torch.stack(tokenized_text, dim=0)
        bsz, num_captions, padding_length = tokenized_text.shape
        batch_texts = tokenized_text.view([-1, padding_length]).to(self.device)
        text_features = self.clip_model.encode_text(batch_texts)
        return text_features
    
    def encode_image(self, images):
        return self.clip_model.encode_image(images)

class CLIP_CLASS_5(nn.Module):
    r'''get emb_com, emb_txt and logits_per_image from CLIP
    '''
    def __init__(self, config, device):
        super().__init__()
        self.clip_model, self.preprocess = clip.load(config.clip_model, device=device)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(config.emb_dim, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 5),
            torch.nn.Softmax(dim=-1)
        ) 
        self.classifier = self.classifier.to(device).to(self.clip_model.dtype)
        self.device = device
    
    def forward(self, batch_images, texts):
        if self.training:
            return self.train_forward(batch_images, texts)
        else:
            return self.test_forward(batch_images, texts)

    def train_forward(self, batch_images, texts):
        texts = list(map(list, zip(*texts)))
        bsz = len(texts)
        num_captions = len(texts[0])

        emb_txt = self.encode_text_samples(texts)
        emb_img = self.encode_image(batch_images.to(self.device))
        class_pred = self.classifier(emb_img)
        emb_txt = emb_txt / emb_txt.norm(dim=-1, keepdim=True)
        emb_txt = emb_txt.view([bsz, num_captions, -1])

        emb_img = emb_img / emb_img.norm(dim=-1, keepdim=True)
        emb_img = emb_img.unsqueeze(2)
        
        logits_per_image = self.clip_model.logit_scale.exp() * torch.bmm(
            emb_txt, emb_img
        )
        logits_per_image = logits_per_image.squeeze(2)
        return logits_per_image, class_pred
    
    def test_forward(self, batch_images, texts):
        texts = list(map(list, zip(*texts)))
        bsz = len(texts)
        num_captions = len(texts[0])

        emb_txt = self.encode_text_samples(texts)
        emb_img = self.encode_image(batch_images.to(self.device))
        emb_txt = emb_txt / emb_txt.norm(dim=-1, keepdim=True)
        emb_txt = emb_txt.view([bsz, num_captions, -1])

        emb_img = emb_img / emb_img.norm(dim=-1, keepdim=True)
        emb_img = emb_img.unsqueeze(2)
        
        logits_per_image = self.clip_model.logit_scale.exp() * torch.bmm(
            emb_txt, emb_img
        )
        logits_per_image = logits_per_image.squeeze(2)
        return logits_per_image


    def encode_text_samples(self, texts):
        tokenized_text = [
            clip.tokenize(["a photo of" + t for t in _texts]) for _texts in texts
        ]
        tokenized_text = torch.stack(tokenized_text, dim=0)
        bsz, num_captions, padding_length = tokenized_text.shape
        batch_texts = tokenized_text.view([-1, padding_length]).to(self.device)
        text_features = self.clip_model.encode_text(batch_texts)
        return text_features
    
    def encode_image(self, images):
        return self.clip_model.encode_image(images)

def get_model_(config, device):
    # load the model
    model = CLIP_CLASS_5(config, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        eps=1e-6,
    )
    if config.model_path:
        model.load_state_dict(torch.load(config.model_path, map_location=device), strict=False)
    return model, optimizer
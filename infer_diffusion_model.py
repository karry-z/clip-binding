#
import argparse
import json
import logging
import os
import pickle
import pprint

import numpy as np
import torch
import tqdm
import pandas as pd
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader

from datasets.clevr_dataset import ObjectDataset, RelDataset
from models import get_model
from utils import set_seed, save_predictions

import torch
import torchvision.transforms as transforms
from models import get_model
from PIL import Image
import torch.nn.functional as F
from datasets.clevr_dataset import RelDataset
from torch.utils.data import DataLoader


DIR_PATH = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)

# set the format of the logger with data, time, level, and the message
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
# import wandb
# wandb.login(key='94cf1e387f47487c4a61b27307ed870991ba63c1')


def run_evaluation(model, optimizer, dataset, config, device):
    """Function to train the model to predict attributes with cross entropy loss.
    Args:
        model (nn.Module): the model to compute the similarity score with the images.
        optimizer (nn.optim): the optimizer with the learnable parameters.
        dataset (torch.utils.data.Dataset): the train dataset
        config (argparse.ArgumentParser): the config
        device (str): torch device
    Returns:
        tuple: the trained model (or the best model) and the optimizer
    """
    model.eval()
    dataloader = DataLoader(
        dataset, batch_size=config.eval_batch_size, shuffle=False
    )
    model.to(device)
    
    progress_bar = tqdm.tqdm(
        total=len(dataloader), disable=None
    )
    imgid = 0
    with torch.no_grad():
        for bid, batch in enumerate(dataloader):
            _, texts = batch
            image = model(texts)
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            # save the images
            for i, img in enumerate(images):
                img = Image.fromarray(img)
                img.save(
                    os.path.join(
                        config.save_dir, f"CLEVR_train_rel_{imgid:06}.png"
                    )
                )
                imgid += 1
            progress_bar.update()

    progress_bar.close()



    return model, optimizer, {}




from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)
BICUBIC = InterpolationMode.BICUBIC
n_px=256
preprocess = Compose(
    [
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)

def getitem(self, idx):
    img_path = self.img_dir + self.ims_labels[idx][0]
    image = Image.open(img_path)  # Image from PIL module
    image = preprocess(image)

    texts = [self.ims_labels[idx][1]]

    return image, texts

RelDataset.__getitem__ = getitem

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", help="name of the experiment", type=str, default="DM"
    )
    parser.add_argument(
        "--text_model_name", help="name of the experiment", type=str, default="clip"
    )
    parser.add_argument(
        "--model_path", help="name of the experiment", default=None
    )
    parser.add_argument(
        "--text_model_path", help="name of the experiment", default=None
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
    parser.add_argument("--epochs", help="number of epochs", default=2, type=int)
    parser.add_argument(
        "--train_batch_size", help="train batch size", default=2, type=int
    )
    parser.add_argument(
        "--eval_batch_size", help="eval batch size", default=2, type=int
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        help="number of gradient accumulation steps",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--evaluate_only",
        help="directly evaluate on the" "dataset without any training",
        action="store_true",
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
    parser.add_argument("--save_dir", help="save path", default='/user/work/pu22650/clip-binding-out/DM_Infer/gen', type=str)
    # parser.add_argument(
    #     "--save_every_n",
    #     default=1,
    #     type=int,
    #     help="saves the model every n epochs; "
    #     "this is useful for validation/grid search",
    # )
    parser.add_argument(
        "--save_model",
        help="indicate if you want to save the model state dict()",
        action="store_true",
    )
    parser.add_argument("--seed", help="seed value", default=0, type=int)
    config = parser.parse_args()

    # set the seed value
    set_seed(config.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(config)
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="clip-binding",
    #     entity="karry-z",
    #     name=config.model_name,
    #     # track hyperparameters and run metadata
    #     # config=dict
    # )
    
    if not config.save_dir:
        config.save_dir = os.path.join(
            DIR_PATH,
            f"data/model/{config.dataset}/{config.model_name}_seed_{config.seed}",
        )

    # get the dataset
    logger.info("loading the dataset...")
    dataset = {split:RelDataset(split=split) for split in ["train", "val", "gen"]}

    # get the model
    logger.info("loading the model...")
    model, optimizer = get_model(dataset["train"], config, device)

    model.to(device)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    if not config.evaluate_only:
        logger.info("training the model...")
        # train and test the model
        model, optimizer, results = run_evaluation(
            model,
            optimizer,
            dataset['train'],
            config,
            device,
        )


    logger.info("done!")

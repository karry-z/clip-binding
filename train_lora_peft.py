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

from datasets.clevr_dataset import ObjectDataset 
from models import get_model
from utils import set_seed, save_predictions
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)

# set the format of the logger with data, time, level, and the message
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

from datasets.read_datasets import DATASET_PATHS
from PIL import Image
import random
REL_PARAMS = {
    "relations": ["front", "behind", "left", "right"],
    "rel-opposites": {
        "front": "behind",
        "behind": "front",
        "left": "right",
        "right": "left",
    },
    "nouns": ["cube", "sphere", "cylinder"],
}

class RelDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.img_dir = DATASET_PATHS["rel"][f"{split}_image_path"]

        # load the labels from the json file
        label_file = DATASET_PATHS["rel"][f"{split}_label_path"]
        with open(label_file, "r") as l:
            self.labels = json.load(l)

        self.ims_labels = [
            (im, p) for im in self.labels for p in self.labels[im]["pos"]
        ]

        self.rel_opposites = REL_PARAMS["rel-opposites"]
        self.nouns = REL_PARAMS["nouns"]
        self.objects = REL_PARAMS["nouns"]
        self.relations = REL_PARAMS["relations"]

        self.concepts = self.objects + self.relations
        self.concept_to_idx = dict(
            [(concept, i) for i, concept in enumerate(self.concepts)]
        )

    def __len__(self):
        # the length of the dataset is the total number of positive labels
        return len(self.ims_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir + self.ims_labels[idx][0]
        image = Image.open(img_path)  # Image from PIL module


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
        random.shuffle(indices)
        texts = [texts[i] for i in indices]
        label = indices.index(0)

        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
        return inputs, label

def choose_dataset(config):
    """
    Choose the dataset: single-color, two-object-color, and rel.
    """
    dataset = {}

    for split in ["train", "val", "gen"]:
        if config.dataset == "single-object" or config.dataset == "two-object":
            _dataset = ObjectDataset(
                split=split,
                dataset=config.dataset,
            )
        elif config.dataset == "rel":
            _dataset = RelDataset(split=split)
        else:
            logger.error(f"{config.dataset} dataset is not found!")
            NotImplementedError(f"{config.dataset} dataset is not found!")
            exit(0)

        dataset[split] = _dataset

    return dataset


def run_evaluation(model, dataset, config, return_preds=False):
    """
    Function to run evaluation on the valiadation and generalization splits.
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=config.eval_batch_size, shuffle=False)
    labels_list = []
    logits_list = []
    progress_bar = tqdm.tqdm(total=len(dataloader), disable=None)
    all_texts = []
    with torch.no_grad():
        for bid, batch in enumerate(dataloader):
            inputs, labels = batch
            inputs = {name: tensor.squeeze(1).to(device) for name, tensor in inputs.items()}
            inputs['input_ids'] = inputs['input_ids'].reshape(-1, 5)
            inputs['attention_mask'] = inputs['attention_mask'].reshape(-1, 5)
            logits_per_image = model(**inputs).logits_per_image
            texts = list(map(list, zip(*texts)))
            all_texts += texts
            batch_target = labels.to(device)
            labels_list.append(batch_target.cpu())
            logits_list.append(logits_per_image.cpu())

            progress_bar.update()
        progress_bar.close()

        labels = torch.cat(labels_list).numpy()
        res = torch.cat(logits_list).numpy()

        preds = np.argmax(res, axis=1)
        acc = np.sum(preds == labels) / len(labels) * 100.0

    preds_idx_list = list(preds)
    preds_text_list = [
        all_texts[i][pred_idx] for i, pred_idx in enumerate(preds_idx_list)
    ]
    labels_text_list = [all_texts[i][label_idx] for i, label_idx in enumerate(labels)]

    if return_preds:
        return acc, preds_text_list, labels_text_list

    return acc


def train_model(model, optimizer, dataset_dict, config, device):
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

    train_dataloader = DataLoader(
        dataset_dict["train"], batch_size=config.train_batch_size, shuffle=True
    )
    model.concept_to_idx = dataset["train"].concept_to_idx

    model.train()
    model.to(device)
    loss_fn = CrossEntropyLoss()
    i = 0
    train_losses = []
    results = []

    for i in range(config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1), disable=None
        )
        trn_results = []
        trn_labels = []
        epoch_train_losses = []
        model.train()
        for bid, batch in enumerate(train_dataloader):
            inputs, labels = batch
            inputs = {name: tensor.squeeze(1).to(device) for name, tensor in inputs.items()}
            inputs['input_ids'] = inputs['input_ids'].reshape(-1, 5)
            inputs['attention_mask'] = inputs['attention_mask'].reshape(-1, 5)
            logits_per_image = model(**inputs).logits_per_image
            batch_target = labels.to(device)
            loss = loss_fn(logits_per_image, batch_target)

            # normalize loss to account for batch accumulation
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (
                bid + 1 == len(train_dataloader)
            ):
                optimizer.step()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})

            trn_labels.append(batch_target.detach().cpu())
            trn_results.append(logits_per_image.detach().cpu())

            progress_bar.update()

        progress_bar.close()

        lbs = torch.cat(trn_labels).detach().cpu().numpy()
        res = torch.cat(trn_results).detach().cpu().numpy()

        choices = np.argmax(res, axis=1)
        train_acc = np.sum(choices == lbs) / len(lbs) * 100.0

        accuracy = {"train": train_acc}
        for split in ["val", "gen"]:
            acc, preds, _labels = run_evaluation(
                model, dataset_dict[split], config, return_preds=True
            )
            accuracy[split] = acc
            save_predictions(preds, _labels, i + 1, split, config.save_dir)

        logger.info(f"Training Accuracy is: {train_acc:.2f}")

        progress_bar.write(
            f"epoch {i +1}, "
            f"train loss {np.mean(epoch_train_losses):.2f}, "
            f"train accuracy {accuracy['train']:.2f}, "
            f"val acc {accuracy['val']:.2f}, "
            f"gen acc {accuracy['gen']:.2f}"
        )

        train_losses.append(np.mean(epoch_train_losses))
        results.append(accuracy)

    return model, optimizer, results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", help="name of the experiment", type=str, default="csp"
    )
    parser.add_argument(
        "--dataset", help="name of the dataset", type=str, default="single-object"
    )
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-06)
    parser.add_argument(
        "--weight_decay", help="weight decay", type=float, default=1e-05
    )
    parser.add_argument(
        "--clip_model", help="clip model type", type=str, default="ViT-L/14"
    )
    parser.add_argument("--epochs", help="number of epochs", default=20, type=int)
    parser.add_argument(
        "--train_batch_size", help="train batch size", default=32, type=int
    )
    parser.add_argument(
        "--eval_batch_size", help="eval batch size", default=64, type=int
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
    parser.add_argument("--save_dir", help="save path", type=str)
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

    if not config.save_dir:
        config.save_dir = os.path.join(
            DIR_PATH,
            f"data/model/{config.dataset}/{config.model_name}_seed_{config.seed}",
        )

    # get the dataset
    logger.info("loading the dataset...")
    dataset = choose_dataset(config)

    # get the model
    logger.info("loading the model...")
    # model, optimizer = get_model(dataset["train"], config, device)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        eps=1e-6,
    )
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)

    if not config.evaluate_only:
        logger.info("training the model...")
        # train and test the model
        model, optimizer, results = train_model(
            model,
            optimizer,
            dataset,
            config,
            device,
        )

        with open(os.path.join(config.save_dir, "config.pkl"), "wb") as fp:
            pickle.dump(config, fp)

        with open(os.path.join(config.save_dir, "results.json"), "w+") as fp:
            json.dump(results, fp)

        if config.save_model:
            torch.save(model.state_dict(), os.path.join(config.save_dir, "final_model.pt"))
    else:
        logger.info("skipping training and directly evaluating the model...")
        for split in ["train", "val", "gen"]:
            acc = run_evaluation(model, dataset[split], config)
            print(f"{split}: {acc:.2f}")

    logger.info("done!")

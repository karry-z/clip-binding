import os
import torch

from .clip import get_clip_model
from .csp import get_csp
from .cdsm import get_cdsm


DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def get_model(train_dataset, config, device):
    if config.model_name == "clip":
        return get_clip_model(config, device)
    elif config.model_name == "csp":
        return get_csp(train_dataset, config, device)
    elif (
        config.model_name == "add"
        or config.model_name == "mult"
        or config.model_name == "conv"
        or config.model_name == "rf"
        or config.model_name == "tl"
        or config.model_name == "cat"
    ):
        return get_cdsm(train_dataset, config, device)
    elif config.model_name == "comp_clip":
        from .comp_clip import get_model_
        return get_model_(config, device)
    elif (config.model_name == "T5_Text"
        or config.model_name == "CLIP_T5_Avg"
        or config.model_name == "CLIP_T5_Cat"):
        from .clip_t5 import get_model_
        return get_model_(config, device)
    elif (config.model_name == "BERT_Text"
          or config.model_name == "CLIP_BERT_Avg"
          or config.model_name == "CLIP_BERT_Cat"
          or config.model_name == "BERT_FT"
          or config.model_name == "CLIP_BERT_Cat_train_CLIP"):
        from .clip_bert import get_model_
        return get_model_(config, device)
    elif (config.model_name == "CLIP_Sym"
          or config.model_name == "MLP_1_100"
          or config.model_name == "MLP_2_100"
          or config.model_name == "MLP_2_300"
          or config.model_name == "MLP_2_500"
          or config.model_name == "MLP_2_800"
          or config.model_name == "Sym_MLP"
          or config.model_name == "MLP_3_100"):

        from .sym import get_model_
        return get_model_(config, device)
    elif config.model_name == "CLIP_Class":
        from .clip_class import get_model_
        return get_model_(config, device)
    elif config.model_name == "DM":
        from .dm import get_model_
        return get_model_(config, device)
    else:
        raise NotImplementedError(
            "Error: Unrecognized Model Name {:s}.".format(config.model_name)
        )

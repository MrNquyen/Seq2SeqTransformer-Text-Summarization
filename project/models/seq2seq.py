import torch
from torch import nn
from project.modules.decoder import Decoder
from project.modules.encoder import Encoder
from utils.registry import registry
from utils.utils import count_nan
from utils.module_utils import _batch_gather
from torch.nn import functional as F
from icecream import ic
import math


class TransformerSummarizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_config = registry.get_config("model_attributes")
        self.device = registry.get_args("device")
        self.writer = registry.get_writer("common")
        self.hidden_size = self.model_config["hidden_size"]
        self.build()

    #-- BUILD
    def build(self):
        self.writer.LOG_INFO("=== Build model params ===")
        self.build_model_params()
        
        self.writer.LOG_INFO("=== Build writer ===")
        self.build_writer()

        self.writer.LOG_INFO("=== Build model layers ===")
        self.build_layers()
        self.build_model_init()

        self.writer.LOG_INFO("=== Build model outputs ===")
        self.build_ouput()

        self.writer.LOG_INFO("=== Build adjust learning rate ===")
        self.adjust_lr()
    

    def build_writer(self):
        self.writer = registry.get_writer("common")


    def build_layers(self):
        self.decoder = Decoder()
        self.encoder = Encoder()

    
    def adjust_lr(self):
        #~ Word Embedding
        self.add_finetune_modules(self.word_embedding)


    #-- ADJUST LEARNING RATE
    def add_finetune_modules(self, module: nn.Module):
        self.finetune_modules.append({
            'module': module,
            'lr_scale': self.model_config["adjust_optimizer"]["lr_scale"],
        })

    def get_optimizer_parameters(self, config_optimizer):
        """
            -----
            Function:
                - Modify learning rate
                - Fine-tuning layer has lower learning rate than others
        """
        optimizer_param_groups = []
        base_lr = config_optimizer["params"]["lr"]
        scale_lr = config_optimizer["lr_scale"]
        base_lr = float(base_lr)
        
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append({
                "params": list(m['module'].parameters()),
                "lr": base_lr * scale_lr
            })
            finetune_params_set.update(list(m['module'].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})
        
        # check_overlap(finetune_params_set, remaining_params)
        return optimizer_param_groups
        
    def init_random(self, shape):
        return torch.normal(
            mean=0.0, 
            std=0.1, 
            size=shape
        ).to(self.device)
        
    #-- FORWARD
    def forward(
            self,
            batch
        ):
        gt_caption = batch["captions"]
        ocr_description = batch["ocr_description"]

        gt_caption_embed = self.encoder(gt_caption)
        ocr_description_embed = self.encoder(ocr_description)

        if self.training:
            pass
        else:
            pass
        
        
    def forward_output(self, results):
        """
        Calculate scores for ocr tokens and common word at each timestep

            Parameters:
            ----------
            results: dict
                - The result output of decoder step

            Return:
            ----------
        """
        pass



import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, PreTrainedModel
from typing import List
from utils.registry import registry

#========= TEXT ENCODER ===========
class PretrainedModel(nn.Module):
    def __init__(self, _type="encoder"):
        super().__init__()
        #-- Load config and args
        self.model_config = registry.get_config("model_attributes")
        self.encoder_config = self.model_config[_type]
        self.device = registry.get_args("device")
        self.max_length = self.encoder_config["max_length"]
        self.load_pretrained()

    def load_pretrained(self):
        self.model_name = self.text_embedding_config["pretrained"]
        config = AutoConfig.from_pretrained(self.model_name)

        #-- Load pretrained
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            config=config
        ).to(self.device)
        self.model.gradient_checkpointing_enable()
        self.init_weights()


    def tokenize(self, texts: List[str]):
        """
            Args:
                - texts: (str): Batch of texts

            Return:
                - dict: Text input has 'input_ids', 'token_type_ids', 'attention_mask'

            Example: 
                - Return a dict = {
                    'input_ids': ..., 
                    'token_type_ids': ..., 
                    'attention_mask': ...,
                }
        """
        input_ids = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"        # return PyTorch tensors directly
        )
        return input_ids # 'input_ids', 'token_type_ids', 'attention_mask'
    
    def text_embedding(self, input_ids):
        """
            Args:
                - input_ids: (str): Input is a output from tokenizer

            Return:
                - Tensor: Tensor of text embeb features

            Example: 
                - input_ids = {
                    'input_ids': ..., 
                    'token_type_ids': ..., 
                    'attention_mask': ...,
                }
        """
        with torch.no_grad():
            features = self.model(**input_ids)
            return features.last_hidden_state


    #-- Common function
    def get_vocab_size(self):
        return len(self.tokenizer.get_vocab())

    #-- Get token
    def get_pad_token(self):
        return self.tokenizer.pad_token
    
    def get_unk_token(self):
        return self.tokenizer.unk_token
    
    def get_cls_token(self):
        return self.tokenizer.cls_token
    
    def get_eos_token(self):
        return self.tokenizer.eos_token
    
    #-- Get token id
    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id
    
    def get_unk_token_id(self):
        return self.tokenizer.unk_token_id
    
    def get_cls_token_id(self):
        return self.tokenizer.cls_token_id
    
    def get_eos_token_id(self):
        return self.tokenizer.eos_token_id

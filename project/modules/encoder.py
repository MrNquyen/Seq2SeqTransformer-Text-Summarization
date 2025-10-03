import torch
import numpy as np

from torch import nn

from project.modules.base import PreTrainedModel
from utils.registry import registry
from utils.vocab import CustomVocab

class Encoder(PreTrainedModel):
    def __init__(self):
        super().__init__(_type="encoder")

    def forward(self, texts):
        """
            Text Embedding the batch of texts provided
            Args:
                - texts: Batch of input texts

            Return:
                - torch.Tensor: text_embed of the given texts - BS, max_length, hidden_size
        """
        inputs = self.tokenize(texts)
        text_embed = self.text_embedding(inputs)
        return text_embed
    


class EncoderDescription(PreTrainedModel):
    def __init__(self):
        super().__init__(_type="encoder")

    def forward(self, texts):
        """
            Text Embedding the batch of texts provided
            Args:
                - texts: Batch of input texts

            Return:
                - torch.Tensor: text_embed of the given texts - BS, max_length, hidden_size
        """
        inputs = self.tokenize(texts)
        text_embed = self.text_embedding(inputs)
        return text_embed
    


class EncoderSummary(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.model_config = registry.get_config("model_attributes")
        self.encoder_config = self.model_config["encoder"]
        self.device = registry.get_args("device")
        self.writer = registry.get_writer("common")
        self.tokenizer = tokenizer
        self.max_dec_length = self.encoder_config["max_dec_length"]

        self.load_vocab()

    #-- Build
    def load_vocab(self):
        vocab_file = self.encoder_config["vocab_file"]
        self.vocab = CustomVocab(
            tokenizer=self.tokenizer,
            vocab_file=vocab_file
        )

    def pad_to_dec_length(self, batch_tokens):
        """
            Padding sequences tokens to max_dec_length 
            Args:
                batch_tokens (List[str]): Batch of list tokens to get inds
                    - Shape: BS, different_length
            
            Return:
                batch_inds (torch.Tensor): Tensor of inds (same max_dec_length)
        """
        start_token = self.vocab.get_start_token()
        end_token = self.vocab.get_end_token()
        pad_token = self.vocab.get_pad_token()
        batch_tokens = [
            [start_token] + tokens[:self.max_dec_length - 2] + [end_token] + [pad_token] * max(0, (self.max_dec_length - 2 - len(tokens)))
            for tokens in batch_tokens
        ]
        return batch_tokens


    def get_word_inds(self, batch_tokens):
        """
            Args:
                batch_tokens (List[str]): Batch of list tokens to get inds
                    - Shape: BS, different_length, 
            
            Return:
                batch_inds (torch.Tensor): Tensor of inds
        """
        batch_tokens = self.pad_to_dec_length(batch_tokens)
        batch_inds = [
            [self.vocab.get_word_idx(token) for token in tokens]
            for tokens in batch_tokens
        ]
        batch_inds = torch.tensor(batch_inds).to(self.device)
        return batch_inds
    
    def batch_decode(self, prev_inds):
        predictions = [
            " ".join([self.vocab.get_idx_word(idx.item()) for idx in sen_inds])
            for sen_inds in prev_inds
        ]
        return predictions
    

    def get_vocab_size(self):
        return self.vocab.get_size()



    
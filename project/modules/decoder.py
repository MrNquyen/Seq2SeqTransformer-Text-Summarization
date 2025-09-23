import torch
import math
from torch import nn
from torch.nn import functional as F
from utils.registry import registry 
from icecream import ic

from project.modules.base import PreTrainedModel


# -- Previous Embedding
class PrevEmbedding(nn.Module):
    def __init__(self, hidden_size, decoder_config):
        super().__init__()
        self.DEC_LENGTH = decoder_config["max_length"] # Max caption output length
        self.hidden_size = hidden_size

        self.positional_embedding = nn.Embedding(
            num_embeddings=self.DEC_LENGTH, 
            embedding_dim=hidden_size
        )

        self.ans_emb_norm = nn.LayerNorm(normalized_shape=self.hidden_size)
        self.emb_layer_norm = nn.LayerNorm(normalized_shape=self.hidden_size)
        self.emb_dropout = nn.Dropout(decoder_config["dropout"])


    def init_pe_weights(self):
        """
            Init weight for self.positional_embedding
        """
        # Create positional encoding sin cos from Attention Is All You Need! paper
        pe = torch.zeros(self.DEC_LENGTH, self.hidden_size)
        position = torch.arange(0, self.DEC_LENGTH, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2).float() *
            (-math.log(10000.0) / self.hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Assign custom weights
        self.positional_embedding.weight.data = pe.clone()

    
    def forward(
            self,
            ans_emb,
            prev_inds
        ):
        """
            Args:
                -     
        """
        # -- Params
        batch_size = prev_inds.shape[0]
        current_seq_length = prev_inds.shape[1]
        vocab_size = ans_emb.shape[0]

        # -- Get prev vector embed
        ans_emb = self.ans_emb_norm(ans_emb)
        ocr_embedding = self.ocr_embedding_norm(ocr_embedding)
        assert ans_emb.size(-1) == ocr_embedding.size(-1)

        ans_emb = ans_emb.unsqueeze(0).expand(batch_size, -1, -1)
        # ic(ans_emb.shape, ocr_embedding.shape)
        look_up_table_embedding = torch.concat(
            [ans_emb, ocr_embedding],
            dim=1
        )

        # ic(look_up_table_embedding.device, prev_inds.device)
        last_word_embeddings = _batch_gather(
            x=look_up_table_embedding, 
            inds=prev_inds
        )

        # -- Position 
        position_ids = torch.arange(
            current_seq_length,
            dtype=torch.long,
            device=ocr_embedding.device
        )
        # ic(position_ids)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.positional_embedding(position_ids)

        # -- Type embedding: 0: common tokens (False) - 1: ocr_tokens (True)
        type_ids = position_ids.ge(vocab_size).long()
        token_type_embedddings = self.token_type_embedding(type_ids)

        # -- Position and token type
        pos_type_embeddings = position_embeddings + token_type_embedddings 
        # ic(pos_type_embeddings.shape)
        pos_type_embeddings = self.emb_layer_norm(pos_type_embeddings)
        pos_type_embeddings = self.emb_dropout(pos_type_embeddings)

        # -- LastWord, Position, token type
        prev_emb = last_word_embeddings + pos_type_embeddings
        return prev_emb # BS, num_prev_words, hidden_size



class Decoder(PreTrainedModel):
    def __init__(self):
        super().__init__(_type="decoder")
        self.encoder = self.model.encoder

    def forward(
            self,
            prev_inds: torch.Tensor,
            texts_emb: torch.Tensor
        ):
        pass

    
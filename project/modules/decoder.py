import torch
import math
from torch import nn
from torch.nn import functional as F
from utils.registry import registry 
from icecream import ic

from project.modules.base import PreTrainedModel
from utils.module_utils import _get_causal_mask


# -- Previous Embedding
class PrevEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.model_config = registry.get_config("model_attributes")
        self.decoder_config = self.model_config["decoder"]
        self.hidden_size = self.model_config["hidden_size"]
        
        self.DEC_LENGTH = self.decoder_config["max_length"] # Max summarize output length

        self.positional_embedding = nn.Embedding(
            num_embeddings=self.DEC_LENGTH, 
            embedding_dim=hidden_size
        )
    
        self.emb_layer_norm = nn.LayerNorm(normalized_shape=self.hidden_size)
        self.emb_dropout = nn.Dropout(self.decoder_config["dropout"])


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
            ans_fixed_embed,
            prev_inds
        ):
        """
            :params ans_fixed_embed    :  common_vocab_len, hidden_size:   All embedding of common vocab
            :params prev_inds                :  BS, list_prev_idx_in_vocab   :   All idx in vocab of prev word in 
            ----
            Note:
                - Idx of ocr token start from: vocab_len + 1
                (Because append ocr_vocab to common_vocab)
                - When training, input all the gt caption and mask, so the model cannot see the future
            ----
            Function:
                - Lookup table embed position, and get prev embeded vector
        """
        # -- Params
        batch_size = prev_inds.shape[0]
        current_seq_length = prev_inds.shape[1]
        vocab_size = common_voc_embedding.shape[0]
        ocr_size = ocr_embedding.shape[1]

        # -- Get prev vector embed
        common_voc_embedding = self.common_voc_embedding_norm(common_voc_embedding)
        ocr_embedding = self.ocr_embedding_norm(ocr_embedding)
        assert common_voc_embedding.size(-1) == ocr_embedding.size(-1)

        common_voc_embedding = common_voc_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        # ic(common_voc_embedding.shape, ocr_embedding.shape)
        look_up_table_embedding = torch.concat(
            [common_voc_embedding, ocr_embedding],
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

    def prev_embedding(self, input_ids):
        return self.text_embedding(input_ids)


    def forward(
            self,
            prev_inds: torch.Tensor,
            texts_emb: torch.Tensor,
            attention_mask: torch.Tensor
        ):
        prev_embed = self.prev_embedding(prev_inds)
        
        #-- Multihead broadcasting
        end_when_reach_maxlen = attention_mask.size(1) # 
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, end_when_reach_maxlen, 1
        )

        #-- Casual mask
        extended_attention_mask[:, :, :, :] = \
            _get_causal_mask(prev_embed.size(1), self.device)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config["num_layers"]
        
        encoder_outputs = self.encoder(
            prev_embed,
            extended_attention_mask,
            head_mask=head_mask
        )
        return encoder_outputs

    
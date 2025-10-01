import torch
import math
from torch import nn
from torch.nn import functional as F
from utils.registry import registry 
from icecream import ic

from project.modules.base import PreTrainedModel
from utils.module_utils import _batch_gather, _get_causal_mask


# -- Previous Embedding
class PrevEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.device = registry.get_args("device")
        self.model_config = registry.get_config("model_attributes")
        self.decoder_config = self.model_config["decoder"]
        self.hidden_size = self.model_config["hidden_size"]
        
        self.DEC_LENGTH = self.decoder_config["max_length"] # Max summarize output length

        self.positional_embedding = nn.Embedding(
            num_embeddings=self.DEC_LENGTH, 
            embedding_dim=hidden_size
        )
        # self.init_pe_weights()
    
        self.emb_layer_norm = nn.LayerNorm(normalized_shape=self.hidden_size)
        self.fixed_ans_emb_norm = nn.LayerNorm(normalized_shape=self.hidden_size)
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
            fixed_ans_emb,
            prev_inds
        ):
        """
            :params fixed_ans_emb    :  common_vocab_len, hidden_size:   All embedding of common vocab
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

        # -- Get prev vector embed
        fixed_ans_emb = self.fixed_ans_emb_norm(fixed_ans_emb)
        fixed_ans_emb = fixed_ans_emb.unsqueeze(0).expand(batch_size, -1, -1)

        last_word_embed = _batch_gather(
            x=fixed_ans_emb, 
            inds=prev_inds
        )

        # -- Position 
        position_ids = torch.arange(
            current_seq_length,
            dtype=torch.long,
            device=self.device
        )
    
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embed = self.positional_embedding(position_ids)
        position_embed = self.emb_layer_norm(position_embed)
        position_embed = self.emb_dropout(position_embed)

        # -- LastWord, Position
        prev_emb = last_word_embed + position_embed
        return prev_emb # BS, num_prev_words, hidden_size



class Decoder(PreTrainedModel):
    def __init__(self):
        super().__init__(_type="decoder")
        self.encoder = self.model.encoder
        self.prev_embedding = PrevEmbedding(hidden_size=self.hidden_size)


    def forward(
            self,
            prev_inds: torch.Tensor,
            input_embed: torch.Tensor,
            fixed_ans_emb: torch.Tensor,
            input_attention_mask: torch.Tensor
        ):
        #-- Input features
        prev_embed = self.prev_embedding(
            fixed_ans_emb=fixed_ans_emb,
            prev_inds=prev_inds
        )

        # Also check dtype of prev_inds:
        assert prev_inds.dtype == torch.long, "prev_inds must be LongTensor for embedding lookup"


        encoder_inputs = torch.cat(
            [input_embed, prev_embed],
            dim=1
        )

        #-- Mask Attention
        dec_mask = torch.zeros(
            prev_embed.size(0),
            prev_embed.size(1),
            dtype=torch.float32,
            device=self.device
        )

        attention_mask = torch.cat(
            [input_attention_mask, dec_mask],
            dim=1
        )
        
        #-- Offsets of each modality in the joint embedding space
        encoder_input_begin = 0
        encoder_input_end = encoder_input_begin + input_embed.size(1)
        dec_input_begin = encoder_input_end
        dec_input_end = dec_input_begin + prev_embed.size(1)


        #-- Multihead broadcasting
        end_when_reach_maxlen = attention_mask.size(1) # 
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, end_when_reach_maxlen, 1
        )

        #-- Casual mask
        extended_attention_mask[:, :, dec_input_begin:, dec_input_begin:] = \
            _get_causal_mask(dec_input_end - dec_input_begin, self.device)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.type_config["nhead"]

        #-- Transformer Encoder
        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        mmt_dec_output = mmt_seq_output[:, dec_input_begin:]

        results = {
            "mmt_dec_output": mmt_dec_output
        }

        return results

    
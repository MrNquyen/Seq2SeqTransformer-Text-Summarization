from project.modules.base import PretrainedModel

class Encoder(PretrainedModel):
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
        input_ids = self.tokenize(texts)
        text_embed = self.text_embedding(input_ids)
        return text_embed

    
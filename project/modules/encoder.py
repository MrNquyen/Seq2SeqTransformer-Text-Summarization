from project.modules.base import PreTrainedModel

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
    


class EncoderSummary(PreTrainedModel):
    def __init__(self):
        super().__init__(_type="decoder")

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

    
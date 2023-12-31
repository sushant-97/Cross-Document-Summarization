from torch import nn

from transformers.modeling_bart import BartEncoder, PretrainedBartModel, PretrainedBartModel
from transformers import BartConfig

class Encoder(PretrainedBartModel):
  def __init__(self, config: BartConfig):
    super().__init__(config)

    padding_idx, vocab_size = config.pad_token_id, config.vocab_size
    self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

    self.encoder = BartEncoder(config, self.shared)
  
  def forward(
      self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False
  ):

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    encoder_outputs = self.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    return encoder_outputs

enc = Encoder.from_pretrained("facebook/bart-base")

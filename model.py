import torch
import torch.nn as nn
from transformers import BertModel
import numpy as np

class DRAQModel(nn.Module):
    def __init__(self, config, device):
        super(DRAQModel, self).__init__()
        self.device = device
        self.bert = BertModel.from_pretrained(config['bertModel'])
        input_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        self.regression = torch.nn.Sequential(
            torch.nn.Linear(input_size, config['num_labels']),
            torch.nn.Sigmoid()
        )
        self.regression.apply(self._init_weights)


    def forward(self,input_ids: torch.Tensor, attention_mask: [torch.Tensor],token_type_ids: [torch.Tensor]) :
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        # token_type_ids = token_type_ids.to(self.device)
        model_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids
            )

        dropout = self.dropout((model_outputs.last_hidden_state)[:, 0])
        qpp_logits =  self.regression(dropout)
        return qpp_logits



    def _init_weights(self,module: nn.Module) -> None:  # type: ignore
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


class MeanPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states.mean(1)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MatchingHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.pooler = Pooler(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc(self.pooler(x))
        return x


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x

class MAEHead(nn.Module):
    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.decoder = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.decoder(x)       
        return x
    
import sys
import os
import traceback
import logging
import json
import re

import torch


def get_warmup_and_linear_decay(max_steps, warmup_steps):
    return lambda step: min(step / warmup_steps, 1. - (step - warmup_steps) / (max_steps - warmup_steps))


def init_optimizer(model, config, train_steps, scale_params_lr=None):
    if scale_params_lr is not None:
        for model, lr_scale in scale_params_lr:
            for param in model.parameters():
                param.optimize_attr['learning_rate'] *= lr_scale
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_deacy': 0.}
    ]

    warmup_steps = int(config.warmup_proportion * train_steps)
    optimizer = torch.optim.AdamW(
        lr=config.learning_rate,
        params=optimizer_grouped_parameters)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=get_warmup_and_linear_decay(train_steps, warmup_steps)
    )
    return lr_scheduler, optimizer


if __name__ == "__main__":
    """run some simple test case"""
    import types
    import torch.nn as nn

    # Define a simple feed-forward neural network
    class SimpleNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x


    model = SimpleNN(input_size=10, hidden_size=5, output_size=2)
    config = types.SimpleNamespace(
        learning_rate=1e-3,
        warmup_proportion=0.1,
        weight_decay=0.2,
        grad_clip=1.0)

    # Initialize the optimizer and learning rate scheduler
    lr_scheduler, optimizer = init_optimizer(model, config, train_steps=10000)

    # Print the optimizer and learning rate scheduler
    print(optimizer)
    print(lr_scheduler)
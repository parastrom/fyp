#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/electron1c/RAT-SQL-pytorch/blob/ea5aef728e539ba7155520990f32ce4cc22092f2/text2sql/utils/nn_utils.py
# Paddle changed to Pytorch

import numpy as np
import torch
from torch import nn


def build_linear(n_in, n_out, name=None, init=None):
    return nn.Linear(n_in, n_out)


def build_layer_norm(n_in, name=None):
    layer_norm = nn.LayerNorm(normalized_shape=n_in)

    if name is not None:
        layer_norm.weight.data.fill_(1.0)
        layer_norm.bias.data.fill_(0.0)

    return layer_norm


def lstm_init(num_layers, hidden_size, *batch_sizes):
    init_size = batch_sizes + (hidden_size,)
    if num_layers is not None:
        init_size = (num_layers,) + init_size
        init = torch.zeros(init_size, device='cuda:0')
    return (init,init)


def batch_gather_2d(var, indices):
    if len(indices.shape) != 2:
        raise ValueError('shape of indices error. it should be a 2-D layers. '
                         'but got shape = %s' % (str(indices.shape),))

    batch_size = indices.shape[0]
    batch_indices_1d = torch.unsqueeze(torch.arange(0, batch_size, dtype=indices.dtype,
                                                    device='cuda:0'), 1)

    seq_len = indices.shape[1]
    batch_indices = batch_indices_1d.repeat([1, seq_len])

    coord_2d = torch.cat(
        [torch.unsqueeze(batch_indices, 2), torch.unsqueeze(indices, 2)],
        dim=2).detach()

    coord_1d = torch.reshape(coord_2d, shape=[-1, 2])
    output_1d = var[coord_1d[..., 0], coord_1d[..., 1]]
    output_2d = torch.reshape(output_1d, [batch_size, seq_len, var.shape[-1]])
    return output_2d


def sequence_mask(seq_hidden, mask, mode='zero'):
    dtype = seq_hidden.dtype

    while len(mask.shape) < len(seq_hidden.shape):
        mask = mask.unsqueeze(-1)

    mask = mask.to(dtype)
    masked = torch.mul(seq_hidden, mask)

    if mode == 'zero':
        return masked

    if mode == '-inf':
        scale_size = -1e5
    elif mode == '+inf':
        scale_size = 1e5
    else:
        raise ValueError(f'mask mode setting error. expect zero/-inf/+inf, but got {mode}')

    add_mask = (mask - 1) * scale_size
    masked = torch.add(masked, add_mask)
    return masked


def pad_sequences(seqs, max_len, value=0., dtype=torch.int64):
    data_max_len = max(len(seq) for seq in seqs)
    max_len = min(max_len, data_max_len)
    padded = [seq[:max_len] + [value] * (max_len - len(seq)) for seq in seqs]
    return torch.tensor(padded, dtype=dtype)


def pad_sequences_for_3d(seqs, max_col, max_num, dtype=torch.int64):
    padded = [torch.cat((seq, torch.zeros((max_col - seq.shape[0], max_num), dtype=dtype))) for seq in seqs]
    return torch.stack(padded)


def pad_index_sequences(seqs, max_col, max_row, dtype=np.int64):
    padded = []
    for query in seqs:
        cols = [col[:max_col] + [0] * (max_col - len(col)) for col in query[:max_row]]
        new_cols = cols + [[0] * max_col for _ in range(max_row - len(cols))]
        padded.append(new_cols)
    return np.array(padded).astype(dtype)

def tensor2numpy(inputs):
    if isinstance(inputs, (list, tuple)):
        return [x.numpy() for x in inputs]
    elif isinstance(inputs, dict):
        outputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                outputs[key] = value.numpy()
            else:
                outputs[key] = value
        return outputs
    elif isinstance(inputs, torch.Tensor):
        return inputs.numpy()
    else:
        raise ValueError('only support inputs to be of type list/tuple/dict/Tensor.' + \
                         f'but got {type(inputs)}')


if __name__ == "__main__":
    """run some simple test cases"""
    seq_input = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 5, 5, 5],
        ], dtype=torch.float32)
    mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ], dtype=torch.float32)

    print(sequence_mask(seq_input, mask, mode='zero'))
    print(sequence_mask(seq_input, mask, mode='-inf'))
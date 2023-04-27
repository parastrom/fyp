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

import sys
import os
import traceback
import logging

import numpy as np
import torch
from torch import nn

from text2sql.models import relational_transformer


class RelationAwareEncoder(nn.Module):
    """Relation-aware encoder"""

    def __init__(self,
                 num_layers,
                 num_heads,
                 num_relations,
                 hidden_size,
                 has_value=False,
                 dropout=0.1):
        super(RelationAwareEncoder, self).__init__()

        self._num_layers = num_layers
        self._num_heads = num_heads
        self._hidden_size = hidden_size
        self._dropout = dropout

        cfg = {
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
            "num_relations": num_relations,
            "hidden_size": hidden_size,
            "hidden_act": "relu",
            "attention_probs_dropout_prob": dropout,
            "hidden_dropout_prob": dropout,
            "initializer_range": 0.02,
        }
        self.encoder = relational_transformer.RelationalTransformerEncoder(cfg)
        if not has_value:
            self.align_attn = relational_transformer.RelationalPointerNet(
                hidden_size, num_relations)
        else:
            self.align_attn = relational_transformer.RelationalPointerNet(
                hidden_size, 0)

    def forward(self,
                q_enc,
                c_enc,
                t_enc,
                c_boundaries,
                t_boundaries,
                relations,
                v_enc=None):
        assert q_enc.shape[0] == 1 and c_enc.shape[0] == 1 and t_enc.shape[
            0] == 1
        return self.forward_unbatched(q_enc, c_enc, t_enc, c_boundaries,
                                      t_boundaries, relations)

    def forward_unbatched(self,
                          q_enc,
                          c_enc,
                          t_enc,
                          c_boundaries,
                          t_boundaries,
                          relations,
                          v_enc=None):
        enc = torch.cat((q_enc, c_enc, t_enc), dim=1)
        # enc = enc.permute([1, 0, 2])

        relations_t = torch.tensor(relations, dtype=torch.int64, device='cuda').unsqueeze(0)
        enc_new, _, _ = self.encoder(enc, relations_t)

        # Split updated_enc again
        c_base = q_enc.shape[1]
        t_base = q_enc.shape[1] + c_enc.shape[1]
        q_enc_new = enc_new[:, :c_base]
        c_enc_new = enc_new[:, c_base:t_base]
        t_enc_new = enc_new[:, t_base:]

        if v_enc is None:
            m2c_align_mat = self.align_attn(enc_new, c_enc_new,
                                            relations_t[:, :, c_base:t_base])
            m2t_align_mat = self.align_attn(enc_new, t_enc_new,
                                            relations_t[:, :, t_base:])
            m2v_align_mat = None
        else:
            enc_new = torch.cat((enc_new, v_enc), dim=1)
            m2c_align_mat = self.align_attn(enc_new, c_enc_new, relations=None)
            m2t_align_mat = self.align_attn(enc_new, t_enc_new, relations=None)
            m2v_align_mat = self.align_attn(enc_new, v_enc, relations=None)

        return ([q_enc_new, c_enc_new, t_enc_new, v_enc],
                [m2c_align_mat, m2t_align_mat, m2v_align_mat])


if __name__ == "__main__":
    """run some simple test cases"""

    cuda = torch.cuda.is_available()
    device = torch.device('cuda') if cuda else torch.device('cpu')

    hidden_size = 4
    q = torch.tensor(
        list(range(12)), dtype=torch.float32, device=device).reshape([1, 3, hidden_size])
    c = torch.tensor(
        list(range(8)), dtype=torch.float32, device=device).reshape([1, 2, hidden_size])
    t = torch.tensor(
        list(range(8)), dtype=torch.float32, device=device).reshape([1, 2, hidden_size])
    c_bound = None
    t_bound = None
    relations = np.zeros([7, 7], dtype=np.int64)
    relations[0, 3] = 10
    relations[0, 1] = 1
    relations[0, 2] = 2
    relations[1, 2] = 1
    relations[1, 4] = 11
    relations[3, 4] = 21
    relations[3, 5] = 31

    model = RelationAwareEncoder(2, 2, 99, hidden_size).to(device=device)
    outputs = model(q, c, t, c_bound, t_bound, relations)
    print(outputs)


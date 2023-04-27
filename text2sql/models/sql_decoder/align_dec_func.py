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


def compute_align_loss(model, desc_enc, example):
    """model: a nl2code decoder"""
    # find relevant columns
    root_node = example.tree
    rel_cols = list(
        reversed([
            val
            for val in model.ast_wrapper.find_all_descendants_of_type(root_node,
                                                                      'column')
        ]))
    rel_tabs = list(
        reversed([
            val
            for val in model.ast_wrapper.find_all_descendants_of_type(root_node,
                                                                      'table')
        ]))
    rel_vals = np.abs(
        list(
            reversed([
                val
                for val in model.ast_wrapper.find_all_descendants_of_type(
                    root_node, 'value')
            ])))

    rel_cols_t = torch.tensor(sorted(list(set(rel_cols))), dtype=torch.int64, device='cuda:0')
    rel_tabs_t = torch.tensor(sorted(list(set(rel_tabs))), dtype=torch.int64, device='cuda:0')
    rel_vals_t = torch.tensor(sorted(list(set(rel_vals))), dtype=torch.int64, device='cuda:0')

    mc_att_on_rel_col = desc_enc.m2c_align_mat.index_select(1, rel_cols_t)
    mc_max_rel_att = mc_att_on_rel_col.max(axis=0).values
    mc_max_rel_att = mc_max_rel_att.clamp(min=1e-9)

    mt_att_on_rel_tab = desc_enc.m2t_align_mat.index_select(1, rel_tabs_t)
    mt_max_rel_att = mt_att_on_rel_tab.max(axis=0).values
    mt_max_rel_att = mt_max_rel_att.clamp(min=1e-9)

    mv_att_on_rel_val = desc_enc.m2v_align_mat.index_select(1, rel_vals_t)
    mv_max_rel_att = mv_att_on_rel_val.max(axis=0).values
    mv_max_rel_att = mv_max_rel_att.clip(min=1e-9)

    value_loss_weight = 2.0
    align_loss = - torch.log(mc_max_rel_att).mean() \
                 - torch.log(mt_max_rel_att).mean() \
                 - value_loss_weight * torch.log(mv_max_rel_att).mean()
    return align_loss


def compute_pointer_with_align(model, node_type, prev_state, prev_action_emb,
                               parent_h, parent_action_emb, desc_enc):
    """compute_pointer_with_align"""
    new_state, attention_weights = model._update_state(
        node_type, prev_state, prev_action_emb, parent_h, parent_action_emb,
        desc_enc)
    # output shape: batch (=1) x emb_size
    output = new_state[0]
    memory_pointer_logits = model.pointers[node_type](output, desc_enc.memory)
    memory_pointer_probs = torch.softmax(
        memory_pointer_logits, dim=1)
    # pointer_logits shape: batch (=1) x num choices
    if node_type == "column":
        pointer_probs = torch.mm(memory_pointer_probs, desc_enc.m2c_align_mat)
    elif node_type == 'table':
        pointer_probs = torch.mm(memory_pointer_probs, desc_enc.m2t_align_mat)
    else:  # value
        pointer_probs = torch.mm(memory_pointer_probs, desc_enc.m2v_align_mat)
    pointer_probs = pointer_probs.clip(min=1e-9)
    pointer_logits = torch.log(pointer_probs)
    return output, new_state, pointer_logits, attention_weights


if __name__ == "__main__":
    """run some simple test cases"""
    pass
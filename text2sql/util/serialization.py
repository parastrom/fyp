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
"""copied from https://github.com/electron1c/RAT-SQL-pytorch"""

def to_dict_with_sorted_values(d, key=None):
    """to dict with sorted values"""
    return {k: sorted(v, key=key) for k, v in d.items()}


def to_dict_with_set_values(d):
    """to dict with set values"""
    result = {}
    for k, v in d.items():
        hashable_v = [tuple(v_elem) if isinstance(v_elem, list) else v_elem for v_elem in v]
        result[k] = set(hashable_v)
    return result


def tuplify(x):
    """tuplify"""
    if not isinstance(x, (tuple, list)):
        return x
    return tuple(tuplify(elem) for elem in x)
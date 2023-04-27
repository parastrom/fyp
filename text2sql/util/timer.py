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
import json
import time
import re


class Timer(object):
    """Stat Cost Time"""

    def __init__(self, msg=""):
        super(Timer, self).__init__()

        self._msg = msg
        self._start = time.time()
        self._last = self._start

    def reset(self, only_last=False, msg=None):
        """reset all setting
        """
        if msg is not None:
            self._msg = msg
        curr_time = time.time()
        self._last = curr_time
        if not only_last:
            self._start = curr_time

    def check(self):
        """check cost time from start
        """
        end = time.time()
        cost = end - self._start
        return cost

    def interval(self):
        """check cost time from lst
        """
        end = time.time()
        cost = end - self._last
        self._last = end
        return cost

    def ending(self):
        """ending checking and log
        """
        cost = '%.2f' % time.time() - self._start
        if self._msg == "":
            log_msg = "cost time: %s" % (cost)
        elif '{}' in self._msg:
            log_msg = self._msg.format(cost)
        else:
            log_msg = self._msg + cost

        logging.info(log_msg)
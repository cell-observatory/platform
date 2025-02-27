
# https://github.com/huggingface/accelerate/blob/main/examples/by_feature/early_stopping.py

# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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


class EarlyStoppingCallback:

    def __init__(self, min_delta=0, patience=5):
        self.min_delta = min_delta
        self.patience = patience
        self.counter = 0
        self.lowest_loss = float("inf")

    def check_early_stopping(self, eval_loss):
        delta = self.lowest_loss - eval_loss
        if delta >= self.min_delta:
            self.lowest_loss = eval_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

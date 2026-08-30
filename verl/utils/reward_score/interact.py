# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

def compute_score(solution_str, ground_truth):
    """Reject the obsolete text-scoring path for TravelGym rollouts.

    TravelGym receives its terminal score from ``InteractTool``'s private
    environment ledger.  It must not be recomputed from a serialized response
    or ground-truth payload, so accidental calls through the generic VERL
    dispatcher fail loudly instead of returning the old ``None`` placeholder.
    """
    raise RuntimeError(
        "TravelGym rewards are terminal-only; use InteractTool.calc_reward "
        "and trainer-side reward metadata"
    )

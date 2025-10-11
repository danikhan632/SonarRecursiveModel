# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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

"""
Standalone M2M100 module for SONAR (from HuggingFace PR #29646)
Simplified to work as a top-level package without transformers internals.
"""

# Direct imports for standalone usage
try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False

if _torch_available:
    from .modeling_m2m_100 import (
        M2M100DecoderModel,
        M2M100Encoder,
        M2M100EncoderModel,
        M2M100ForConditionalGeneration,
        M2M100Model,
        M2M100PreTrainedModel,
    )

from .configuration_m2m_100 import M2M100Config, M2M100OnnxConfig
from .tokenization_m2m_100 import M2M100Tokenizer

__all__ = [
    "M2M100Config",
    "M2M100OnnxConfig",
    "M2M100Tokenizer",
]

if _torch_available:
    __all__.extend([
        "M2M100DecoderModel",
        "M2M100Encoder",
        "M2M100EncoderModel",
        "M2M100ForConditionalGeneration",
        "M2M100Model",
        "M2M100PreTrainedModel",
    ])

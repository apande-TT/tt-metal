# SPDX-License-Identifier: Apache-2.0
"""vLLM generator adapter for minicpm5_2b (LlamaForCausalLM) on Tenstorrent.

The tt-metal vLLM plugin registers this class as TTLlamaForCausalLM via vllm_metadata.json.
"""

try:
    from models.tt_transformers.tt.generator_vllm import LlamaForCausalLM as _Base
except Exception:  # available only inside the serving image
    _Base = object


class TTLlamaForCausalLM(_Base):
    """TT generator for LlamaForCausalLM (base: LlamaForCausalLM)."""

    # Stock arch: the base generator already serves this architecture. Nothing to override.
    pass

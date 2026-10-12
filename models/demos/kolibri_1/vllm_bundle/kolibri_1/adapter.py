# SPDX-License-Identifier: Apache-2.0
"""vLLM generator adapter for kolibri_1 (Kolibri1ForCausalLM) on Tenstorrent.

The tt-metal vLLM plugin registers this class as TTKolibri1ForCausalLM via vllm_metadata.json.
"""

try:
    from models.tt_transformers.tt.generator_vllm import LlamaForCausalLM as _Base
except Exception:  # available only inside the serving image
    _Base = object


class TTKolibri1ForCausalLM(_Base):
    """TT generator for Kolibri1ForCausalLM (base: LlamaForCausalLM)."""

    # TODO(author): this arch is not a plugin built-in. Wire the base generator to this
    #   demo's optimized model builder / weights loader (see the demo's tt/pipeline.py) and
    #   set any arch-specific config the base expects.
    pass

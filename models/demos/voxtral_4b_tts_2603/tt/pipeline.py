# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ONE shared chained TTNN pipeline for `mistralai/Voxtral-4B-TTS-2603`.

Both `demo/` and `tests/e2e/` import and call the functions here, so a passing
test guarantees a working demo -- there is exactly one copy of the wiring.

Three task heads are exposed, and between them they route all 10 graduated
modules from Source B:

  Call 1  text_generation  -- the composed leaf-stub stack
                              token_embed, rotary_embedding, r_m_s_norm,
                              attention, mlp, m_l_p, decoder_layer, layer,
                              decoder_head
  Call 2  hidden_states    -- the graduated whole-stack port
                              model
  Call 3  acoustic         -- the checkpoint's `acoustic_transformer` section,
                              composed from r_m_s_norm, attention, mlp,
                              rotary_embedding, decoder_head (see tt/acoustic.py)

SECTIONS. `consolidated.safetensors` declares THREE repeated block stacks --
`layers` (26), `audio_tokenizer.decoder_blocks` (8) and
`acoustic_transformer.layers` (3). Calls 1 and 2 port the first; Call 3 ports
the third. The audio tokenizer / vocoder is NOT ported: it is a weight-normed
causal-conv + sliding-window-attention stack with layer scale and QK norm, and
this checkpoint ships no reference implementation for it, so there is nothing
to gate a port against. That hole is recorded here, in the README and in
e2e_plan.json rather than papered over.

`PIPELINE_STAGES` is derived from Source A's config: the reference is
`MistralForCausalLM`, `is_encoder_decoder=False`, so the phases are
`[prefill, decode]`. There is no `encode` (decoder-only) and no `vocode`,
because the vocoder above is the part that has no port.
"""
from __future__ import annotations

import os

import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common

PIPELINE_STAGES = ["prefill", "decode"]

TASK_HEADS = ("text_generation", "hidden_states", "acoustic")

# A stack of fewer than three same-typed blocks is INVISIBLE to the structural walk that sizes and
# caps repeated stacks (`perf_automation/cc_optimize/_op_sig_probe.py::_walk_for_stacks` requires
# >= 3 members), so a cap below this floor hides the very stack the cap exists to size. Clamped up
# with a printed message, never silently.
MIN_DISCOVERABLE_LAYERS = 3

# Head name -> the attribute holding it. Only `text_generation` differs, and spelling that out
# beats a `str.replace` that would silently rename any head containing the substring.
_HEAD_ATTR = {"text_generation": "generation"}

# The pinned capacity C for the sequence axis. The VARIABLE dim is the sequence length, whose bound
# is `config.max_position_embeddings` (128000) -- far past anything a trace can hold, so the stage
# is pinned to a fixed tile-aligned C instead. 64 covers the package's real 32-token prompts plus
# the 16-step decode horizon with room to spare. Overridable, and shrunk (with a printed fallback)
# if a capture overflows the trace region.
DEFAULT_TRACE_CAPACITY = int(os.environ.get("VOXTRAL_TRACE_C", "64"))

# Additive mask fill; finite in bfloat16 so a fully-masked row cannot become inf - inf.
_MASK_NEG = -1e9

# Call 1's composed stack must keep all four block kinds alive (decoder_layer, layer,
# split+mlp, split+m_l_p) or a graduated stub becomes structurally absent rather than
# merely built fewer times.
MIN_GENERATION_LAYERS = 4


class VoxtralPipeline:
    """The resident pipeline object: both task heads plus the per-stage trace contract.

    Stacks are held as plain Python lists of same-typed elements so a structural walk can find,
    size and cap them, and the HF reference stays reachable on `.reference_model` -- it is ground
    truth for how many sections the model has and how deep each is.
    """

    PIPELINE_STAGES = PIPELINE_STAGES

    def __init__(self, device, hf_model, generation=None, hidden_states=None, acoustic=None, counter=None, batch=None):
        self.device = device
        self.reference_model = hf_model
        self.config = hf_model.config
        self.generation = generation
        self.hidden_states = hidden_states
        self.acoustic = acoustic
        self.counter = counter if counter is not None else common.InvocationCounter()
        self.batch = common.DEFAULT_BATCH if batch is None else int(batch)
        self.tokenizer = common.load_tokenizer()
        self.trace_capacity = DEFAULT_TRACE_CAPACITY
        self._stage_buffers: dict = {}

    # ---- task heads ----------------------------------------------------------------

    def run_text_generation(self, input_ids=None, horizon=None, **kwargs):
        """Call 1: real text generation over the composed graduated stack."""
        from models.demos.voxtral_4b_tts_2603.tt import generation

        if self.generation is None:
            raise RuntimeError("pipeline was built without the text_generation head")
        if input_ids is None:
            input_ids, _ = common.build_batch_inputs(batch=self.batch)
        return generation.run_text_generation(self.generation, input_ids, horizon=horizon, **kwargs)

    def run_hidden_states(self, input_ids=None, **kwargs):
        """Call 2: text -> last_hidden_state over the graduated whole-stack `model` port."""
        from models.demos.voxtral_4b_tts_2603.tt import hidden_states as hs

        if self.hidden_states is None:
            raise RuntimeError("pipeline was built without the hidden_states head")
        if input_ids is None:
            input_ids, _ = common.build_batch_inputs(batch=self.batch)
        return hs.run_hidden_states(self.hidden_states, input_ids, **kwargs)

    def run_acoustic(self, lm_hidden=None, input_ids=None, **kwargs):
        """Call 3: the checkpoint's `acoustic_transformer` section, fed by the text backbone.

        With no `lm_hidden` the text stack is run first and its hidden state is handed over as a
        DEVICE tensor -- the two sections are chained on the device, not through the host.
        """
        from models.demos.voxtral_4b_tts_2603.tt import acoustic as ac

        if self.acoustic is None:
            raise RuntimeError("pipeline was built without the acoustic head")
        owned = None
        if lm_hidden is None:
            if self.generation is None:
                raise RuntimeError("the acoustic head needs a text hidden state; build text_generation too")
            if input_ids is None:
                input_ids, _ = common.build_batch_inputs(batch=self.batch)
            owned = lm_hidden = self.generation.forward_hidden(input_ids)
        try:
            return ac.run_acoustic(self.acoustic, lm_hidden, **kwargs)
        finally:
            if owned is not None:
                ttnn.deallocate(owned)

    # ---- structure -----------------------------------------------------------------

    @property
    def stacks(self) -> dict:
        """The repeated blocks, one plain list per section, for sizing / capping / attribution."""
        out = {}
        if self.generation is not None:
            out["text_generation"] = self.generation.layers
        if self.hidden_states is not None:
            out["hidden_states"] = self.hidden_states.layers
        if self.acoustic is not None:
            out["acoustic"] = self.acoustic.layers
        return out

    def describe(self) -> dict:
        return {
            "batch": self.batch,
            "stages": list(self.PIPELINE_STAGES),
            "heads": [name for name in TASK_HEADS if getattr(self, _HEAD_ATTR.get(name, name), None) is not None],
            "stack_depths": {k: len(v) for k, v in self.stacks.items()},
            "reference_depth": len(self.reference_model.model.layers),
            "trace_capacity": self.trace_capacity,
        }

    # =====================================================================================
    # COMMAND 3 -- the trace contract, one set of hooks per entry in PIPELINE_STAGES.
    #
    # Both stages ride the SAME composed text stack (this is a decoder-only model), pinned to a
    # fixed capacity C on the sequence axis. Everything shape-dependent -- the padded ids, the
    # RoPE cos/sin and the causal mask -- is taken FROM THE HF REFERENCE and uploaded into
    # persistent device buffers OUTSIDE the trace, so `<stage>_trace_step()` reads only those
    # buffers and fires no host op.
    #
    # prefill RIGHT-pads: causality leaves rows [0:real_len] bit-identical and the pad keys are
    #   masked, so the answer is `hidden[:, :real_len]`. The next-token row index (real_len-1) is
    #   fixed by the setup, so the slice is a static one the trace can record.
    # decode LEFT-pads: the newest token is pinned at row C-1 every step, so the traced step is
    #   index-invariant as the context grows. The graduated blocks expose no KV-cache surface, so a
    #   decode step recomputes the resident context at fixed C -- host-free and traceable, but NOT
    #   an incremental cache. Recorded as a hole in e2e_plan.json and the README.
    # =====================================================================================

    def _require_generation(self):
        if self.generation is None:
            raise RuntimeError("the trace contract lives on the text_generation head; build it first")
        return self.generation

    # ---- constants, taken from the HF reference so they match the golden exactly -----

    def _hf_rope(self, position_ids):
        """(cos, sin) straight out of `hf.model.rotary_emb` -- the reference's own tables."""
        hf = self.reference_model
        carrier = torch.zeros(1, position_ids.shape[-1], hf.config.hidden_size, dtype=torch.float32)
        with torch.no_grad():
            cos, sin = hf.model.rotary_emb(carrier, position_ids)
        return cos.to(torch.float32), sin.to(torch.float32)

    def _hf_causal_mask(self, attention_mask, position_ids, seq_len):
        """The additive causal mask from `transformers.masking_utils.create_causal_mask`.

        Passing a 0/1 `attention_mask` is what masks the padded positions: the padded KEYS become
        -inf columns, so the real rows are unchanged by the padding.
        """
        from transformers.masking_utils import create_causal_mask

        hf = self.reference_model
        batch = int(attention_mask.shape[0])
        embeds = torch.zeros(batch, seq_len, hf.config.hidden_size, dtype=torch.float32)
        with torch.no_grad():
            mask = create_causal_mask(
                config=hf.config,
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=position_ids,
            )
        if mask is None:
            # create_causal_mask returns None when the attention backend builds its own; the port
            # needs an explicit additive mask, so fall back to the same tensor it would imply.
            blocked = torch.ones(seq_len, seq_len, dtype=torch.bool).triu(1)
            mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float32).masked_fill_(blocked, _MASK_NEG)
            pad = attention_mask[:1] == 0
            mask = mask.masked_fill(pad.reshape(1, 1, 1, seq_len), _MASK_NEG)
        mask = mask.to(torch.float32)
        finite = torch.finfo(torch.float32).min
        return mask.masked_fill(mask <= finite / 2, _MASK_NEG)

    def _upload(self, t, dtype):
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)

    def _stage_setup(self, stage, inputs):
        """Shared body of `<stage>_trace_setup`: pin C and pre-upload every constant."""
        stack = self._require_generation()
        input_ids = inputs["input_ids"]
        if input_ids.dim() == 1:
            input_ids = input_ids.reshape(1, -1)
        capacity = int(inputs.get("capacity") or self.trace_capacity)
        batch, real_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        bound = int(self.config.max_position_embeddings)
        if capacity > bound:
            raise ValueError(f"capacity {capacity} exceeds max_position_embeddings {bound}")
        if real_len > capacity:
            raise ValueError(f"real length {real_len} exceeds the pinned capacity {capacity}")

        pad_id = int(getattr(self.config, "bos_token_id", 1) or 1)
        padded = torch.full((batch, capacity), pad_id, dtype=torch.long)
        keep = torch.zeros(batch, capacity, dtype=torch.long)
        positions = torch.zeros(batch, capacity, dtype=torch.long)
        if stage == "prefill":
            padded[:, :real_len] = input_ids
            keep[:, :real_len] = 1
            positions[:, :real_len] = torch.arange(real_len)
            last_row = real_len - 1
        else:
            padded[:, capacity - real_len :] = input_ids
            keep[:, capacity - real_len :] = 1
            positions[:, capacity - real_len :] = torch.arange(real_len)
            last_row = capacity - 1

        cos, sin = self._hf_rope(positions[:1])
        mask = self._hf_causal_mask(keep, positions, capacity)

        buffers = {
            "ids": ttnn.from_torch(
                padded.to(torch.uint32).contiguous(),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            ),
            "ids_host": ttnn.from_torch(
                padded.to(torch.uint32).contiguous(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
            "cos": self._upload(cos.reshape(1, 1, capacity, -1), stack.act_dtype),
            "sin": self._upload(sin.reshape(1, 1, capacity, -1), stack.act_dtype),
            # The additive mask goes up in bfloat16, NOT in `act_dtype`: its only consumer is the
            # fused flash-attention op, which takes bf16/bf8_b/bf4_b and nothing wider, and -1e9 is
            # just as absorbing after a softmax at bf16's precision as at fp32's. Uploading it wide
            # would buy a per-layer typecast and no accuracy.
            "mask": self._upload(mask.reshape(mask.shape[0], 1, capacity, capacity), ttnn.bfloat16),
            "capacity": capacity,
            "real_len": real_len,
            "batch": batch,
            "last_row": last_row,
            "padded_ids": padded,
            "keep": keep,
            "positions": positions,
        }
        self._stage_buffers[stage] = buffers
        return buffers

    def _stage_step(self, stage, sample):
        """Shared body of `<stage>_trace_step`: ONE host-op-free forward at the pinned shape."""
        stack = self._require_generation()
        buf = self._stage_buffers.get(stage)
        if buf is None:
            raise RuntimeError(f"call {stage}_trace_setup() before {stage}_trace_step()")
        hidden = stack.forward_resident(buf["ids"], (buf["cos"], buf["sin"]), buf["mask"])
        width = int(hidden.shape[-1])
        last = ttnn.slice(hidden, (0, buf["last_row"], 0), (buf["batch"], buf["last_row"] + 1, width))
        logits = stack.head(last)
        ttnn.deallocate(last)
        if not sample:
            return {"hidden": hidden, "logits": logits}
        ttnn.deallocate(hidden)
        row_major = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        token = ttnn.argmax(row_major, dim=-1)
        ttnn.deallocate(row_major)
        ttnn.deallocate(logits)
        return {"token": token}

    def _stage_inputs(self):
        """The ZERO-ARG seam. Model-specific assembly lives here, behind the fixed name.

        Returns exactly the value `<stage>_trace_setup` takes. The inputs are the SAME real golden
        inputs the e2e PCC test and the demo drive -- `common.build_batch_inputs()`, the tekken
        tokenizer over 32 distinct texts. (Source B's `_captured/` carries a single 64-token row
        for `model` / `token_embed`, which cannot supply 32 independent samples; the captured row
        is checked separately by `tests/e2e/test_e2e_hidden_states.py::test_captured_bringup_golden`.)
        """
        input_ids, _ = common.build_batch_inputs(batch=self.batch)
        return {"input_ids": input_ids, "capacity": self.trace_capacity}

    # ---- prefill ---------------------------------------------------------------------

    def prefill_trace_inputs(self):
        return self._stage_inputs()

    def prefill_trace_setup(self, inputs):
        return self._stage_setup("prefill", inputs)

    def prefill_trace_step(self):
        return self._stage_step("prefill", sample=False)

    def prefill_trace_items(self) -> int:
        """Items retired by ONE `prefill_trace_step()`: every pinned position, for every sample.

        The repeated blocks process all C positions of all B samples, so the total is B*C. Stating
        1 here would price the stage's arithmetic ceiling (2 x params x items) B*C times too small
        and then report a compute-bound stage as memory-bound.
        """
        buf = self._stage_buffers.get("prefill")
        capacity = buf["capacity"] if buf else self.trace_capacity
        batch = buf["batch"] if buf else self.batch
        return int(batch) * int(capacity)

    # ---- decode ----------------------------------------------------------------------

    def decode_trace_inputs(self):
        return self._stage_inputs()

    def decode_trace_setup(self, inputs):
        return self._stage_setup("decode", inputs)

    def decode_trace_step(self):
        return self._stage_step("decode", sample=True)

    def decode_trace_items(self) -> int:
        """Items retired by ONE `decode_trace_step()`: one token per sample, so B."""
        buf = self._stage_buffers.get("decode")
        return int(buf["batch"]) if buf else int(self.batch)

    # ---- the AR decode contract -------------------------------------------------------

    def decode_prefill(self, inputs):
        """Seed the resident decode context. No cross-attention: this model is decoder-only.

        HONEST LIMITATION: the graduated blocks are full-attention with no KV-cache surface, so
        there is no self-attn KV to seed either -- what is seeded is the left-padded context at
        the pinned capacity, which `decode_step()` re-reads (never re-uploads) each step.
        """
        return self._stage_setup("decode", inputs)

    def decode_step(self):
        return self._stage_step("decode", sample=True)

    # There is deliberately NO `decode_write_inputs(input_ids)` here any more. It was a host ->
    # device write of the next decode window, i.e. exactly the host token feed the on-device
    # contract forbids: the loop would pick a token on device, drag it to the host, splice it into
    # a torch window and upload the whole thing again. The feed now lives entirely on device --
    # `generation.append_token` concatenates the `ttnn.argmax` result onto the resident context --
    # and the traced decode step reads the resident buffers `decode_prefill` seeded.

    # ---- selftests ---------------------------------------------------------------------

    def trace_capture_selftest(self, device=None, pcc_target: float = 0.99) -> bool:
        """Capture, execute and release ONE trace per stage; True only if all of them match.

        Stage traces must not co-reside, so each is released before the next is captured. If a
        capture overflows the trace region the capacity is halved and the fallback is PRINTED --
        never silently dropped.
        """
        device = device if device is not None else self.device
        stack = self._require_generation()
        ok = True
        for stage in self.PIPELINE_STAGES:
            capacity = self.trace_capacity
            while True:
                inputs = getattr(self, f"{stage}_trace_inputs")()
                inputs["capacity"] = capacity
                getattr(self, f"{stage}_trace_setup")(inputs)
                eager = getattr(self, f"{stage}_trace_step")()
                reference = {k: ttnn.to_torch(v).to(torch.float32) for k, v in eager.items()}
                for value in eager.values():
                    ttnn.deallocate(value)
                try:
                    tid = ttnn.begin_trace_capture(device, cq_id=0)
                    traced = getattr(self, f"{stage}_trace_step")()
                    ttnn.end_trace_capture(device, tid, cq_id=0)
                except RuntimeError as exc:
                    if "trace" not in str(exc).lower() or capacity <= 32:
                        raise
                    capacity //= 2
                    print(
                        f"[voxtral_4b_tts_2603] {stage}: trace capture overflowed the region; "
                        f"shrinking capacity C to {capacity} and retrying"
                    )
                    continue
                ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
                for key, ref in reference.items():
                    got = ttnn.to_torch(traced[key]).to(torch.float32)
                    score = common.pcc(got, ref)
                    print(f"[voxtral_4b_tts_2603] trace {stage}.{key}: C={capacity} PCC={score}")
                    ok = ok and score >= pcc_target
                ttnn.release_trace(device, tid)
                for value in traced.values():
                    ttnn.deallocate(value)
                self._release_stage(stage)
                break
        return ok

    def _release_stage(self, stage):
        buf = self._stage_buffers.pop(stage, None)
        if not buf:
            return
        for key in ("ids", "cos", "sin", "mask"):
            try:
                ttnn.deallocate(buf[key])
            except Exception:  # noqa: BLE001 - already-freed buffers are fine to skip
                pass

    def host_op_selftest(self) -> dict:
        """The AUTHORITATIVE fully-on-device check, per task head.

        Input ENCODING (tokenize) and the one-time weight build happen OUTSIDE the observed region;
        the model math -- encoded ids through the embedding, every block, the head and the
        on-device sampling -- happens INSIDE it. ttnn ops do not dispatch through torch, so a truly
        on-device forward fires ZERO host aten ops; anything that shows up is host compute the
        ttnn-crossing checks cannot see.
        """
        from scripts.tt_hw_planner import host_op_observer

        results = {}

        if self.generation is not None:
            stack = self.generation
            input_ids, _ = common.build_batch_inputs(batch=self.batch)
            self.prefill_trace_setup({"input_ids": input_ids, "capacity": self.trace_capacity})
            buf = self._stage_buffers["prefill"]
            with host_op_observer.observe_host_ops() as ops:
                hidden = stack.forward_resident(buf["ids"], (buf["cos"], buf["sin"]), buf["mask"])
                width = int(hidden.shape[-1])
                last = ttnn.slice(hidden, (0, buf["last_row"], 0), (buf["batch"], buf["last_row"] + 1, width))
                logits = stack.head(last)
                row_major = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
                token = ttnn.argmax(row_major, dim=-1)
            for value in (hidden, last, logits, row_major, token):
                ttnn.deallocate(value)
            self._release_stage("prefill")
            results["text_generation"] = host_op_observer.verdict(list(ops))

        if self.hidden_states is not None:
            stub = self.hidden_states.stub
            input_ids, _ = common.build_batch_inputs(batch=self.batch)
            prepared = self.hidden_states.prepare_inputs(input_ids)
            with host_op_observer.observe_host_ops() as ops:
                out = stub(
                    prepared["input_ids"],
                    position_embeddings=prepared["position_embeddings"],
                    attention_mask=prepared["attention_mask"],
                )
            ttnn.deallocate(out)
            for key in ("input_ids", "attention_mask"):
                ttnn.deallocate(prepared[key])
            for value in prepared["position_embeddings"]:
                ttnn.deallocate(value)
            results["hidden_states"] = host_op_observer.verdict(list(ops))

        if self.acoustic is not None:
            stack = self.acoustic
            # The conditioning hidden state is staged OUTSIDE the observed region, exactly as the
            # encoded ids are for the other heads; only the acoustic forward is observed.
            seq_len = common.DEFAULT_SEQ_LEN
            carrier = torch.zeros(self.batch, seq_len, int(stack.config.hidden_size), dtype=torch.float32)
            staged = ttnn.from_torch(
                carrier.contiguous(), dtype=stack.act_dtype, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            stack.stage_constants(seq_len)
            with host_op_observer.observe_host_ops() as ops:
                hidden, semantic = stack.forward_semantic_logits(staged)
            for value in (hidden, semantic, staged):
                ttnn.deallocate(value)
            results["acoustic"] = host_op_observer.verdict(list(ops))

        results["on_device"] = all(v["on_device"] for k, v in results.items() if k != "on_device")
        return results


def _resolve_depth(layers, prefill_layers, decode_layers, available, minimum, label, why):
    """One repeated text stack is shared by BOTH stages, so the two overrides must collapse.

    `layers` is the default depth for every repeated block; `prefill_layers` / `decode_layers` are
    the per-stack overrides named after the PIPELINE_STAGES entries that own a stack. This model
    has a single text decoder feeding both stages, so when the two overrides disagree the build
    takes the max and PRINTS that it collapsed them rather than silently picking one.
    """
    picks = [v for v in (prefill_layers, decode_layers) if v is not None]
    if len(picks) == 2 and picks[0] != picks[1]:
        print(
            f"[voxtral_4b_tts_2603] prefill_layers={prefill_layers} and decode_layers={decode_layers} "
            f"address the SAME shared text stack; collapsing to {max(picks)}"
        )
    depth = max(picks) if picks else layers
    if depth is None:
        return available
    depth = int(depth)
    if depth <= 0:
        raise ValueError(f"layers={depth} would build a zero-layer model; pass None for every layer")
    if depth < minimum:
        print(f"[voxtral_4b_tts_2603] {label}: layers={depth} would {why}; clamping up to {minimum}")
        depth = minimum
    return min(depth, available)


def build_pipeline(device, model=None, layers=None, prefill_layers=None, decode_layers=None, **kwargs):
    """Construct and RETURN the resident pipeline object. Does not run anything.

    `layers` caps the depth built for EVERY repeated stack (None = every layer, never 0);
    embeddings, norms, rotary tables and the LM head stay intact so a capped build still
    exercises every DISTINCT op the full model runs, just fewer times. `prefill_layers` /
    `decode_layers` are the per-stack overrides, each falling back to `layers`.

    Any extra demo kwargs (text, prompt, language, ...) are accepted and ignored: the resident
    build derives its shapes from the config, not from a prompt.
    """
    from models.demos.voxtral_4b_tts_2603.tt import acoustic as ac
    from models.demos.voxtral_4b_tts_2603.tt import generation
    from models.demos.voxtral_4b_tts_2603.tt import hidden_states as hs

    heads = kwargs.pop("heads", None) or TASK_HEADS
    heads = tuple(heads)
    unknown = [h for h in heads if h not in TASK_HEADS]
    if unknown:
        raise ValueError(f"unknown head(s) {unknown}; this pipeline exposes {list(TASK_HEADS)}")
    batch = kwargs.pop("batch", None)

    env_layers = os.environ.get("TT_PERF_LAYERS")
    if layers is None and prefill_layers is None and decode_layers is None and env_layers:
        layers = int(env_layers)

    hf_model = model if model is not None else common.load_reference_model()
    available = len(hf_model.model.layers)
    counter = common.InvocationCounter()

    gen_stack = None
    if "text_generation" in heads:
        depth = _resolve_depth(
            layers,
            prefill_layers,
            decode_layers,
            available,
            MIN_GENERATION_LAYERS,
            "text_generation",
            "leave a graduated block structurally absent",
        )
        gen_stack = generation.build_generation_stack(device, hf_model, layers=depth, counter=counter)

    hs_stack = None
    if "hidden_states" in heads:
        depth = _resolve_depth(
            layers,
            prefill_layers,
            decode_layers,
            available,
            MIN_DISCOVERABLE_LAYERS,
            "hidden_states",
            "hide the stack from the structural walk that sizes and caps it",
        )
        hs_stack = hs.build_hidden_states_stack(device, hf_model, layers=depth, counter=counter)

    # The acoustic section declares three blocks in total and is NOT capped: see
    # `acoustic.build_acoustic_stack` -- capping it would take it below the depth at which a stack
    # is discoverable at all, which is the opposite of what a cap is for.
    ac_stack = ac.build_acoustic_stack(device, counter=counter) if "acoustic" in heads else None

    return VoxtralPipeline(
        device,
        hf_model,
        generation=gen_stack,
        hidden_states=hs_stack,
        acoustic=ac_stack,
        counter=counter,
        batch=batch,
    )


# =========================================================================================
# The ZERO-ARG module entries the bring-up observers bind.
#
# `scripts/tt_hw_planner/_host_op_probe.py` and `_trace_capture_probe.py` import THIS module in a
# fresh process with no device open and call `host_op_selftest()` / `trace_capture_selftest()` with
# no arguments. The methods of the same name on `VoxtralPipeline` are the implementations and take
# an already-built pipeline on an already-open device -- which is what `tests/e2e/` drives. These
# two functions are the standalone wrappers: they own a device for the length of the check and
# hand it straight to the same code path, so the observed pipeline and the tested pipeline are the
# same object built the same way. The open itself lives in `device_session` (outside `tt/`) so the
# pipeline package keeps its no-self-open guarantee.
# =========================================================================================


def _selftest_pipeline(device, heads):
    return build_pipeline(device, heads=heads)


def trace_capture_selftest(device=None, pcc_target: float = 0.99) -> bool:
    """Capture / execute / release one real device trace per stage. True only if all of them match.

    The trace contract lives on the `text_generation` head, so that is the head built here.
    """
    if device is not None:
        return _selftest_pipeline(device, ("text_generation",)).trace_capture_selftest(device, pcc_target=pcc_target)

    from models.demos.voxtral_4b_tts_2603 import device_session

    with device_session.selftest_device() as own:
        return _selftest_pipeline(own, ("text_generation",)).trace_capture_selftest(own, pcc_target=pcc_target)


def host_op_selftest(device=None) -> dict:
    """The AUTHORITATIVE fully-on-device verdict, for EVERY task head. Zero host aten ops, or fail."""
    if device is not None:
        return _selftest_pipeline(device, TASK_HEADS).host_op_selftest()

    from models.demos.voxtral_4b_tts_2603 import device_session

    # No trace is captured here, so the trace region is dead weight next to a 26-layer build.
    with device_session.selftest_device(trace_region_size=0) as own:
        return _selftest_pipeline(own, TASK_HEADS).host_op_selftest()

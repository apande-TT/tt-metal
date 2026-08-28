# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TEXT ENCODER stage of the FLUX.2-klein-9B TTNN pipeline.

Two heads share one checkpoint (``Qwen3ForCausalLM``, 9 B, bf16) and one set of
ten graduated bring-up stubs:

``Flux2PromptEmbedStage``
    what the image heads consume.  Verbatim
    ``Flux2KleinPipeline._get_qwen3_prompt_embeds``: run the Qwen3 trunk with
    ``output_hidden_states=True``, take ``hidden_states[k]`` for k in (9, 18, 27),
    ``stack(dim=1).permute(0,2,1,3).reshape(B, L, 3*4096)``.  ``hidden_states[k]``
    is the residual activation AFTER k layers (``hidden_states[0]`` is the embedding
    output), so only layers 0..26 run and neither ``model.norm`` nor ``lm_head``
    is on this path.  The concat order along the last axis is tap9|tap18|tap27.

``Qwen3CausalLmStage``
    the text-generation head: the whole ``Qwen3Model`` in one graduated port
    (``encoder_stack``) plus ``lm_head`` (``decoder_head``).

Port -> position map (each graduated stub sits at its OWN position, its output
feeding the next):

===========================  ==============================================
stub                         position
===========================  ==============================================
token_embed                  model.embed_tokens
rotary_embedding             model.rotary_emb
layer                        model.layers[0]                      (fused)
r_m_s_norm x2 + attention
  + mlp                      model.layers[1]                      (DECOMPOSED)
r_m_s_norm x2 + attention
  + m_l_p                    model.layers[2]                      (DECOMPOSED)
decoder_layer                model.layers[3] .. model.layers[26]  (24 ports)
encoder_stack                model            (embed + 36 layers + final norm)
decoder_head                 lm_head
===========================  ==============================================

pin / step
----------
Every head is split into a ``pin`` that does ALL the host work once and a
``step`` that is host-op-free, so the same body can be traced.  The graduated
stubs build their RoPE tables and their additive score bias on the HOST and
cache them -- ``layer`` / ``decoder_layer`` / ``attention`` key the rope cache on
``id(position_embeddings)`` and the bias cache on ``id(attention_mask)``,
``encoder_stack`` keys them on ``seq_len`` and ``id(attention_mask)``.  ``pin``
therefore stages the device buffers, builds those host-side constants ONCE, and
runs one full forward so every cache inside every port is warm; ``step`` hands
back the very same Python objects, so each of those lookups is a pure cache hit
and no ``ttnn.from_torch`` / ``torch.*`` fires.  The resident dict owns the cache
keys, and the stage additionally retains every object it has ever handed to a
stub cache, so a dropped resident can never let CPython recycle an ``id()`` into
a stale entry.  ``pin`` may be called repeatedly at different lengths; residents
are independent.

No torch compute happens in either forward path.  torch is used for weight
marshalling at build time, inside ``pin`` for the additive attention bias (a
constant of the sequence, built exactly as the graduated stubs build their own
causal bias), and for two host round trips the graduated stub bodies force:

* ``rotary_embedding`` returns ttnn ``(cos, sin)``, but ``layer`` /
  ``decoder_layer`` / ``attention`` broadcast ``position_embeddings`` with
  ``torch.Tensor.expand`` / ``.contiguous()``, which ``ttnn.Tensor`` does not
  provide.  So the pair is read back to host bf16 (a copy, no arithmetic) inside
  ``pin`` and handed on.  Both halves stay bf16 end to end, which is what HF's
  ``Qwen3RotaryEmbedding`` returns for a bf16 model and what the graduated
  ``encoder_stack`` stages internally.
* the decode loop reads back ONE int PER STREAM per step to test the stop rule.

Mask.  HF hands the trunk the padding mask and ``create_causal_mask`` turns it
into causal+padding.  ``layer`` / ``decoder_layer`` / ``encoder_stack`` build a
causal bias themselves when ``attention_mask is None``, but ``attention`` adds no
bias at all in that case, so this stage always builds ONE additive float
``(B, 1, L, L)`` bias on the host -- 0 where a query may attend, -1e9 on the
strict upper triangle or on a padded key column -- and passes it to every layer
port and every attention port.  It is deliberately non-constant along the score
axis so the stubs' ``_score_bias`` / ``_mask_bias`` materialise and broadcast it
instead of dropping it as a softmax no-op.  Padded rows therefore match HF,
which matters because ``prompt_embeds`` includes the pad positions.

Batch
-----
Both heads carry a leading batch axis of B INDEPENDENT samples -- B different
prompts, run as ONE device program per layer, never a Python loop over samples.
What is and is not per-sample:

* **per sample**: the ids ``(B, L)``, the additive bias ``(B, 1, L, L)`` (32
  prompts have 32 different real lengths, so a single shared ``(1, 1, L, L)``
  bias would hand samples 1..31 sample 0's padding), the activations, the
  logits, and the decode's live/finished flag.
* **shared, broadcast**: the RoPE ``(cos, sin)`` tables.  HF derives positions
  from ``cache_position`` -- ``arange(L)`` for every row, padding included --
  not from the mask, so one ``(1, n_heads, L, head_dim)`` table is the RIGHT
  answer for all B rows and ttnn broadcasts it over the leading axis.

The causal-LM head pads a batch on the LEFT (what HF's own batched
``generate()`` does), so every row's last real token sits in the same column and
the whole batch shares ONE cursor -- that is what lets the decode advance all B
streams with one ``ttnn.concat`` per step.  The prompt-embed head keeps the
tokenizer's right padding, since ``prompt_embeds`` must include the pad
positions in the same places HF puts them.

Nothing here changes the TP=8 weight sharding: batch is a separate axis from the
sharded head / intermediate / vocab axes.
"""

from __future__ import annotations

import torch

import ttnn

# Host-side staging and readback live OUTSIDE tt/ (see host_inputs' docstring): they
# run before and after the device forward, never inside it.  Imported by name so the
# call sites read the same as they did when the helpers lived here.
from ..host_inputs import replicate_mapper, to_host  # noqa: F401  -- to_host is re-exported
from . import stubs
from .depth import stack_depth

#: The bring-up directory this stage owns.
STAGE = "text_encoder"

#: Same fill the graduated stubs use for their own causal bias, so a supplied
#: mask and a stub-built one are numerically interchangeable.
MASK_FILL = -1e9


# ---------------------------------------------------------------------- batch


def batch_of(x) -> int:
    """The leading-axis width of an id tensor, host or device.  A 1-D sequence is
    one sample, which is what every rank-1 caller in the pipeline means."""
    shape = getattr(x, "shape", None)
    if shape is None or len(shape) < 2:
        return 1
    return int(shape[0])


def stage_batched_ids(tokens, device):
    """``host_inputs.stage_ids`` with the leading batch axis KEPT.

    ``stage_ids`` reshapes to ``(1, -1)``, which is right for one sample and
    silently splices B prompts into a single B*L-token sequence.  Same uint32 /
    ROW_MAJOR contract -- ids must never become bfloat16, which cannot hold an id
    above 256 exactly -- and the same replication across the mesh.
    """
    if not isinstance(tokens, torch.Tensor):
        return tokens  # already staged
    rows = tokens.reshape(1, -1) if tokens.ndim < 2 else tokens.reshape(int(tokens.shape[0]), -1)
    return ttnn.from_torch(
        rows.to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        mesh_mapper=replicate_mapper(device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ----------------------------------------------------------------------- mask


def keep_rows(attention_mask, width: int) -> torch.Tensor:
    """A mask of any accepted form as a ``(B, width)`` bool matrix."""
    keep = attention_mask.reshape(-1, int(attention_mask.shape[-1])).to(torch.bool)
    return keep[:, :width]


def real_length(input_ids, attention_mask=None) -> int:
    """The column just past the LAST real token, shared by every row.

    The trace contract hands ``pin_*`` a sequence ALREADY padded out to the
    traced capacity, so the tensor's length is the capacity, not the token count
    -- the mask is what says where the real tokens stop.

    Two padding conventions reach this function and both are honoured:

    * RIGHT padding (a LEADING run of 1s) -- the tokenizer's and the pipeline's
      ``_pad_ids``' form for a single prompt; the answer is that row's token count.
    * LEFT padding (a TRAILING run of 1s) -- what a batched greedy decode needs,
      because it puts every row's last real token in the SAME column, so B streams
      share one cursor and advance with one device concat per step.

    Either way the answer is ``max(last real column) + 1``, and a row whose real
    tokens are not one contiguous run is rejected rather than silently mis-sliced.
    """
    width = int(input_ids.shape[-1])
    if attention_mask is None:
        return width
    keep = keep_rows(attention_mask, width)
    counts = keep.sum(dim=1)
    columns = torch.arange(int(keep.shape[1])).reshape(1, -1).expand_as(keep)
    first = torch.where(keep, columns, torch.full_like(columns, int(keep.shape[1]))).amin(dim=1)
    last = torch.where(keep, columns, torch.full_like(columns, -1)).amax(dim=1)
    if not bool(((counts == 0) | (last - first + 1 == counts)).all()):
        raise NotImplementedError(
            "attention_mask rows must be ONE contiguous run of 1s (left- or " "right-padded); got interior zeros"
        )
    return int(torch.where(counts == 0, torch.full_like(last, width), last + 1).amax())


def attention_bias(seq_len: int, attention_mask=None) -> torch.Tensor:
    """The additive score bias HF's ``create_causal_mask`` is equivalent to.

    ``(B, 1, L, L)``: 0 where query q of sample b may attend to key k,
    ``MASK_FILL`` where it may not, i.e. ``k <= q`` AND ``attention_mask[b, k] ==
    1``.  ONE ROW PER SAMPLE -- B prompts have B different padding patterns, and
    a shared ``(1, 1, L, L)`` bias would give samples 1..B-1 sample 0's padding
    while raising no error at all.  Built on the host once per pinned sequence:
    it is a constant of (L, padding), not of the activations.
    """
    allowed = torch.ones(1, seq_len, seq_len, dtype=torch.bool).tril()
    if attention_mask is not None:
        keep = keep_rows(attention_mask, seq_len)
        allowed = allowed & keep.reshape(int(keep.shape[0]), 1, seq_len)
    bias = torch.zeros(allowed.shape, dtype=torch.float32).masked_fill(~allowed, MASK_FILL)
    return bias.reshape(int(bias.shape[0]), 1, seq_len, seq_len)


def decode_keep(attention_mask, batch: int, length: int, capacity: int, cursor: int) -> torch.Tensor:
    """Which key columns a decode query may read, for the WHOLE run, ``(B, capacity)``.

    Columns at or after the cursor are the slots the generated tokens land in, so
    they are real by the time any query row reaches them, and the pinned causal
    bias keeps them unread before that.  Columns before the cursor are real iff
    the caller's mask says so -- which is how B different prompt lengths ride
    through ONE bias object, and the bias has to be one object because the
    graduated stubs cache it on ``id()``.
    """
    keep = torch.ones(int(batch), int(capacity), dtype=torch.int64)
    if attention_mask is not None:
        supplied = attention_mask.reshape(-1, int(attention_mask.shape[-1])).to(torch.int64)
        n = min(int(supplied.shape[-1]), int(length), int(capacity), int(cursor))
        keep[:, :n] = supplied[:, :n]
    keep[:, int(cursor) :] = 1
    return keep


# --------------------------------------------------------------- shared helper


class PortFactory:
    """The one shared helper: builds a graduated stub into a port at a named
    position, binds it to the ledger, and owns the two bits of bookkeeping both
    heads need (consumption reporting and pinned-object retention).

    ``stubs.load_stub_module`` byte-compares the live stub body against its
    graduated snapshot first (Gate 1), so nothing here can drift from bring-up.
    """

    def __init__(self, device, ledger=None) -> None:
        self.device = device
        self.ledger = ledger
        #: every host object ever handed to an id()-keyed stub cache
        self.pinned: list = []

    def build(self, name: str, position: str, torch_module):
        module = stubs.load_stub_module(STAGE, name)
        port = module.build(self.device, torch_module)
        if self.ledger is None:
            return port
        return self.ledger.bind(STAGE, name, position, port)

    def consumed(self, *tensors) -> None:
        """Tell the ledger these port outputs were used.

        The ledger follows dataflow by tensor identity, which only holds while a
        port's output is handed to the next port unchanged.  Three joints in this
        stage break the identity without breaking the dataflow: a residual
        ``ttnn.add`` (the sum, not the branch, is what flows on), the last-row
        ``ttnn.slice`` before the LM head, and the ``rotary_embedding`` host
        round trip the stub bodies force.  Report those explicitly rather than
        leave a real consumer invisible.
        """
        if self.ledger is None:
            return
        for tensor in tensors:
            if tensor is not None:
                self.ledger.mark_final(tensor)

    def residual_add(self, residual, branch):
        out = ttnn.add(residual, branch)
        self.consumed(branch)
        return out

    def retain(self, *objects) -> None:
        """Keep a strong reference to every object used as an ``id()`` cache key
        inside a graduated stub.  Without this, dropping an old resident would
        let CPython hand the same ``id()`` to a new bias / rope tuple and the
        stub would answer from a stale cache entry."""
        self.pinned.extend(objects)


# ----------------------------------------------------------- prompt-embed head


class _TrunkBlock:
    """One position of the Qwen3 trunk, whichever graduated stubs fill it.

    Every element of ``Flux2PromptEmbedStage.blocks`` is one of these, so a
    structure walk sees a single repeated type and can size the stack even
    though positions 1 and 2 are decomposed into four ports each.

    Two things this class must keep for that walk to work at all:

    * **no ``__slots__``.**  ``find_all_stacks`` keeps only the sequence elements
      that carry a ``__dict__``, so a slotted block is an INVISIBLE block and the
      whole trunk reads as "no stack here".
    * **ports staged in ``build()``, not in ``__init__``.**  The stack has to exist
      before any weight touches the device, so that constructing the pipeline is
      enough to make its sections walkable.
    """

    def __init__(self, position: str, index: int, factory: PortFactory, hf_layer) -> None:
        self.position = position
        self.index = int(index)
        #: "fused" (one whole-block port) | "decomposed" (four ports)
        self.kind = "decomposed" if self.index in (1, 2) else "fused"
        self.ports: dict = {}
        self._factory = factory
        self._hf = hf_layer

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"_TrunkBlock({self.position}, {self.kind}, {sorted(self.ports)})"

    def build(self) -> "_TrunkBlock":
        """Stage this position's graduated ports.  Idempotent."""
        if self.ports:
            return self
        i, layer = self.index, self._hf
        if self.kind == "decomposed":
            # DECOMPOSED: the two RMSNorm ports are built from the actual
            # Qwen3RMSNorm modules at this position.  They expose .weight and
            # .variance_epsilon, which is exactly what the r_m_s_norm stub
            # reads, so no shim object is needed.  Position 1 takes the `mlp`
            # stub, position 2 the `m_l_p` stub -- two scaffold names for the
            # same module, each graduated on its own source.
            mlp_stub = "mlp" if i == 1 else "m_l_p"
            self.ports = {
                "input_layernorm": self._factory.build(
                    "r_m_s_norm", f"model.layers.{i}.input_layernorm", layer.input_layernorm
                ),
                "self_attn": self._factory.build("attention", f"model.layers.{i}.self_attn", layer.self_attn),
                "post_attention_layernorm": self._factory.build(
                    "r_m_s_norm",
                    f"model.layers.{i}.post_attention_layernorm",
                    layer.post_attention_layernorm,
                ),
                "mlp": self._factory.build(mlp_stub, f"model.layers.{i}.mlp", layer.mlp),
            }
        else:
            name = "layer" if i == 0 else "decoder_layer"
            self.ports = {"block": self._factory.build(name, self.position, layer)}
        return self

    def __call__(self, hidden_states, position_embeddings, attention_mask):
        if self.kind == "fused":
            return self.ports["block"](
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )

        # h = x + attn(input_layernorm(x)); y = h + mlp(post_attention_layernorm(h))
        x = hidden_states
        normed = self.ports["input_layernorm"](hidden_states=x)
        attn = self.ports["self_attn"](
            hidden_states=normed,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        x = self._factory.residual_add(x, attn)
        normed = self.ports["post_attention_layernorm"](hidden_states=x)
        mlp = self.ports["mlp"](hidden_states=normed)
        return self._factory.residual_add(x, mlp)


class Flux2PromptEmbedStage:
    """``prompt_embeds`` for the image heads: ``(B, L, 3 * 4096)``."""

    #: From ``Flux2KleinPipeline`` (``text_encoder_out_layers``).  Not invented here.
    OUT_LAYERS = (9, 18, 27)

    def __init__(self, device, hf_text_encoder, *, ledger=None, layers=None) -> None:
        """STRUCTURE ONLY -- no weight is staged here.

        ``self.blocks`` is the trunk's repeated stack and it exists in full the moment
        the stage is constructed, so a structure walk over a freshly-built pipeline
        sees the section at its real depth without a device having been written to.
        ``build()`` stages the ports the first time the stage is actually used.
        """
        self.device = device
        self.ledger = ledger
        self._factory = PortFactory(device, ledger)
        self._hf = hf_text_encoder

        model = hf_text_encoder.model
        available = len(model.layers)
        # hidden_states[27] is the residual activation after 27 layers, so the trunk
        # stops at layers[26]: model.norm and lm_head are NOT on this path.
        depth = min(max(self.OUT_LAYERS), available)
        self._n_layers = depth if layers is None else stack_depth(layers, available)

        self.token_embed = None
        self.rotary_embedding = None
        self.blocks: list = [
            _TrunkBlock(f"model.layers.{i}", i, self._factory, model.layers[i]) for i in range(self._n_layers)
        ]
        self._built = False

        # Clamp the taps to the depth actually built, so a capped trunk still
        # runs and still emits a 12288-wide tensor (the taps then coincide).
        self.out_layers = tuple(min(tap, self._n_layers) for tap in self.OUT_LAYERS)

    def build(self) -> "Flux2PromptEmbedStage":
        """Stage every port this trunk needs, on the device given at construction.
        Idempotent, and called by every entry point, so a caller never has to."""
        if self._built:
            return self
        model = self._hf.model
        self.token_embed = self._factory.build("token_embed", "model.embed_tokens", model.embed_tokens)
        self.rotary_embedding = self._factory.build("rotary_embedding", "model.rotary_emb", model.rotary_emb)
        for block in self.blocks:
            block.build()
        self._built = True
        return self

    # ------------------------------------------------------------- properties
    @property
    def staged(self) -> bool:
        """True once ``build()`` has put weights on the device.  The pipeline reads
        this to decide whether releasing this stage frees anything."""
        return self._built

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def hidden_size(self) -> int:
        return int(self.build().token_embed.embedding_dim)

    @property
    def out_features(self) -> int:
        return len(self.OUT_LAYERS) * self.hidden_size

    # -------------------------------------------------------------- pin / step
    def pin(self, input_ids, attention_mask) -> dict:
        """All host work for one (batch of sequences, mask), once.

        Stages the ``(B, L)`` ids into a persistent device buffer, builds the rope
        pair and the additive bias as single host objects, then runs ONE full
        forward so every id()-keyed cache inside every graduated port is warm.  The
        returned dict owns those objects; ``step`` must hand back the very same ones.

        B samples ride the leading axis of ONE program per layer.  The bias is
        per-sample ``(B, 1, L, L)`` because B prompts pad at B different columns;
        the rope pair stays ``(1, L, head_dim)`` and broadcasts, because HF takes
        its positions from ``arange(L)`` for every row -- padding included -- not
        from the mask.
        """
        self.build()
        seq_len = int(input_ids.shape[-1])
        batch = batch_of(input_ids)

        bias = attention_bias(seq_len, attention_mask)
        # ONE rotary call for the whole trunk, positions 0..L-1, shared by all B rows.
        position_ids = torch.arange(seq_len, dtype=torch.int32).reshape(1, seq_len)
        cos_tt, sin_tt = self.rotary_embedding(position_ids=position_ids)
        position_embeddings = (to_host(cos_tt, self.device), to_host(sin_tt, self.device))
        self._factory.consumed(cos_tt, sin_tt)

        resident = {
            "head": "prompt_embeds",
            "seq_len": seq_len,
            "batch": batch,
            "input_ids": stage_batched_ids(input_ids, self.device),
            "attention_mask": bias,
            "position_embeddings": position_embeddings,
        }
        self._factory.retain(bias, position_embeddings)
        warm = self._forward(resident)  # warms every stub cache at this seq_len
        del warm
        return resident

    def step(self, resident) -> ttnn.Tensor:
        """One host-op-free forward over a pinned resident set. ``(B, L, 12288)``."""
        return self._forward(resident)

    def __call__(self, input_ids, attention_mask) -> ttnn.Tensor:
        return self.step(self.pin(input_ids, attention_mask))

    # ---------------------------------------------------------------- forward
    def _forward(self, resident) -> ttnn.Tensor:
        position_embeddings = resident["position_embeddings"]
        bias = resident["attention_mask"]

        x = self.token_embed(input_ids=resident["input_ids"])  # (B, L, 4096), TILE

        taps: dict = {}
        for depth, block in enumerate(self.blocks, start=1):
            x = block(x, position_embeddings, bias)
            if depth in self.out_layers:
                taps[depth] = x

        wanted = [taps[tap] for tap in self.out_layers]
        self._factory.consumed(*wanted)
        return ttnn.concat(wanted, dim=-1)  # tap9 | tap18 | tap27


# ------------------------------------------------------- text-generation head


class _TruncatedQwen3Model:
    """Weight-marshalling proxy exposing exactly what ``encoder_stack.build``
    reads -- ``.config``, ``.layers``, ``.embed_tokens`` and
    ``.state_dict()["norm.weight"]`` -- with ``layers`` truncated.

    A shallow view: the kept ``Qwen3DecoderLayer`` objects and the state_dict
    entries are the HF model's own tensors, and the cached HF model is never
    mutated, so the other pipeline stages still see all 36 layers.
    """

    def __init__(self, model, n_layers: int) -> None:
        self.config = model.config
        self.layers = list(model.layers)[:n_layers]
        self.embed_tokens = model.embed_tokens
        self.norm = model.norm
        self._model = model
        self._n_layers = n_layers

    def state_dict(self):
        kept = {}
        for key, value in self._model.state_dict().items():
            if key.startswith("layers."):
                index = int(key.split(".")[1])
                if index >= self._n_layers:
                    continue
            kept[key] = value
        return kept


class Qwen3CausalLmStage:
    """The text->text head: ``Qwen3Model`` + ``lm_head``, greedy, no KV cache.

    The graduated ``encoder_stack`` body has no cache, so each decode step
    re-runs the whole prefix.  That is what the stub is; nothing is bolted on.
    """

    def __init__(self, device, hf_text_encoder, *, ledger=None, layers=None) -> None:
        """STRUCTURE ONLY; ``build()`` stages the two ports (see
        ``Flux2PromptEmbedStage.__init__`` for why the split exists)."""
        self.device = device
        self.ledger = ledger
        self._factory = PortFactory(device, ledger)
        self._hf = hf_text_encoder

        model = hf_text_encoder.model
        available = len(model.layers)
        self._n_layers = available if layers is None else stack_depth(layers, available)

        self.encoder_stack = None
        self.decoder_head = None
        #: The repeated block list.  ``encoder_stack`` is monolithic: it stages ALL
        #: of its layers' weights once, at build time, and holds them resident for
        #: the whole run -- nothing here is loaded or evicted per step -- so the
        #: weight dict it keeps for each layer is what this head's stack is made of.
        #: Filled by build(), because those dicts exist only once the port is staged;
        #: the prompt-embed stage holds the walkable copy of this same section.
        self.blocks: list = []
        self._built = False

        #: host logits row per decode step, when generate(collect_logits=True)
        self.step_logits: list = []
        #: the resident set left behind by the last generate()
        self.last_resident = None

    def build(self) -> "Qwen3CausalLmStage":
        if self._built:
            return self
        model = self._hf.model
        trunk = model if self._n_layers == len(model.layers) else _TruncatedQwen3Model(model, self._n_layers)
        self.encoder_stack = self._factory.build("encoder_stack", "model", trunk)
        self.decoder_head = self._factory.build("decoder_head", "lm_head", self._hf.lm_head)
        self.blocks = list(self.encoder_stack.layers)
        self._built = True
        return self

    # ------------------------------------------------------------- properties
    @property
    def staged(self) -> bool:
        """True once ``build()`` has put weights on the device."""
        return self._built

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def vocab_size(self) -> int:
        return int(self.build().decoder_head.vocab_size)

    # ------------------------------------------------------------ pin / step
    def _pin(self, head: str, ids, capacity: int, real: int, bias) -> dict:
        resident = {
            "head": head,
            "ids": ids,
            "capacity": int(capacity),
            "attention_mask": bias,
            # a 1-element list, so a step can advance the cursor without the
            # caller's resident dict being replaced
            "cursor": [int(real)],
        }
        self._factory.retain(bias)
        warm = self._trunk(resident)  # warms rope / bias / embed-table caches
        del warm
        return resident

    def pin_prefill(self, input_ids, attention_mask=None) -> dict:
        """Pin one prefill: ``(B, L)`` ids on device, bias on host, every cache warm.

        ``input_ids`` may already be padded out to the traced capacity; the mask
        decides which column is the last REAL token, so the returned logits row is
        the one HF's ``logits[:, -1, :]`` would give for the unpadded prompt.  At
        B > 1 the batch is LEFT-padded, so that column is the same for every row
        and one slice serves all B.
        """
        self.build()
        ids = stage_batched_ids(input_ids, self.device)
        seq_len = int(ids.shape[-1])
        real = real_length(input_ids, attention_mask)
        return self._pin("prefill", ids, seq_len, real, attention_bias(seq_len, attention_mask))

    def prefill_step(self, resident) -> ttnn.Tensor:
        """One host-op-free prefill forward: last-row logits ``(B, 1, V)``."""
        return self._trunk(resident)

    def prefill_logits(self, input_ids) -> ttnn.Tensor:
        return self.prefill_step(self.pin_prefill(input_ids))

    def pin_decode(self, input_ids, attention_mask, capacity: int, *, pad_id: int = 0) -> dict:
        """Pin the sequence axis to ``capacity`` for a whole decode run.

        The buffer is ``[prompt columns][zero pad]`` at length ``capacity``, one
        row per sample.  The trunk is causal, so query row q only ever reads keys
        0..q: columns ``[0:cursor]`` are bit-identical to running the unpadded
        prefix, and the junk tail is inert.  That is what lets ``decode_step`` keep
        ONE traced shape however many tokens have been generated -- only the row
        index of the final slice moves.

        ``input_ids`` may arrive already padded out to ``capacity`` (that is what
        the pipeline's ``_pad_ids`` produces); ``attention_mask`` is then what says
        which of those columns are real, and the cursor starts on the last real
        column rather than on the last column of the buffer.

        Batch.  At B > 1 the prompts are LEFT-padded, so every row's last real
        token is in column ``cursor - 1`` and all B streams share one cursor and
        one ``ttnn.concat`` per step.  Their B different prompt lengths are carried
        by the pinned per-row bias, which is built ONCE (the graduated stubs cache
        it on ``id()``, so a per-step mask object would miss every cache).
        """
        self.build()
        capacity = int(capacity)
        length = int(input_ids.shape[-1])
        batch = batch_of(input_ids)
        real = real_length(input_ids, attention_mask)
        if capacity < real:
            raise ValueError(f"capacity {capacity} is shorter than the {real}-token prompt")

        if isinstance(input_ids, torch.Tensor):
            rows = input_ids.reshape(batch, -1)
            padded = torch.zeros(batch, capacity, dtype=torch.int64)
            keep_n = min(length, capacity)
            padded[:, :keep_n] = rows[:, :keep_n]
            ids = stage_batched_ids(padded, self.device)
        elif length > capacity:  # already staged: trim / extend on device
            ids = ttnn.slice(input_ids, [0, 0], [batch, capacity])
        elif length < capacity:
            pad = stage_batched_ids(torch.zeros(batch, capacity - length, dtype=torch.int64), self.device)
            ids = ttnn.concat([input_ids, pad], dim=1)
        else:
            ids = input_ids

        # Columns before the cursor are real iff the caller's mask says so -- for a
        # right-padded single prompt that is all of them, which reproduces the plain
        # causal bias this used to pin; for a left-padded batch it is what keeps
        # sample b from attending to sample b's own pad columns.  Columns at or after
        # the cursor become real generated tokens as the cursor advances.
        keep = decode_keep(attention_mask, batch, length, capacity, real)
        resident = self._pin("decode", ids, capacity, real, attention_bias(capacity, keep))
        resident["batch"] = batch
        # a spare all-zero buffer to re-pad the tail from, so advance() never
        # touches the host
        resident["pad"] = stage_batched_ids(torch.zeros(batch, capacity, dtype=torch.int64), self.device)
        # per-stream decode bookkeeping, staged here so the loop stays host-free:
        # `live` is 1 while a stream has not yet emitted a stop id, and a dead
        # stream is frozen to `pad_id` instead of being fed its own next argmax.
        resident["live"] = stage_batched_ids(torch.ones(batch, 1, dtype=torch.int64), self.device)
        resident["ones"] = stage_batched_ids(torch.ones(batch, 1, dtype=torch.int64), self.device)
        resident["pad_id"] = stage_batched_ids(torch.full((batch, 1), int(pad_id), dtype=torch.int64), self.device)
        return resident

    def decode_step(self, resident) -> ttnn.Tensor:
        """ONE decode step at the pinned capacity: last-row logits ``(B, 1, V)``.

        Host-op-free.  Every tensor shape is fixed by ``(batch, capacity)``; the
        only thing that moves between steps is the integer row index of the final
        slice, which is shared by all B streams because they are left-padded.
        """
        return self._trunk(resident)

    def advance(self, resident, token) -> None:
        """Write ``token`` ``(B, 1)`` at the cursor, keeping the buffer at the
        pinned capacity.  Device ops only -- the trunk's input shape never changes,
        and all B streams move together."""
        cursor = resident["cursor"][0]
        capacity = resident["capacity"]
        batch = int(resident["ids"].shape[0])
        if cursor + 1 > capacity:
            raise ValueError(f"decode ran past the pinned capacity {capacity}")
        pieces = [ttnn.slice(resident["ids"], [0, 0], [batch, cursor]), token]
        if cursor + 1 < capacity:
            pieces.append(ttnn.slice(resident["pad"], [0, 0], [batch, capacity - cursor - 1]))
        resident["ids"] = ttnn.concat(pieces, dim=1)
        resident["cursor"][0] = cursor + 1

    # ------------------------------------------------ per-stream stop, on device
    def _freeze(self, resident, token):
        """``token`` for a live stream, the pad id for a finished one, ``(B, 1)``.

        Device-only, uint32: ``token * live + pad * (1 - live)``.  A stream that
        has already emitted a stop id is frozen rather than fed its own next
        argmax, which is what "one finished stream does not drag the batch" means.
        """
        live = resident.get("live")
        if live is None:
            return token
        dead = ttnn.sub(resident["ones"], live)
        return ttnn.add(ttnn.mul(token, live), ttnn.mul(resident["pad_id"], dead))

    def _retire(self, resident, token, stops) -> None:
        """Clear the live flag of every stream whose ``token`` is a stop id.

        On device, and elementwise per row: ``live *= 1 - sum(token == s)``.  The
        stop ids are distinct, so at most one comparison fires; a stream that is
        already dead stays dead whatever its frozen token compares to.
        """
        live = resident.get("live")
        if live is None or not stops:
            return
        hit = ttnn.eq(token, int(stops[0]))
        for stop in stops[1:]:
            hit = ttnn.add(hit, ttnn.eq(token, int(stop)))
        resident["live"] = ttnn.mul(live, ttnn.sub(resident["ones"], hit))

    # ---------------------------------------------------------------- forward
    def _trunk(self, resident) -> ttnn.Tensor:
        hidden = self.encoder_stack(
            input_ids=resident["ids"], attention_mask=resident["attention_mask"]
        )  # (B, capacity, 4096)
        row = resident["cursor"][0]
        width = int(hidden.shape[-1])
        batch = int(hidden.shape[0])
        last = ttnn.slice(hidden, [0, row - 1, 0], [batch, row, width])  # (B, 1, 4096)
        self._factory.consumed(hidden)
        return self.decoder_head(hidden_states=last)  # (B, 1, V)

    # --------------------------------------------------------------- generate
    def generate(
        self,
        input_ids,
        max_new_tokens: int,
        stop_ids=None,
        collect_logits: bool = False,
        attention_mask=None,
        *,
        capacity: int | None = None,
        pad_id: int = 0,
    ):
        """Greedy decode of B streams in lockstep over a pinned, fixed-capacity buffer.

        Host-compute-free in the loop: the trunk, the row slice, the head, the
        argmax, the per-stream freeze and the id write all run on device; only the
        B new ids come back, to test the stop rule.  There is ONE program per step
        for the whole batch -- never a loop over samples.

        ``input_ids`` is ``(B, L)``.  For B > 1 it must be LEFT-padded with the
        matching ``(B, L)`` ``attention_mask``, which is what HF's own batched
        ``generate()`` does: it puts every row's last real token in the same
        column, so one argmax row per step serves all B.  For B == 1 the mask may
        stay None.

        The loop stops when EVERY stream has emitted a stop id, or at
        ``max_new_tokens``, whichever comes first.  A stream that has stopped is
        frozen to ``pad_id`` on device and its later tokens are dropped, so a long
        stream cannot put words in a finished one's mouth.

        Returns, B == 1:  ``list[int]``  (unchanged).
        Returns, B > 1:   ``list[list[int]]``, row b truncated at row b's OWN first
        stop id.
        With ``collect_logits=True`` the second element is the per-step last-row
        logits, ``(B, V)`` fp32 -- read back strictly AFTER the on-device argmax,
        so it is a measurement tap and never feeds the decode.
        """
        stops = sorted({int(s) for s in (stop_ids or [])})
        batch = batch_of(input_ids)
        real = real_length(input_ids, attention_mask)
        resident = self.pin_decode(input_ids, attention_mask, capacity or real + int(max_new_tokens), pad_id=pad_id)
        self.last_resident = resident
        self.step_logits = []
        rows: list = [[] for _ in range(batch)]
        done = [False] * batch

        for _ in range(int(max_new_tokens)):  # max_new_tokens is always the cap
            logits = self.decode_step(resident)
            token = ttnn.argmax(logits, dim=-1)  # (B, 1) uint32 ROW_MAJOR
            self._factory.consumed(logits)
            self.advance(resident, self._freeze(resident, token))
            self._retire(resident, token, stops)

            chosen = to_host(token, self.device).reshape(-1)
            if collect_logits:  # measurement tap, strictly after the device argmax
                self.step_logits.append(to_host(logits, self.device).reshape(batch, -1).to(torch.float32))
            for b in range(batch):  # bookkeeping only; no sample's maths is in here
                if done[b]:
                    continue
                rows[b].append(int(chosen[b]))
                if rows[b][-1] in stops:
                    done[b] = True
            if all(done):
                break

        self._factory.consumed(resident["ids"])
        new_ids = rows[0] if batch == 1 else rows
        return (new_ids, self.step_logits) if collect_logits else new_ids

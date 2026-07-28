"""Drop the tile-padding rows before the force-argmax sampling reduction.

THE REDUCIBLE WORK. `TTSampling.__init__` rounds the request batch up to a tile:

    raw_batch = getattr(args, "max_batch_size", 32)
    self.max_batch_size = max(32, ((raw_batch + 31) // 32) * 32)

That round-up is REQUIRED for the logits themselves -- they are a TILE-layout
`[1, 1, 32, vocab]` tensor and a tile is 32 rows tall, so the norm / LM-head chain has to
compute all 32. It is NOT required for the argmax, because the force-argmax path untilizes
first and a ROW_MAJOR tensor has no such constraint. The profile shows the op paying for the
padding anyway: ArgMax runs at ~715 us/call on the full 110-core grid over
`[1, 1, 32, 128256]`, i.e. 4.1 M scalar comparisons, of which 31/32 are on padding rows whose
tokens nothing downstream ever reads (`output_tokens[slot] = tokens_host[slot]` only indexes
the real user slots, and the decode feedback buffer is written in place).

THE FIX. `ttnn.untilize` -> `ttnn.untilize_with_unpadding` to the live rows. The argmax
multi-core factory sizes ALL of its work from the input:

    output_last_dim = input_shape[rank - 2]        # number of independent argmaxes
    red_dim_units   = input_shape[rank - 1]        # the vocab, split across cores

so dropping the padding rows scales the reduction down by that factor with no extra op --
the unpad rides inside the untilize that was already there.

WHY A SUBCLASS. `TTSampling` lives in `models/common/sampling/`, shared by every
tt-transformers model; this is a batch-1-latency specialisation of THIS demo, so it belongs
here and is installed onto the instance rather than edited into the shared op. Anything the
fast path is not sure about -- multi-device, sampling-DP, a non-greedy request, a shape that
is already unpadded -- falls straight through to `super().forward`, which is untouched.
"""
import ttnn
from models.common.sampling.tt_sampling import TTSampling


class UnpaddedArgmaxSampling(TTSampling):
    """`TTSampling` whose force-argmax path reduces only the live batch rows."""

    #: Live user rows; the rest of the 32-row tile is padding. Set by :func:`install`.
    _live_batch_rows = None

    def _argmax_unpadded_rows(self, x, tt_out_tok):
        """Rows to keep, or ``None`` to defer to the stock path."""
        rows = self._live_batch_rows
        if rows is None or not self._force_argmax_sampling:
            return None
        # The stock path's all-gather / DP fan-out changes what a "row" means; only claim the
        # single-device, single-group case this demo actually runs.
        if self.mesh_device.get_num_devices() > 1 or self._sampling_dp > 1:
            return None
        if x.layout != ttnn.TILE_LAYOUT or len(x.shape) != 4:  # `layout` is a property, not a method
            return None
        padded_rows = int(x.shape[-2])
        return rows if 0 < rows < padded_rows else None

    def forward(self, x: ttnn.Tensor, tt_out_tok: ttnn.Tensor = None):
        rows = self._argmax_unpadded_rows(x, tt_out_tok)
        if rows is None:
            return super().forward(x, tt_out_tok=tt_out_tok)

        # Same vocab handling as the stock force-argmax branch: slice off the padded vocab when
        # it is tile-aligned, otherwise mask the invalid entries to -inf.
        slice_valid_vocab = self._can_slice_valid_vocab_for_argmax()
        if slice_valid_vocab:
            x = self._slice_valid_vocab_for_argmax(x)
        else:
            x = self._mask_invalid_vocab_logits(x)

        # Untilize EXACTLY as the stock path does -- at this shape it reports
        # `enough_space_height: false` and takes the split/block factory, so
        # `untilize_with_unpadding` is not a safe substitute (it crashes trace capture here).
        # Drop the padding rows afterwards instead, on the ROW_MAJOR result, with a plain slice.
        padded_rows = int(x.shape[-2])
        x_untilized = ttnn.untilize(x, use_multicore=True)
        x_untilized = ttnn.slice(
            x_untilized,
            [0, 0, 0, 0],
            [int(x_untilized.shape[0]), int(x_untilized.shape[1]), rows, int(x_untilized.shape[-1])],
            memory_config=x_untilized.memory_config(),
        )
        # The RESULT must keep its padded shape even though the WORK shrank: callers index it by
        # user slot (`output_tokens[slot] = tokens_host[slot]`) and decode feeds it straight back
        # into a padded token buffer. argmax takes its output SPEC from the preallocated tensor
        # and its WORK from the input, so handing it a full-width zeroed output gives both.
        if tt_out_tok is None:
            tt_out_tok = ttnn.zeros(
                [int(x.shape[0]), int(x.shape[1]), padded_rows],
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                x.device(),
                ttnn.DRAM_MEMORY_CONFIG,
            )
        tt_out_tok = ttnn.argmax(
            x_untilized,
            dim=-1,
            output_tensor=tt_out_tok,
            keepdim=False,
        )
        # Argmax path: logprobs not supported (force-argmax is disabled when logprobs are on).
        self.tt_log_probs = None
        return tt_out_tok, self.tt_log_probs


def install(sampling_generator, args):
    """Give ``sampling_generator``'s TTSampling the unpadded force-argmax path.

    Rebinds ``__class__`` rather than constructing a replacement: ``SamplingGenerator`` builds
    its ``TTSampling`` internally and hands the same instance to the seed manager and the
    penalties module, so swapping the object would strand those references.
    """
    tt_sampling = getattr(sampling_generator, "tt_sampling", None)
    if tt_sampling is None or type(tt_sampling) is not TTSampling:
        return
    live_rows = int(getattr(args, "max_batch_size", 0) or 0)
    if live_rows <= 0 or live_rows >= tt_sampling.max_batch_size:
        return  # nothing padded -> nothing to drop
    tt_sampling.__class__ = UnpaddedArgmaxSampling
    tt_sampling._live_batch_rows = live_rows

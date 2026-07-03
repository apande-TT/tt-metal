# TTNN port: `lambda` of `ACE-Step/acestep-v15-base`

**This env's `transformers` could not load the HF model**, so the upstream source could not be auto-inlined. The prompt below is still self-contained — you just need to paste the relevant `nn.Module` source into the `## HF source` section manually.

Loader error (for context):
```
Could not resolve any candidate path for `lambda` of ACE-Step/acestep-v15-base. Tried: []
```

---

## How to use this prompt

1. Visit the upstream HF transformers source for `ACE-Step/acestep-v15-base`:
   - transformers/src/transformers/models/acestep/modeling_acestep.py
2. Find the `nn.Module` class that corresponds to `lambda` — likely one of these access paths inside the model:
  (no candidate paths registered)
3. Copy the class source (plus any helper classes it calls) into the `## HF source` block below.
4. Paste the entire `## Prompt` section into your chat assistant.
5. Save the response and apply it:

```bash
python -m scripts.tt_hw_planner bringup ACE-Step/acestep-v15-base \
    --apply-response lambda _synth_responses/lambda.py
python -m scripts.tt_hw_planner bringup ACE-Step/acestep-v15-base \
    --run-tests --component lambda
```

The response will overwrite `models/demos/hf_eager/acestep_v15_base/_stubs/lambda.py` after a syntax check + backup.

---

## Prompt

### System

You are an expert TTNN engineer porting PyTorch modules from HuggingFace transformers to the Tenstorrent TTNN runtime. You produce Python code that runs on Tenstorrent Wormhole/Blackhole hardware. You follow project conventions exactly. You never invent ops that are not in the provided cheatsheet.

### Task

Port the PyTorch class below to TTNN. Produce the complete contents of `_stubs/lambda.py` and nothing else.

TARGET
------
model_id        : ACE-Step/acestep-v15-base
component name  : lambda
class to expose : Lambda
module-level fn : lambda
HF reference    : transformers/src/transformers/models/acestep/modeling_acestep.py
new-model shape : {}
candidate paths inside the HF model:
  (no candidate paths registered)

## HF source

```python
# PASTE THE UPSTREAM HF SOURCE HERE.
#
# Open the file at `transformers/src/transformers/models/acestep/modeling_acestep.py` in the
# transformers GitHub repo, copy the nn.Module class that corresponds
# to `lambda`, and paste it between these fences. Include
# any helper classes that the class' __init__ / forward references.
```

## TTNN op cheatsheet (the only ops you may use)

```
TTNN op surface available for synthesis. Use ONLY these. If a feature you
need is missing, leave a `# TODO(ttnn-gap): ...` comment and skip it rather
than inventing an op.

# Tensor lifecycle
ttnn.from_torch(t: torch.Tensor, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None) -> ttnn.Tensor
ttnn.to_torch(t: ttnn.Tensor) -> torch.Tensor
ttnn.to_device(t, device)
ttnn.from_device(t)
ttnn.to_layout(t, layout)
ttnn.deallocate(t)

# Layouts and dtypes
ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32, ttnn.uint32, ttnn.int32

# Linear algebra
ttnn.matmul(a, b, *, transpose_a=False, transpose_b=False, dtype=None, memory_config=None) -> ttnn.Tensor
ttnn.linear(input, weight, *, bias=None, transpose_b=True, dtype=None) -> ttnn.Tensor
ttnn.add(a, b), ttnn.sub(a, b), ttnn.mul(a, b), ttnn.div(a, b)

# Activations
ttnn.relu(x), ttnn.gelu(x), ttnn.silu(x), ttnn.sigmoid(x), ttnn.tanh(x)
ttnn.softmax(x, dim=-1)

# Norms
ttnn.layer_norm(x, *, weight=None, bias=None, epsilon=1e-5) -> ttnn.Tensor
ttnn.rms_norm(x, *, weight=None, epsilon=1e-6) -> ttnn.Tensor
ttnn.group_norm(x, *, num_groups, weight=None, bias=None, epsilon=1e-5)

# Shape ops
ttnn.reshape(x, shape: tuple) -> ttnn.Tensor
ttnn.permute(x, dims: tuple) -> ttnn.Tensor
ttnn.transpose(x, dim0, dim1) -> ttnn.Tensor
ttnn.concat(tensors: list, dim) -> ttnn.Tensor
ttnn.split(x, sections, dim)
ttnn.unsqueeze(x, dim), ttnn.squeeze(x, dim)
ttnn.repeat(x, shape), ttnn.repeat_interleave(x, repeats, dim)
ttnn.tile(x, dims)

# Convs / pools (vision)
ttnn.conv2d(input, weight, *, bias=None, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, batch_size, input_height, input_width)
ttnn.max_pool2d(x, *, batch_size, input_h, input_w, channels, kernel_size, stride, padding=(0,0))
ttnn.avg_pool2d(x, *, batch_size, input_h, input_w, channels, kernel_size, stride, padding=(0,0))
ttnn.upsample(x, scale_factor)  # nearest only
ttnn.interpolate(x, size=None, scale_factor=None, mode="nearest")

# Fused transformer
ttnn.transformer.scaled_dot_product_attention(q, k, v, *, is_causal=False, attn_mask=None, scale=None) -> ttnn.Tensor
ttnn.transformer.attention_softmax(scores, *, scale=None, attention_mask=None)

# Convention: every TTNN op returns a new ttnn.Tensor on-device; chain them.
# Convention: weight tensors are loaded once in __init__ via ttnn.from_torch
#             from torch_module.state_dict(); do NOT re-upload per call.

# COMMON MISTAKES — ttnn.Tensor has a DIFFERENT API than torch.Tensor.
# Calling torch methods on a ttnn.Tensor raises AttributeError at runtime.
# These are typical hallucinations the agent makes after reading the torch
# reference; substitute the listed ttnn replacement instead.
ttnn_t.float()              # WRONG  -> ttnn.typecast(t, ttnn.float32)
                            #          OR ttnn.to_torch(t).float() if you need torch
ttnn_t.to(device)           # WRONG  -> ttnn.to_device(t, device)
ttnn_t.cpu()                # WRONG  -> ttnn.to_torch(t).cpu()
ttnn_t.numpy()              # WRONG  -> ttnn.to_torch(t).numpy()
ttnn_t.detach()             # WRONG  -> ttnn tensors are already detached
ttnn_t.contiguous()         # WRONG  -> ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
ttnn_t + 0.5                # WRONG  -> ttnn.add(t, 0.5)
ttnn_t * 2.0                # WRONG  -> ttnn.mul(t, 2.0)
ttnn_t / 2.0                # WRONG  -> ttnn.div(t, 2.0)
ttnn_t - other              # WRONG  -> ttnn.sub(t, other)
ttnn_t @ other              # WRONG  -> ttnn.matmul(t, other)
torch.cat([ttnn_t, ...])    # WRONG  -> ttnn.concat([t, ...], dim=...)
torch.matmul(ttnn_t, ...)   # WRONG  -> ttnn.matmul(t, ...)
torch.nn.functional.relu(ttnn_t)  # WRONG  -> ttnn.relu(t)
```

## Project conventions (mandatory)

```
Project conventions for synthesized TTNN modules:

1. File MUST start with this exact SPDX header (no modifications):
       # SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
       #
       # SPDX-License-Identifier: Apache-2.0

2. Imports: only `import ttnn`, `import torch` (for weight prep only), and
   standard library. Do NOT import transformers in the synthesized body —
   weight loading happens via the torch_module passed into __init__.

3. Public surface MUST export TWO names that downstream code depends on:
     class <ComponentClassName>:
         def __init__(self, device, torch_module):  ...
         def __call__(self, x): ...            # may take *args, **kwargs
     def build(device, torch_module):
         return <ComponentClassName>(device, torch_module)
   And the original module-level function name MUST still exist as a thin
   wrapper that lazily constructs an instance using a default device. Use
   this template at the end of the file:
       _instance = None
       def <component_safe>(*args, **kwargs):
           global _instance
           if _instance is None:
               raise RuntimeError(
                   "Synthesized TTNN module requires `build(device, torch_module)`. "
                   "Call it from the PCC test's `_build_ttnn_port`."
               )
           return _instance(*args, **kwargs)

4. Weight loading: inside __init__, walk torch_module.state_dict() and
   convert each tensor you need via:
       self.w_<name> = ttnn.from_torch(state_dict["<key>"], dtype=ttnn.bfloat16,
                                       layout=ttnn.TILE_LAYOUT, device=device)
   Cache shapes/scales as plain Python attributes when possible.

5. Forward: every intermediate is a ttnn.Tensor. Do NOT call .cpu(), .numpy(),
   or .item(). Do NOT roundtrip through torch in the hot path.

6. If a feature in the HF reference cannot be expressed with the cheatsheet
   ops, raise NotImplementedError("ttnn-gap: <what's missing>") in the body
   so the bring-up loop surfaces it as a real PCC failure, and approximate
   with the closest available op for the rest of the file. Do NOT emit a
   comment for this — raise, don't narrate.

7. NO COMMENTS in the body. Specifically:
       - Do NOT add docstrings for classes, methods, or modules.
       - Do NOT add inline `# ...` comments that narrate what the code does.
       - Do NOT add section banners or `# ===`-style separators.
   The only comments allowed in the file are the 3-line SPDX header from
   rule 1. The code must be self-explanatory; if a step needs explaining,
   pick a clearer variable / method name instead of a comment. If the
   bring-up loop wants context, it reads the prompt — not the source.

8. Output FORMAT: use the Write tool to write the file to the exact
   `WRITE THIS FILE:` path shown in the prompt header. Do NOT paste the
   file contents into chat; do NOT respond with the Python source as
   chat text. The bring-up loop reads files from disk (under the demo's
   `_synth_responses/` directory), not from your chat output. After you
   call Write, your chat message can be just a 1-line confirmation like
   "Wrote `_synth_responses/<component>.py`". No code fences, no prose.
```

## Output

Use the Write tool to write the complete file contents to the `WRITE THIS FILE:` path in the section header above. The content must start with the 3-line SPDX header and contain only Python source (no markdown fences). Do NOT paste the file into chat; the bring-up loop reads it from disk.

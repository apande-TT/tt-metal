# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The demo / e2e inputs. PROVENANCE: TOOL-AUTHORED.

No published usage example was discovered for mistralai/Devstral-Small-2-24B-Instruct-2512, so ONE content
input per task was authored -- the smallest ordinary instance of what this coding model is for -- and it is
reused for every batch row. Rows differ only by their sampling seed (the generation_config samples at
temperature 0.15), seed = row index.

  text_generation      TEXT_PROMPT, sent through the checkpoint's own chat template (default system prompt)
  image_text_to_text   a 224x224 white image with a two-line Python function rendered on it (load_image)
                       plus IMAGE_PROMPT
"""
from __future__ import annotations

TEXT_PROMPT = "Write a Python function that returns the sum of two numbers."
IMAGE_PROMPT = "What does the code in this image do? Answer in one sentence."
IMAGE_CODE = "def add(a, b):\n    return a + b"
IMAGE_SIZE = (224, 224)


def load_image(path=None):
    """The image input: `path` if given, else the tool-authored code snippet rendered deterministically."""
    from PIL import Image, ImageDraw, ImageFont

    if path:
        return Image.open(path).convert("RGB")
    img = Image.new("RGB", IMAGE_SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # Pillow < 10.1 has no sized default font
        font = ImageFont.load_default()
    draw.multiline_text((12, 80), IMAGE_CODE, fill=(0, 0, 0), font=font, spacing=8)
    return img

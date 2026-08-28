# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""How deep a capped repeated stack is allowed to be.

The `layers` knob makes a cheap build by holding fewer of a section's repeated
blocks.  There is a floor on that, and it is not 1.

The profiler discovers a model's sections by WALKING the built pipeline
(`find_all_stacks` in `models/experimental/perf_automation/cc_optimize/_op_sig_probe.py`)
and only recognises a list as a repeated stack once it holds **three** same-typed
blocks.  A section capped to one or two blocks is therefore not a shallow section --
it is an absent one: nothing downstream can size it, cap it, or attribute time to it,
so its depth gets inferred for the whole run and the section is profiled as though it
did not exist.  Two blocks also drop the second decomposed variant of a stack whose
first positions are the decomposed ones, which is a second way to make a cap silently
change the model rather than shorten it.

So a cap floors at three blocks -- unless the section genuinely has fewer, in which
case its real depth is the floor and nothing is invented.
"""

from __future__ import annotations

#: `find_all_stacks`'s own threshold. Not a preference: below it the walk yields nothing.
MIN_DISCOVERABLE_STACK = 3


def stack_depth(cap, available: int) -> int:
    """How many of `available` repeated blocks to hold for a requested `cap`.

    `None` means the whole section.  Anything else is clamped into
    ``[min(MIN_DISCOVERABLE_STACK, available), available]``.
    """
    total = int(available)
    if cap is None:
        return total
    floor = min(MIN_DISCOVERABLE_STACK, total)
    return max(floor, min(int(cap), total))

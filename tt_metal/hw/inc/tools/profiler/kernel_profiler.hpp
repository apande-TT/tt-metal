// TEMPORARY local shim (not upstream). tt-lang emits the pre-27cad27903 spelling:
// #include "tools/profiler/kernel_profiler.hpp". Upstream 27cad27903 (#60172) dropped
// root_ + "tt_metal" from the JIT include path list (tt_metal/jit_build/build.cpp),
// so the bare "tools/..." spelling no longer resolves, and the header itself now lives
// at hw/inc/api/debug/kernel_profiler.hpp.
// Delete this file once tt-lang emits the new spelling.
// Companion to the experimental/circular_buffer.h shim, which covers the same mismatch
// for the header a67eb83be3 moved.
#pragma once
#include "api/debug/kernel_profiler.hpp"

// TEMPORARY local shim (not upstream). tt-lang 1.1.1 emits the pre-a67eb83be3
// spelling: #include "experimental/circular_buffer.h" + experimental::CircularBuffer.
// Upstream a67eb83be3 renamed the header to hw/inc/api/dataflow/circular_buffer.h
// and moved the class out of the experimental namespace.
// Delete this file once on tt-lang >= 1.1.6, which emits the new spelling.
#pragma once
#include "api/dataflow/circular_buffer.h"

namespace experimental {
using ::CircularBuffer;
}

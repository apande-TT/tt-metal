# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Session fixtures: the mesh is opened ONCE here (the sole device opener) and the resident pipeline is built
ONCE through tt.pipeline.build_pipeline, then shared by both task heads' tests."""
from __future__ import annotations

import pytest

from models.demos.devstral_small_2_24b_instruct_2512.mesh import close_mesh, open_mesh
from models.demos.devstral_small_2_24b_instruct_2512.tt import pipeline as P


@pytest.fixture(scope="session")
def devstral_mesh():
    mesh = open_mesh(trace_region_size=P.TRACE_REGION_SIZE)
    yield mesh
    close_mesh(mesh)


@pytest.fixture(scope="session")
def hf_model():
    return P.load_hf_model()


@pytest.fixture(scope="session")
def processor():
    return P.load_processor()


@pytest.fixture(scope="session")
def pipe(devstral_mesh, hf_model):
    # Batch comes from $TT_PERF_BATCH (the gate sets 32); build_pipeline reads it when batch is not passed.
    pipeline = P.build_pipeline(devstral_mesh, model=hf_model)
    print(f"PERF_BATCH_STREAMS={pipeline.batch}", flush=True)
    return pipeline

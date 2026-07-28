# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3_1_8b_p150.tt.ccl import tt_distributed_rmsnorm, tt_sharded_distributed_rmsnorm
from models.demos.llama3_1_8b_p150.tt.common import Mode


class DistributedNorm(LightweightModule):
    def __init__(self, norm, args, tt_ccl, prefetcher=None, TG=False, ag_config_key=None, enable_all_gather=True):
        self.norm = norm
        self.args = args
        self.tt_ccl = tt_ccl
        self.prefetcher = prefetcher
        self.ag_config_key = ag_config_key

        # Flag to control whether all_gather is performed after distributed norm (can be disabled when output should remain sharded)
        self.enable_all_gather = enable_all_gather

        if TG:
            core_grid_ln = (
                min(4, args.dim // 4 // 32 // 8),
                8,
            )  # dividing by 4 and 8 for num_cols and num_rows of mesh, and 32 for tile size
            num_cores_ln = core_grid_ln[0] * core_grid_ln[1]
            hidden_size_per_device_distributed_ln = args.dim // 4
            self.gather_in_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(1, 1, 32, hidden_size_per_device_distributed_ln),
                core_grid=ttnn.CoreGrid(y=core_grid_ln[0], x=core_grid_ln[1]),
                strategy=ttnn.ShardStrategy.WIDTH,
            )
            self.ln_prg_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(core_grid_ln[1], core_grid_ln[0]),
                subblock_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
                block_h=1,
                block_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
                inplace=False,
            )
            self.ln_sharded_stats_memcfg = ttnn.create_sharded_memory_config(
                shape=[1, 1, 32, 32 * 4],
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.WIDTH,
            )
            self.ln_cfg = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
        self.TG = TG

    def forward(self, x, mode: Mode, norm_config=None):
        """Apply a norm, possibly gathering inputs if required."""

        sharded_output_config = norm_config.get("sharded_output_config") if norm_config else None

        if self.TG:
            if mode == Mode.DECODE:
                return tt_sharded_distributed_rmsnorm(
                    x,
                    epsilon=self.norm.eps,
                    gamma=self.norm.weight_distributed,
                    mesh_device=self.args.mesh_device,
                    tt_ccl=self.tt_ccl,
                    ln_sharded_input_memcfg=self.gather_in_mem_cfg,
                    ln_sharded_progcfg=self.ln_prg_cfg,
                    ln_sharded_stats_memcfg=self.ln_sharded_stats_memcfg,
                )
            else:
                return tt_distributed_rmsnorm(
                    x,
                    epsilon=self.norm.eps,
                    gamma=self.norm.weight_distributed,
                    mesh_device=self.args.mesh_device,
                    tt_ccl=self.tt_ccl,
                    compute_kernel_config=self.ln_cfg,
                )

        input_mem_cfg = sharded_output_config if mode == Mode.DECODE else ttnn.DRAM_MEMORY_CONFIG

        # Distributed norm already performs a gather
        if self.args.is_multichip and not self.args.is_distributed_norm(mode):
            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=self.args.model_config[self.ag_config_key]["num_links"]
                if self.ag_config_key and mode == "decode"
                else self.tt_ccl.get_num_links(1),
                topology=self.args.ccl_topology(),
                memory_config=input_mem_cfg,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=self.args.model_config[self.ag_config_key]["chunks_per_sync"]
                if self.ag_config_key and mode == "decode"
                else 10,
                num_workers_per_link=self.args.model_config[self.ag_config_key]["num_workers_per_link"]
                if self.ag_config_key and mode == "decode"
                else 2,
                num_buffers_per_channel=2,
                subdevice_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
            )
        else:
            # PREFILL, single chip: the residual stream may already be in L1 (short prefill), and
            # forcing it to DRAM here costs a copy IN plus a copy back OUT for every norm -- measured
            # as 704 CopyDeviceOperation calls / 3.03 ms, which is most of what the L1 residual won.
            # The interleaved rms_norm kernel reads an interleaved input whatever its buffer type, so
            # leave the tensor where it already is.
            if mode == Mode.PREFILL and not self.args.is_multichip:
                input_mem_cfg = x.memory_config()
            x = ttnn.to_memory_config(x, input_mem_cfg)

        # GRID rung for LayerNormDeviceOperation. Prefill took the INTERLEAVED rms_norm kernel, which
        # the profiler measured on ONE core at 62.5 us/call -- 757 such calls, 47.3 ms, i.e. ~90% of
        # the whole norm bucket, against a 0.85 ms roofline for all 1679 norms. The interleaved kernel
        # parallelises over ROWS, and a [32, dim] prefill is a single tile-row, so it can only ever get
        # one core; the SHARDED kernel parallelises over WIDTH and does the same shape in 6.1-6.5 us on
        # 32-64 cores. create_sharded_norm_config's block_h is tile_padded_batch_rows // 32, so the
        # existing config is valid exactly when the prefill is that many rows -- which a short
        # (32-token) padded prefill now is. So shard the input, run the sharded kernel, and hand the
        # result back in the layout the prefill graph expects. The two reshards are on a [32, dim]
        # tensor (~256 KB) against ~56 us saved per norm.
        _prefill_sharded_norm = (
            mode == Mode.PREFILL
            and not self.TG
            and not self.args.is_multichip
            and not self.args.is_distributed_norm(mode)
            and norm_config is not None
            and norm_config.get("sharded_program_config") is not None
            and norm_config.get("sharded_output_config") is not None
            and int(x.shape[-2]) == self.args.tile_padded_batch_rows
            and int(x.shape[-4]) == 1
            and int(x.shape[-3]) == 1
        )
        if _prefill_sharded_norm:
            _restore_mem_cfg = x.memory_config()
            x_sharded_in = ttnn.to_memory_config(x, norm_config["sharded_output_config"])
            y = self.norm(x_sharded_in, mode=mode, in_sharded=True, out_sharded=True, norm_config=norm_config)
            ttnn.deallocate(x_sharded_in)
            out = ttnn.to_memory_config(y, _restore_mem_cfg)
            # MUST free the sharded intermediate. Leaving it alive leaks a [32, dim] L1 buffer per norm
            # call -- 2 norms x 32 layers -- which exhausts L1 and pushes every later allocation to
            # DRAM: measured as a 45% GLOBAL slowdown (ff1/ff3 99.5 -> 146.4 us/call, ff2 126.6 ->
            # 212.9) even though the norm itself got 3.2x faster. The norm win is only real with this.
            if out is not y:
                ttnn.deallocate(y)
            return out

        x = self.norm(
            x, mode=mode, in_sharded=(mode == Mode.DECODE), out_sharded=(mode == Mode.DECODE), norm_config=norm_config
        )

        # Distributed norm requires a gather
        if self.args.is_distributed_norm(mode) and self.enable_all_gather:
            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=self.tt_ccl.get_num_links(1),
                topology=self.args.ccl_topology(),
                memory_config=x.memory_config(),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

        return x

"""2026-11-20 r1zero patch: in verl megatron_workers.rollout_mode, log the training process's GPU memory (allocated / reserved /
inactive split) right after training and after the weight sync, and call torch.cuda.ipc_collect() before SGLang re-allocates its KV cache.
Purpose: find out why the training process keeps ~16.5 GiB during generation from step 2 on (15:39 OOM). Original saved as .orig_rollmem."""
import ast, shutil, sys
P = '/root/miniconda3/lib/python3.12/site-packages/verl/workers/megatron_workers.py'
s = open(P).read()
if '_r1z_log_mem' in s:
    print('already patched'); sys.exit(0)
helper = '''def _r1z_log_mem(tag):  # r1zero patch (11/20): where does the training process keep ~16.5 GiB during generation?
    try:
        import torch.distributed as _dist
        _rank = _dist.get_rank() if _dist.is_initialized() else -1
        _st = torch.cuda.memory_stats()
        print("[r1z mem] rank %d %s: allocated %.2f GiB, reserved %.2f GiB, inactive_split %.2f GiB, active_blocks %d"
              % (_rank, tag, torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30,
                 _st.get("inactive_split_bytes.all.current", 0) / 2**30, _st.get("active.all.current", 0)), flush=True)
    except Exception as _e:  # noqa: BLE001
        print("[r1z mem] %s failed: %s" % (tag, _e), flush=True)


'''
a0 = '\nclass ActorRolloutRefWorker'
assert s.count(a0) == 1, 'class anchor'
s = s.replace(a0, '\n' + helper + a0.lstrip('\n'), 1)
a1 = '        """Context switch hybridengine to rollout mode."""\n        aggressive_empty_cache(force_sync=True)\n'
assert s.count(a1) == 1, 'rollout_mode start anchor'
s = s.replace(a1, a1 + '        _r1z_log_mem("rollout_mode start (after training, before loading weights)")\n')
a2 = ('            offload_megatron_model_to_cpu(self.actor.actor_module)\n        aggressive_empty_cache(force_sync=True)\n'
      '        if self.config.rollout.free_cache_engine:\n            await self.rollout.resume(tags=["kv_cache"])\n')
assert s.count(a2) == 1, 'kv resume anchor'
b2 = ('            offload_megatron_model_to_cpu(self.actor.actor_module)\n        aggressive_empty_cache(force_sync=True)\n'
      '        try:  # r1zero patch (11/20): free CUDA-IPC blocks SGLang has already consumed\n'
      '            torch.cuda.ipc_collect()\n'
      '            torch.cuda.empty_cache()\n'
      '        except Exception as _e:  # noqa: BLE001\n'
      '            print("[r1z mem] ipc_collect failed: %s" % _e, flush=True)\n'
      '        _r1z_log_mem("rollout_mode before kv resume (after weight sync, offload and ipc_collect)")\n'
      '        if self.config.rollout.free_cache_engine:\n            await self.rollout.resume(tags=["kv_cache"])\n')
s = s.replace(a2, b2)
ast.parse(s)
shutil.copy(P, P + '.orig_rollmem')
open(P, 'w').write(s)
print('patched', P)

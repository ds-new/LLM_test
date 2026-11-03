"""模拟 `python -m verl.trainer.main_ppo`（只用于干跑与测试，不做任何计算）。

行为尽量贴近真实 verl，用来检验多智能体 loop 的监控与决策逻辑：
  * `--cfg job` 打印配置树（含 loop 会用到的全部键）；
  * Ray 风格日志前缀与 np.float64(...) 数值；val_before_train 的第 0 步验证；
  * 按 save_freq 保存 global_step_N/actor/（模型、优化器、extra_state 分片 + huggingface/），
    写 latest_checkpointed_iteration.txt，按 max_actor_ckpt_to_keep 轮转删除；resume_mode=auto 断点续训；
  * rollout.gpu_memory_utilization 高于 FAKE_OOM_IF_GMU_ABOVE 时第 2 步报 CUDA OOM（检验降级阶梯）；
  * 验证准确率曲线随预设不同：grpo 先升后降且熵塌缩，dapo_lite 稳步上升。
"""

import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

KEYS = """algorithm.adv_estimator algorithm.use_kl_in_reward algorithm.norm_adv_by_std_in_grpo algorithm.gamma algorithm.lam
data.train_files data.val_files data.train_batch_size data.max_prompt_length data.max_response_length
data.filter_overlong_prompts data.truncation data.image_key data.shuffle data.seed data.prompt_key
actor_rollout_ref.model.path actor_rollout_ref.model.override_config actor_rollout_ref.model.trust_remote_code actor_rollout_ref.model.use_remove_padding
actor_rollout_ref.model.enable_gradient_checkpointing actor_rollout_ref.actor.optim.lr actor_rollout_ref.actor.ppo_epochs
actor_rollout_ref.actor.ppo_mini_batch_size actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
actor_rollout_ref.actor.use_dynamic_bsz actor_rollout_ref.actor.ppo_max_token_len_per_gpu actor_rollout_ref.actor.use_kl_loss
actor_rollout_ref.actor.kl_loss_coef actor_rollout_ref.actor.kl_loss_type actor_rollout_ref.actor.entropy_coeff
actor_rollout_ref.actor.clip_ratio actor_rollout_ref.actor.clip_ratio_low actor_rollout_ref.actor.clip_ratio_high
actor_rollout_ref.actor.loss_agg_mode actor_rollout_ref.actor.fsdp_config.param_offload
actor_rollout_ref.actor.fsdp_config.optimizer_offload actor_rollout_ref.actor.checkpoint.save_contents
actor_rollout_ref.rollout.name actor_rollout_ref.rollout.tensor_model_parallel_size
actor_rollout_ref.rollout.gpu_memory_utilization actor_rollout_ref.rollout.n actor_rollout_ref.rollout.free_cache_engine
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu
actor_rollout_ref.rollout.val_kwargs.do_sample actor_rollout_ref.rollout.val_kwargs.n actor_rollout_ref.rollout.temperature
actor_rollout_ref.ref.log_prob_use_dynamic_bsz actor_rollout_ref.ref.log_prob_max_token_len_per_gpu
actor_rollout_ref.ref.fsdp_config.param_offload custom_reward_function.path custom_reward_function.name
trainer.critic_warmup trainer.logger trainer.project_name trainer.experiment_name trainer.n_gpus_per_node trainer.nnodes
trainer.save_freq trainer.test_freq trainer.total_epochs trainer.total_training_steps trainer.default_local_dir
trainer.resume_mode trainer.val_before_train trainer.max_actor_ckpt_to_keep critic.optim.lr reward_model.enable""".split()


def print_cfg():
    tree = {}
    for k in KEYS:
        node = tree
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = None
    import yaml
    print("[fake] importing ray ...")  # 模拟 import 时的杂散输出
    print(yaml.safe_dump(tree, sort_keys=False))


def curve(preset, frac):
    if preset.startswith("dapo"):
        return 0.35 + 0.16 * (1 - math.exp(-4 * frac)), 1.2 - 0.3 * frac
    if preset.startswith("drgrpo"):
        return 0.35 + 0.12 * (1 - math.exp(-3 * frac)), 1.1 - 0.4 * frac
    # grpo：前 30% 上升，之后退化；熵快速塌缩
    acc = 0.35 + 0.10 * min(1.0, frac / 0.3) - (0.25 * (frac - 0.3) if frac > 0.3 else 0)
    return acc, 1.2 * math.exp(-5 * frac) + 0.02


def main():
    argv = sys.argv[1:]
    if "--cfg" in argv:
        print_cfg()
        return 0
    if os.environ.get("FAKE_IMPORT_ERROR"):
        print("(TaskRunner pid=1) ImportError: cannot import name 'flash_attn_func' from 'flash_attn'", flush=True)
        return 1
    o = {}
    for a in argv:
        k, _, v = a.lstrip("+").partition("=")
        o[k] = v
    save = Path(o["trainer.default_local_dir"])
    save.mkdir(parents=True, exist_ok=True)
    tf = Path(o["data.train_files"])
    rows = sum(1 for l in tf.read_text().splitlines() if l.strip()) if tf.suffix == ".jsonl" else 64
    bsz = int(o["data.train_batch_size"])
    total = int(o["trainer.total_epochs"]) * max(1, math.ceil(rows / bsz))
    save_freq, test_freq = int(o["trainer.save_freq"]), int(o["trainer.test_freq"])
    keep = int(o.get("trainer.max_actor_ckpt_to_keep", "0") or 0)
    preset = o.get("trainer.experiment_name", "x_grpo").split("_", 1)[-1]
    gmu = float(o["actor_rollout_ref.rollout.gpu_memory_utilization"])
    oom_above = float(os.environ.get("FAKE_OOM_IF_GMU_ABOVE", "9"))
    latest = save / "latest_checkpointed_iteration.txt"
    start = int(latest.read_text()) if latest.exists() else 0
    pid = os.getpid()
    print(f"[fake verl] preset={preset} rows={rows} total_steps={total} resume_from={start}", flush=True)

    def val_metrics(step):
        acc, _ = curve(preset, step / total)
        return {"val-core/hiyouga/geometry3k/acc/mean@1": round(acc, 4),
                "val-core/hiyouga/geometry3k/reward/mean@1": round(acc + 0.09, 4)}

    if start == 0 and o.get("trainer.val_before_train", "True") == "True":
        vm = val_metrics(0)
        print(f"(TaskRunner pid={pid}) step:0 - " + " - ".join(f"{k}:np.float64({v})" for k, v in vm.items()), flush=True)
    for step in range(start + 1, total + 1):
        time.sleep(float(os.environ.get("FAKE_STEP_SECONDS", "0.03")))
        if gmu > oom_above and step >= start + 2:
            print("Traceback (most recent call last):\n  ...\ntorch.OutOfMemoryError: CUDA out of memory. "
                  "Tried to allocate 2.00 GiB", flush=True)
            return 1
        frac = step / total
        acc, ent = curve(preset, frac)
        m = {"critic/score/mean": round(acc + 0.05, 4), "actor/entropy": round(ent, 4),
             "response_length/mean": 420.0, "response_length/clip_ratio": 0.02, "actor/pg_clipfrac": 0.01,
             "actor/grad_norm": 0.3, "timing_s/step": 61.5}
        if step % test_freq == 0 or step == total:
            m.update(val_metrics(step))
        if step % save_freq == 0 or step == total:
            d = save / f"global_step_{step}" / "actor"
            (d / "huggingface").mkdir(parents=True, exist_ok=True)
            for name in ("model_world_size_1_rank_0.pt", "optim_world_size_1_rank_0.pt",
                         "extra_state_world_size_1_rank_0.pt"):
                (d / name).write_text(name)
            (d / "huggingface" / "config.json").write_text("{}")
            (d / "huggingface" / "fake_quality.json").write_text(json.dumps({"acc": curve(preset, frac)[0]}))
            latest.write_text(str(step))
            if keep > 0:
                ckpts = sorted(save.glob("global_step_*"), key=lambda p: int(p.name.split("_")[-1]))
                for old in ckpts[:-keep]:
                    shutil.rmtree(old)
        line = " - ".join(f"{k}:np.float64({v})" if i % 2 else f"{k}:{v}" for i, (k, v) in enumerate(m.items()))
        print(f"(TaskRunner pid={pid}) step:{step} - {line}", flush=True)
    print("[fake verl] training finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

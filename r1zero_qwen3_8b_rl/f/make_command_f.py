"""从 D 的训练命令（run/command_r1zero_qwen3_8b_dapo.sh）生成 F 的命令：只改下面列出的参数，其余沿用第三轮验证过的设置
（GRPO + clip-higher 0.2/0.28 + 双裁剪 + token 级平均 + 动态采样，lr 1e-6，每批更新 8 次，Megatron TP2 + SGLang TP1）。
用法：python f/make_command_f.py smoke|stage1 MODEL_HF [显存档位 0/1/2] -> 写 run/command_f_<mode>.sh"""
import os
import re
import sys
import time

R = '/root/autodl-tmp/r1zero'
mode, model = sys.argv[1], sys.argv[2]
tier = int(sys.argv[3]) if len(sys.argv) > 3 else 0
tier = tier + int(os.environ.get('F_MIN_TIER', '0'))  # 11/21 01:4x Claude: was max(); with F_MIN_TIER=1 the first OOM retry (raw tier 1) stayed at tier 1, so escalation was a no-op; now raw tier is an offset on top of F_MIN_TIER  # 11/20 16:0x：0 档第 2 步会 OOM（训练进程生成时还占 16.5 GiB），resume_f5 设 F_MIN_TIER=1
s = open(R + '/run/command_r1zero_qwen3_8b_dapo.sh').read()

OV = {
    'data.train_files': R + '/run/data/train_f.parquet',
    'data.val_files': R + '/run/data/val_select_f.parquet',
    'data.max_response_length': '16384',
    'actor_rollout_ref.model.path': model,
    'actor_rollout_ref.actor.ppo_max_token_len_per_gpu': '18432',
    'actor_rollout_ref.rollout.n': '16',
    'data.gen_batch_size': '48',  # 11/20 13:4x（用户同意）：生成批 48 > 训练批 32，被过滤掉几组也不用再生成一整批
    '+actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size': '2048',  # 11/20 16:0x：15:39 的 OOM 是一次 4.64 GiB 分配（8192 token × 词表 × fp32 logits），分块预填改小
    'custom_reward_function.path': R + '/ma/reward_f.py',
    'reward_model.overlong_buffer.len': '4096',
    'trainer.logger': '[console,file]',
    'trainer.project_name': 'r1zero_f',
    'trainer.experiment_name': 'f_sftrl',
    'trainer.total_training_steps': '20',
    'trainer.save_freq': '2',  # 11/20 16:0x：第 2 步 OOM 时还没有存盘，第 1 步整步作废；改成每 2 步存一次
    'trainer.test_freq': '10',
    'trainer.val_before_train': 'False',
    'trainer.default_local_dir': R + '/run/ckpt/f_sftrl',
    'trainer.max_actor_ckpt_to_keep': '1',
    'trainer.validation_data_dir': '/root/r1zero_valgen/f',
    'trainer.resume_mode': 'auto',
}
if mode == 'smoke':  # 1 步、4 题 x 16 条、存一次盘：测 16K 长度下的显存、判分、HF 存盘
    OV.update({'data.train_files': R + '/run/data/f_smoke.parquet', 'data.train_batch_size': '4', 'data.gen_batch_size': '4',
               'actor_rollout_ref.actor.ppo_mini_batch_size': '4', 'algorithm.filter_groups.enable': 'False',
               'trainer.experiment_name': 'f_smoke', 'trainer.total_training_steps': '1', 'trainer.save_freq': '1',
               'trainer.test_freq': '-1', 'trainer.default_local_dir': R + '/run/ckpt/f_smoke',
               'actor_rollout_ref.actor.optim.lr_warmup_steps': '0'})  # 11/20: warmup 3 >= total 1 trips Megatron's assert
if tier >= 1:  # 显存不足时降档：推理引擎少占一些、logprob 批小一些
    OV.update({'actor_rollout_ref.rollout.gpu_memory_utilization': '0.55', 'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu': '18432'})
if tier >= 2:
    OV.update({'actor_rollout_ref.rollout.gpu_memory_utilization': '0.5', 'actor_rollout_ref.actor.ppo_max_token_len_per_gpu': '17408',
               'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu': '17408'})

raw_tier = int(sys.argv[3]) if len(sys.argv) > 3 else 0
if raw_tier == 0 and os.environ.get('F_GMU'):  # 11/20 17:14 用户同意：第 2 步实测余量够就把 SGLang 显存比例提到 0.60；OOM 升档（raw tier >= 1）后不再用，回到档位默认值
    OV['actor_rollout_ref.rollout.gpu_memory_utilization'] = os.environ['F_GMU']

if os.environ.get('F_GEN_BATCH'):  # 11/20 17:2x：第 1 步 48 题只有 29 组可用、要再生成一整批；从第 3 步起生成批 64（训练批仍 32 题 x 16 条）
    OV['data.gen_batch_size'] = os.environ['F_GEN_BATCH']
# 11/21 16:4x（用户同意）：存盘里没有学习率调度器状态，续训时 Megatron 调度器从 num_steps=0 重新预热，lr_warmup_init=0 时
# 续训后的第一步学习率是 0（只更新 Adam 动量、不改权重，白跑一步）。已经有存盘（resume_mode=auto 会从它续训）时改成：
# 第一步 lr/3、第二步起满学习率（lr_warmup_steps=1、lr_warmup_init=lr/3）。F_RESUME_LR=0 可以关掉。只影响以后新生成的命令。
if mode == 'stage1' and os.environ.get('F_RESUME_LR', '1') != '0':
    try:
        _it = int(open(OV['trainer.default_local_dir'] + '/latest_checkpointed_iteration.txt').read().strip())
    except Exception:  # noqa: BLE001
        _it = 0
    if _it >= 1:
        _m = re.search(r'actor_rollout_ref\.actor\.optim\.lr=(\S+)', s)
        _lr = float(_m.group(1)) if _m else 1e-6
        OV['actor_rollout_ref.actor.optim.lr_warmup_steps'] = '1'
        OV['actor_rollout_ref.actor.optim.lr_warmup_init'] = '%.6g' % (_lr / 3)
        print('resume from checkpoint %d: lr warmup 1 step from %.6g to %.6g' % (_it, _lr / 3, _lr))
for k, v in OV.items():
    pat = re.compile(r'(?<=\s)(\+{0,2})' + re.escape(k) + r'=\S*')
    n = len(pat.findall(s))
    if n == 1:
        s = pat.sub(lambda m: m.group(1) + k + '=' + v, s)
    elif n == 0:
        body = s.rstrip()
        if body.endswith('\\'):
            body = body[:-1].rstrip()
        s = body + ' \\\n  ' + k + '=' + v + '\n'
    else:
        raise SystemExit('ambiguous key %s (%d)' % (k, n))
metrics = R + '/run/metrics_f_%s_%s.jsonl' % (mode, time.strftime('%m%d_%H%M'))
# 不加 expandable_segments：verl 的 SGLang 用 torch_memory_saver 释放显存，和它不兼容
assert s.count('env \\\n  PYTHONPATH=') == 1
s = s.replace('env \\\n  PYTHONPATH=', 'env \\\n  VERL_FILE_LOGGER_PATH=' + metrics + ' \\\n  PYTHONPATH=')
s = '# 由 f/make_command_f.py 生成（%s，mode=%s，显存档位 %d）\n' % (time.strftime('%F %T'), mode, tier) + s
open(R + '/run/command_f_%s.sh' % mode, 'w').write(s)
print('written run/command_f_%s.sh; metrics %s' % (mode, metrics))

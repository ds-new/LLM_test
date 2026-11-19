"""模型框架智能体：verl（Megatron 训练 + SGLang 推理）的配置——显存档位、checkpoint 策略、奖励管理器，启动前用 Hydra 预校验。

第二轮（formal2）相对第一轮的改动：
  * 每 2 步存盘、每 4 步验证（原来每 25 步验证、每 10 步存盘；2026-11-15 应用户要求验证由每 2 步改为每 4 步以省时间，
    只影响挑最佳的粒度，不影响训练）；验证用评选集 val_select（400 条），用来挑最佳 checkpoint；
  * verl 不再自己删 checkpoint（max_actor_ckpt_to_keep=null）：verl 的删除发生在写新 checkpoint 之前，
    中途崩溃会两头落空；改由测试评估智能体在新 checkpoint 写完后再删旧的，并把最佳的复制到系统盘；
  * 奖励改用 DAPO 奖励管理器 + 超长软惩罚：回答超过 8192-2048=6144 token 后线性扣分，到 8192 扣满 1 分；
    判对判错仍用 ma/reward.py，验证集的 acc 指标不受惩罚影响；
  * 只存模型权重（数据盘 50 GB 放不下优化器状态）：续训时 Adam 状态和学习率调度器从头开始，所以续训用 3 步预热。

第三轮备选（第二轮结束后由 loop3 使用，第二轮不受影响）：
  * formal2_dyn：在 formal2 基础上加 DAPO 动态采样——组内 16 条全对或全错（acc 方差为 0）的题不参与更新，
    不够 64 题就再生成一批补齐（最多 6 批）。verl 0.6.1 的 main_ppo 没实现这一步（filter_groups 只是配置项），
    所以改用官方 recipe/dapo 的训练器（third_party/verl_v061，GitHub v0.6.1 原样，只改了 Hydra 搜索路径）。
    每批仍生成 64 题（gen_batch_size=train_batch_size），保证 data.pt 里的数据读取进度能接着用。
  * stage16k：上下文 8K → 16K（DeepScaleR 式分阶段加长）+ 动态采样，超长惩罚缓冲区 4096
    （超过 12288 token 开始扣分），从第二轮的最佳 checkpoint 出发，单独的实验名和 checkpoint 目录；
    显存档位另用 LEVELS_16K（单条序列最长 1024+16384，ppo_max_token_len_per_gpu 必须 ≥ 17408）。
"""
import subprocess

from .common import CODE, DATA, ENV, LOGS, MODEL, ROOT, RUN

NAME = "framework"
EXP = "r1zero_qwen3_8b_dapo"
EXP_16K = EXP + "_16k"
CKPT_DIR = RUN / "ckpt" / EXP
CKPT_DIR_16K = RUN / "ckpt" / EXP_16K
RECIPE_ROOT = ROOT / "third_party" / "verl_v061"
VALGEN_DIR = "/root/r1zero_valgen"  # 验证样本落盘（系统盘），供结果分析智能体检查截断/重复
OVERLONG = {"len": 2048, "penalty_factor": 1.0}
DYN = {"metric": "acc", "max_num_gen_batches": 6}
# 显存档位：OOM 后逐级降低。gmu = SGLang 占整卡显存的比例（权重 + KV 缓存）；lp_tok = 重算 logprob 时每卡每批 token 上限。
# 2026-11-15 提速：1 档 gmu 0.45 → 0.70、lp_tok 10240 → 24576。依据：生成时 KV 缓存使用率 0.98–1.00 且有 50–75 个请求排队，
# 而训练阶段 SGLang 会释放显存、生成阶段 Megatron 全部卸载到 CPU，生成时卡上大量显存闲置。只改执行效率，不改算法和超参。
# 原 1 档挪到 2 档作退路：新配置一旦显存不足，决策智能体降一档即回到原配置。
# 2026-11-16 提速第二步：生成改为 4 个单卡引擎（rtp=1，训练仍 TP=2）。实测双卡引擎 100 并发时每个解码步约 110 毫秒、
# 单条只有 45–126 token/秒——本机禁用 GPU 直连，TP=2 每个 token 要经主机内存同步 72 次；单卡引擎没有这部分通信。
LEVELS = [
    {"tp": 2, "recompute": "selective", "max_tok": 12288, "gmu": 0.50},
    {"tp": 2, "recompute": "full", "max_tok": 10240, "gmu": 0.70, "lp_tok": 24576, "rtp": 1},
    # 2026-11-18：C 第 136 步生成时 SGLang 算 logprob 的临时张量 OOM（还差 0.03 GiB）。新增这一档：只把显存比例 0.70 降到 0.65，其余同上
    {"tp": 2, "recompute": "full", "max_tok": 10240, "gmu": 0.65, "lp_tok": 24576, "rtp": 1},
    {"tp": 2, "recompute": "full", "max_tok": 10240, "gmu": 0.70, "lp_tok": 24576},
    {"tp": 2, "recompute": "full", "max_tok": 10240, "gmu": 0.45},
    {"tp": 4, "recompute": "full", "max_tok": 8192, "gmu": 0.40},
]
# 16K 上下文的档位（未实测，第一次启动时从 0 档开始，OOM 再降）
LEVELS_16K = [
    {"tp": 2, "recompute": "full", "max_tok": 18432, "gmu": 0.45},
    {"tp": 4, "recompute": "full", "max_tok": 18432, "gmu": 0.40},
    {"tp": 4, "recompute": "full", "max_tok": 17408, "gmu": 0.35},
]


def levels_for(kind):
    return LEVELS_16K if kind in ("stage16k", "eval16k") else LEVELS


def plan(kind, level=0, steps=None, exp=None, resume_path=None, start_from=None, extra=None):
    """kind: smoke | formal（第一轮）| formal2（第二轮）| formal2_dyn | stage16k（第三轮备选）| eval | eval16k
    steps：训练到第几步（global step，不是再训多少步）；start_from：stage16k 的起点 checkpoint（global_step_N 目录）。"""
    lv = levels_for(kind)[level]
    base = dict(reward="naive", overlong=False, overlong_len=OVERLONG["len"], ckpt_keep=1, val_only=False,
                resume_path=None, start_from=None, valgen=False, dyn=False, reward_file="reward.py", valgen_dir=None)
    # 第三轮起（审计后，run/audit/AUDIT.md）：判分用修正版 reward_v2（原版会把约 2.4% 的正确答案判错），
    # 训练集去掉与评测集重复的 6 道题。第二轮（formal2）保持原样，保证同一轮内奖励口径一致。
    round3 = dict(train=DATA / "train_dapo_r3.parquet", reward_file="reward_v2.py")
    # train_dapo_r3：行数和顺序与原训练集相同，只把 6 道泄露题换成别的题的副本——续训时 checkpoint 里存的
    # 数据加载器状态（data.pt）按原数据集长度记录位置，行数不变才不会错位（train_dapo_dedup 少 6 行，不用于续训）
    # 最终评测：完整验证集 + AIME 2026 留出集（从未参与挑选；已去掉训练集里有原题的第 1 题）
    aime26 = DATA / "test_aime26.parquet"
    eval_val = [DATA / "val.parquet", aime26] if aime26.exists() else DATA / "val.parquet"
    formal2 = dict(train=DATA / "train_dapo.parquet", val=DATA / "val_select.parquet", bsz=64, mini=32, n=16, resp=8192,
                   steps=steps or 150, val_before=True, test_freq=4, save_freq=2, exp=EXP, warmup=3,
                   reward="dapo", overlong=True, ckpt_keep=None, valgen=True)
    if kind == "smoke":
        p = dict(train=DATA / "train_xr750.parquet", val=DATA / "val_small.parquet", bsz=16, mini=8, n=4, resp=2048,
                 steps=steps or 3, val_before=False, test_freq=-1, save_freq=-1, exp="smoke", warmup=0)
    elif kind == "formal":
        p = dict(train=DATA / "train_dapo.parquet", val=DATA / "val.parquet", bsz=64, mini=32, n=16, resp=8192,
                 steps=steps or 150, val_before=True, test_freq=25, save_freq=10, exp=EXP, warmup=10)
    elif kind == "formal2":
        p = formal2
    elif kind == "formal2_dyn":
        p = {**formal2, **round3, "dyn": True}
    elif kind == "formal3a":  # 第三轮 A（2026-11-17）：只把每批数据的更新次数从 2 次提到 8 次（ppo_mini_batch_size 32 → 8）
        p = {**formal2, **round3, "mini": 8}
    elif kind == "formal3b":  # 第三轮 B：A + 动态采样
        p = {**formal2, **round3, "mini": 8, "dyn": True}
    elif kind == "formal3c":  # 第三轮 C（2026-11-17）：A 的配置 + 学习率 1e-6 → 2e-6；另外记录生成端概率（只监控，不修正，不影响训练）
        p = {**formal2, **round3, "mini": 8, "lr": 2e-6, "logprob_mon": True}
    elif kind == "formal3d":  # 第三轮 C'：B 的配置（含动态采样）+ 学习率 2e-6 + 生成端概率监控
        p = {**formal2, **round3, "mini": 8, "dyn": True, "lr": 2e-6, "logprob_mon": True}
    elif kind == "formal3e":
        # 第三轮 D（2026-11-18）：只训边界题（train_frontier.parquet，由 ma/build_frontier.py 按离线通过率选出）。
        # 每题 n=32、每步 32 题：每步回答总数仍是 1,024；mini=4：每批仍更新 8 次。学习率、是否动态采样由 loop3 按 B、C 的结论经 extra 传入
        p = {**formal2, **round3, "train": DATA / "train_frontier.parquet", "bsz": 32, "mini": 4, "n": 32, "logprob_mon": True}
    elif kind == "eval_profile":
        # 第三轮 D：离线测训练题通过率。profile_r3.parquet 里每题重复 8 行；温度 1.0、top_p 1.0，和训练时的采样一致
        p = dict(train=DATA / "train_dapo.parquet", val=DATA / "profile_r3.parquet", bsz=64, mini=32, n=16, resp=8192,
                 steps=steps or 150, val_before=True, test_freq=-1, save_freq=-1, exp=exp, warmup=0, reward="dapo", overlong=True,
                 ckpt_keep=None, val_only=True, resume_path=str(resume_path), valgen=True, valgen_dir=f"{VALGEN_DIR}/{exp}",
                 reward_file="reward_v2.py", val_temp=1.0, val_top_p=1.0)
    elif kind == "stage16k":
        if not start_from:
            raise ValueError("stage16k 需要 start_from（起点 checkpoint 目录）")
        p = {**formal2, **round3, "dyn": True, "resp": 16384, "overlong_len": 4096, "exp": EXP_16K, "steps": steps,
             "start_from": str(start_from)}
    elif kind in ("eval", "eval16k"):
        p = dict(train=DATA / "train_dapo.parquet", val=eval_val, bsz=64, mini=32, n=16, valgen=True,
                 valgen_dir=f"{VALGEN_DIR}/final_{exp}", reward_file="reward_v2.py",
                 resp=16384 if kind == "eval16k" else 8192, steps=steps or 150, val_before=True, test_freq=-1,
                 save_freq=-1, exp=exp, warmup=0, reward="dapo", overlong=True, ckpt_keep=None, val_only=True,
                 resume_path=str(resume_path), overlong_len=4096 if kind == "eval16k" else OVERLONG["len"])
    else:
        raise ValueError(kind)
    p = {**base, **p}
    p.update(extra or {})
    if not p["steps"]:
        raise ValueError(f"{kind} 需要 steps")
    p["ckpt_dir"] = RUN / "ckpt" / ("_eval" if kind.startswith("eval") else p["exp"])
    p.update(kind=kind, level=level, **lv)
    return p


def overrides(p):
    A, R, M = "actor_rollout_ref.actor", "actor_rollout_ref.rollout", "actor_rollout_ref.actor.megatron"
    keep = "null" if p["ckpt_keep"] is None else p["ckpt_keep"]
    o = [
        "algorithm.adv_estimator=grpo", "algorithm.use_kl_in_reward=False", "algorithm.norm_adv_by_std_in_grpo=True",
        f"data.train_files={p['train']}", "data.val_files=" + ("[" + ",".join(map(str, p["val"])) + "]" if isinstance(p["val"], list) else str(p["val"])), f"data.train_batch_size={p['bsz']}",
        "data.max_prompt_length=1024", f"data.max_response_length={p['resp']}", "data.filter_overlong_prompts=True",
        "data.truncation=error", "data.shuffle=True",
        f"actor_rollout_ref.model.path={MODEL}",
        f"{A}.optim.lr={p.get('lr', 1e-6)}", f"{A}.optim.lr_warmup_steps={p['warmup']}", f"{A}.optim.weight_decay=0.1",
        f"{A}.ppo_mini_batch_size={p['mini']}", f"{A}.use_dynamic_bsz=True", f"{A}.ppo_max_token_len_per_gpu={p['max_tok']}",
        f"{A}.clip_ratio_low=0.2", f"{A}.clip_ratio_high=0.28", f"{A}.clip_ratio_c=10.0", f"{A}.loss_agg_mode=token-mean",
        f"{A}.use_kl_loss=False", f"{A}.entropy_coeff=0", f"{A}.ppo_epochs=1",
        f"{A}.checkpoint.save_contents=[model]",
        f"{M}.tensor_model_parallel_size={p['tp']}", f"{M}.pipeline_model_parallel_size=1", f"{M}.use_mbridge=True",
        f"{M}.param_offload=True", f"{M}.grad_offload=True", f"{M}.optimizer_offload=True", f"{M}.sequence_parallel=True",
        f"{M}.override_transformer_config.attention_backend=auto",
        f"{R}.name=sglang", f"{R}.tensor_model_parallel_size={p.get('rtp', 2)}", f"{R}.gpu_memory_utilization={p['gmu']}",
        # 本机 GPU 直连(P2P)不可用：SGLang 自带的 custom all-reduce 走 P2P 会在解码时卡死，改用 NCCL
        f"+{R}.engine_kwargs.sglang.disable_custom_all_reduce=True",
        # 两个推理引擎生成长度不均时，先完成的一组会在集合通信上等另一组；默认 10 分钟超时会把训练杀掉（第一轮第 20 步实测）
        "actor_rollout_ref.nccl_timeout=3600",
        f"{R}.n={p['n']}", f"{R}.temperature=1.0", f"{R}.top_p=1.0", f"{R}.log_prob_use_dynamic_bsz=True",
        f"{R}.log_prob_max_token_len_per_gpu={p.get('lp_tok') or p['max_tok']}",
        f"{R}.val_kwargs.do_sample=True", f"{R}.val_kwargs.temperature={p.get('val_temp', 0.6)}", f"{R}.val_kwargs.top_p={p.get('val_top_p', 0.95)}", f"{R}.val_kwargs.n=1",
        f"custom_reward_function.path={CODE / p['reward_file']}", "custom_reward_function.name=compute_score",
        f"reward_model.reward_manager={p['reward']}",
        "trainer.n_gpus_per_node=4", "trainer.nnodes=1", "trainer.logger=[console]", "trainer.project_name=r1zero",
        f"trainer.experiment_name={p['exp']}", f"trainer.total_training_steps={p['steps']}", "trainer.total_epochs=10",
        f"trainer.save_freq={p['save_freq']}", f"trainer.test_freq={p['test_freq']}", f"trainer.val_before_train={p['val_before']}",
        f"trainer.default_local_dir={p['ckpt_dir']}", f"trainer.max_actor_ckpt_to_keep={keep}",
        "trainer.log_val_generations=0",
    ]
    if p["overlong"] and p["dyn"]:
        # recipe/dapo 的训练器从 reward_model.overlong_buffer 读超长惩罚（不读 reward_kwargs），max_resp_len 取 data.max_response_length
        o += ["reward_model.overlong_buffer.enable=True", f"reward_model.overlong_buffer.len={p['overlong_len']}",
              f"reward_model.overlong_buffer.penalty_factor={OVERLONG['penalty_factor']}", "reward_model.overlong_buffer.log=False"]
    elif p["overlong"]:
        o += ["+reward_model.reward_kwargs.overlong_buffer_cfg.enable=True",
              f"+reward_model.reward_kwargs.overlong_buffer_cfg.len={p['overlong_len']}",
              f"+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor={OVERLONG['penalty_factor']}",
              "+reward_model.reward_kwargs.overlong_buffer_cfg.log=False",
              f"+reward_model.reward_kwargs.max_resp_len={p['resp']}"]
    if p["dyn"]:
        # 按每条回答的 acc（0/1）在组内的方差过滤：全对/全错的题 advantage 为 0，不参与更新
        o += ["algorithm.filter_groups.enable=True", f"algorithm.filter_groups.metric={DYN['metric']}",
              f"algorithm.filter_groups.max_num_gen_batches={DYN['max_num_gen_batches']}",
              f"data.gen_batch_size={p['bsz']}"]
    if p["val_only"]:
        o += ["++trainer.val_only=True", "trainer.resume_mode=resume_path", f"trainer.resume_from_path={p['resume_path']}"]
    elif p["start_from"] and not (p["ckpt_dir"] / "latest_checkpointed_iteration.txt").exists():
        # 新阶段第一次启动：从指定 checkpoint 载入；之后中断续训时自己的目录里已有 checkpoint，走 auto
        o += ["trainer.resume_mode=resume_path", f"trainer.resume_from_path={p['start_from']}"]
    else:
        o += ["trainer.resume_mode=auto"]
    if p["valgen"]:
        o += [f"++trainer.validation_data_dir={p['valgen_dir'] or VALGEN_DIR}"]
    if p.get("logprob_mon"):
        # SGLang 把生成时的 token 概率一并返回；algorithm.rollout_correction 默认 rollout_is/rollout_rs 都是 null，
        # 只算 rollout_corr/ 指标（生成端和训练端概率的 KL、困惑度差、χ²），不加权、不丢样本
        o += ["actor_rollout_ref.rollout.calculate_log_probs=True"]
    if p["recompute"] == "full":
        o += [f"{M}.override_transformer_config.recompute_granularity=full",
              f"{M}.override_transformer_config.recompute_method=uniform",
              f"{M}.override_transformer_config.recompute_num_layers=1"]
    return o


def command(p):
    if p.get("dyn"):
        # 用官方 recipe/dapo 的训练器（含动态采样）；Ray 工作进程继承这个 PYTHONPATH 才能导入 recipe 包
        return ["env", f"PYTHONPATH={RECIPE_ROOT}:{ENV.get('PYTHONPATH', '')}", "python", "-m", "recipe.dapo.main_dapo",
                "--config-name=dapo_megatron_trainer", *overrides(p)]
    return ["python", "-m", "verl.trainer.main_ppo", "--config-name=ppo_megatron_trainer", *overrides(p)]


class FrameworkAgent:
    def __init__(self, board):
        self.board = board

    def validate(self, p):
        """只组装配置不运行（Hydra --cfg job），尽早发现写错的配置键。"""
        r = subprocess.run([*command(p), "--cfg", "job"], capture_output=True, text=True, env=ENV, timeout=600, cwd=str(RUN))
        ok = r.returncode == 0
        (LOGS / f"cfg_{p['exp']}.yaml").write_text(r.stdout + "\n" + r.stderr[-4000:], encoding="utf-8")
        self.board.post(NAME, "validate", {"exp": p["exp"], "ok": ok, **({} if ok else {"err": (r.stderr or r.stdout)[-800:]})})
        return ok, (r.stderr or r.stdout)[-1500:]

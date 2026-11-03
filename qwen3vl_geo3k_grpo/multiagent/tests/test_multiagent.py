"""单元测试：python -m pytest multiagent/tests -q（工程根目录，无需 GPU）。"""

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------- 奖励函数
T = "<think>r</think> "


@pytest.mark.parametrize("out,gt,exp", [
    (T + "\\boxed{2\\sqrt{221}}", "2 \\sqrt { 221 }", 1),
    (T + "\\boxed{\\sqrt{884}}", "2 \\sqrt { 221 }", 1),
    (T + "\\boxed{29.73}", "2 \\sqrt { 221 }", 1),
    (T + "\\boxed{29.7}", "2 \\sqrt { 221 }", 0),      # 只有 1 位小数不算
    (T + "\\boxed{30}", "2 \\sqrt { 221 }", 0),
    (T + "\\boxed{\\dfrac{26}{3}}", "\\frac { 26 } { 3 }", 1),
    (T + "\\boxed{8.67}", "\\frac { 26 } { 3 }", 1),
    (T + "\\boxed{120^{\\circ}}", "120", 1),
    (T + "\\boxed{120°}", "120", 1),
    (T + "\\boxed{m\\angle 1 = 30}", "30", 1),
    (T + "\\boxed{12\\pi}", "12 \\pi", 1),
    (T + "\\boxed{12 \\text{ cm}}", "12", 1),
    (T + "\\boxed{3.01}", "3", 0),                       # 整数答案不接受近似
    (T + "\\boxed{4} hmm \\boxed{3}", "3", 1),           # 以最后一个 boxed 为准
    ("<think>x = 3</think> so the answer is 3", "3", 0),  # 严格模式：无 boxed 不得分
    (T + "\\boxed{\\frac{\\sqrt{3}}{2}}", "\\frac { \\sqrt { 3 } } { 2 }", 1),
    (T + "\\boxed{2^{10}}", "1024", 1),
    (T + "\\boxed{9^{99999}}", "1", 0),                  # 不允许超大乘方
    (T + "\\boxed{__import__('os')}", "1", 0),
])
def test_grade(out, gt, exp):
    from reward_fn_geo3k import grade
    assert grade(out, gt)["acc"] == exp


def test_compute_score_dict_and_format(monkeypatch):
    from reward_fn_geo3k import compute_score
    monkeypatch.setenv("GEO3K_FORMAT_BONUS", "0.1")
    r = compute_score("hiyouga/geometry3k", T + "\\boxed{5}", "5")
    assert r == {"score": 1.1, "acc": 1.0, "format": 1.0, "boxed": 1.0}
    assert all(isinstance(v, float) for v in r.values())          # verl 会对额外字段求均值
    assert compute_score("x", "\\boxed{5} <think>a</think>", "5")["format"] == 0.0  # boxed 必须在 </think> 之后
    monkeypatch.setenv("GEO3K_FORMAT_BONUS", "0")
    assert compute_score("x", T + "\\boxed{4}", "5")["score"] == 0.0


def test_non_strict_mode(monkeypatch):
    from reward_fn_geo3k import grade
    monkeypatch.setenv("GEO3K_STRICT_BOXED", "0")
    assert grade("<think>x = 7</think> the answer is 3", "3")["acc"] == 1.0  # 只看 </think> 之后


def test_ground_truth_self_check():
    from reward_fn_geo3k import self_check_ground_truth
    for gt in ["3", "2 \\sqrt { 221 }", "\\frac { 26 } { 3 }", "12 \\pi", "5 \\sqrt { 3 }"]:
        assert self_check_ground_truth(gt)
    assert not self_check_ground_truth("")


# ---------------------------------------------------------------- 日志解析
def test_parse_step_line():
    from multiagent.trainer_agent import parse_step_line, val_accuracy
    line = ("(TaskRunner pid=4242) step:12 - actor/entropy:np.float64(0.53) - critic/score/mean:0.61 - "
            "val-core/hiyouga/geometry3k/acc/mean@1:np.float64(0.475) - val-core/hiyouga/geometry3k/reward/mean@1:0.56 - "
            "training/global_step:12 - perf/x:nan")
    step, m = parse_step_line(line)
    assert step == 12 and m["actor/entropy"] == 0.53 and m["critic/score/mean"] == 0.61
    acc, kind, key = val_accuracy(m)
    assert abs(acc - 47.5) < 1e-9 and kind == "acc"
    assert parse_step_line("Training Progress:  10%|█  | 12/120") is None
    assert val_accuracy({"val/test_score/hiyouga/geometry3k": 0.4})[1] == "score"
    assert val_accuracy({"critic/score/mean": 0.4})[0] is None


def test_classify_crash():
    from multiagent.trainer_agent import classify_crash
    assert classify_crash("torch.OutOfMemoryError: CUDA out of memory. Tried") == "gpu_oom"
    assert classify_crash("ray.exceptions.OutOfMemoryError: Task was killed due to the node running low on memory") == "host_oom"
    assert classify_crash("OSError: [Errno 28] No space left on device") == "disk_full"
    assert classify_crash("ValueError: boom") == "other"


# ---------------------------------------------------------------- 显存规划
def test_memory_plan():
    from multiagent.trainer_agent import memory_plan
    one80 = memory_plan(80, 1, 4.4, 0, 3072)
    assert one80["optimizer_offload"] and not one80["param_offload"]       # 70GB 训练状态放不进单张 80G
    two80 = memory_plan(80, 2, 4.4, 0, 3072)
    assert not two80["optimizer_offload"]
    one24 = memory_plan(24, 1, 4.4, 0, 3072)
    assert one24["param_offload"] and one24["optimizer_offload"]
    l3 = memory_plan(80, 8, 4.4, 3, 3072)
    assert l3["param_offload"] and l3["gpu_memory_utilization"] <= 0.45
    assert memory_plan(24, 2, 4.4, 0, 3072)["rollout_tp"] == 2


# ---------------------------------------------------------------- 数据清洗（合成数据，含注入缺陷）
@pytest.fixture()
def dry_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MA_DRY_RUN", "1")
    monkeypatch.setenv("MA_RUN_DIR", str(tmp_path / "run"))
    for m in list(sys.modules):
        if m.startswith("multiagent"):
            del sys.modules[m]
    yield tmp_path / "run"
    for m in list(sys.modules):
        if m.startswith("multiagent"):
            del sys.modules[m]


def test_data_clean(dry_env):
    from multiagent.common import Bus, load_config
    from multiagent.data_agent import DataAgent
    cfg = load_config()
    agent = DataAgent(cfg, Bus())
    acq = agent.acquire()
    assert acq["counts"]["train"] == 70 and acq["counts"]["test"] == 24
    rep = agent.clean()
    tr = rep["splits"]["train"]
    assert tr["issues"]["duplicate"] == 1
    assert tr["issues"]["leak_to_eval"] == 1
    assert tr["issues"]["bad_image"] == 1
    assert tr["issues"]["unverifiable_gt"] == 1
    assert tr["issues"]["opaque_answer"] == 1
    assert tr["kept"] == 65
    assert rep["splits"]["test"]["kept"] == 24                    # test 不删题
    from geo3k_data import read_split
    for rec in read_split(agent.path("clean", "train")):
        c = rec["prompt"][0]["content"]
        assert c.count("<image>") == 1 and c.startswith("<image>") and len(rec["images"]) == 1


def test_overrides_strict(dry_env):
    from multiagent.architect_agent import ArchitectAgent, PRESETS
    from multiagent.common import Blackboard, Bus, load_config
    from multiagent.trainer_agent import TrainerAgent
    cfg = load_config()
    bus = Bus()
    arch = ArchitectAgent(cfg, bus)
    caps = arch.probe_verl()
    assert caps["ok"] and caps["mode"] == "strict"
    tr = TrainerAgent(cfg, bus, Blackboard())
    prop = {"preset": "dapo_lite", **PRESETS["dapo_lite"], "max_response_length": 2048}
    plan = tr.make_plan(1, prop, {"train_file": "t.jsonl", "val_file": "v.jsonl", "max_prompt_length": 1024},
                        {"params_b": 4.4})
    args, dropped = tr.overrides(plan)
    assert not any(a.startswith("+") for a in args)
    assert "actor_rollout_ref.actor.clip_ratio_high=0.28" in args
    assert "actor_rollout_ref.actor.use_kl_loss=False" in args
    assert "trainer.test_freq=%d" % plan["save_freq"] in args
    # 旧版 verl 缺少 clip_ratio_high：dapo_lite 视为不可用
    capfile = dry_env / "verl_capabilities.json"
    c = json.loads(capfile.read_text())
    c["keys"] = [k for k in c["keys"] if k != "actor_rollout_ref.actor.clip_ratio_high"]
    capfile.write_text(json.dumps(c))
    with pytest.raises(RuntimeError):
        tr.overrides(plan)


def test_architect_proposals(dry_env):
    from multiagent.architect_agent import ArchitectAgent
    from multiagent.common import Bus, load_config
    a = ArchitectAgent(load_config(), Bus())
    a.probe_verl()
    first = a.propose([])
    assert first["preset"] == "grpo"
    r1 = {"preset": "grpo", "plan": {"max_response_length": 2048}, "val_gain": 3.0,
          "diagnostics": {"entropy_collapse": False, "response_clip_ratio": 0.15}}
    nxt = a.propose([r1])
    assert nxt["preset"] == "drgrpo" and nxt["max_response_length"] == 3072
    r1["diagnostics"] = {"entropy_collapse": True, "entropy_first": 1.0, "entropy_last": 0.1}
    assert a.propose([r1])["preset"] == "dapo_lite"


def test_rollout_engine():
    from multiagent.trainer_agent import rollout_engine
    assert rollout_engine("auto", {"sglang": "0.5", "vllm": "0.11"}) == "sglang"
    assert rollout_engine("auto", {"sglang": None, "vllm": "0.11"}) == "vllm"
    assert rollout_engine("auto", {"sglang": None, "vllm": None}) is None
    assert rollout_engine("sglang", {"sglang": None, "vllm": "0.11"}) is None


def test_find_local_model(tmp_path, monkeypatch):
    from multiagent import common
    d = tmp_path / "qwen3_vl_4b"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
    monkeypatch.setenv("MODEL_PATH", str(d))
    assert common.find_local_model() is None          # 没有 safetensors 不算有效权重
    (d / "model-00001-of-00002.safetensors").write_bytes(b"x")
    assert common.find_local_model() == str(d)


def test_format_template_opened_think(monkeypatch):
    from reward_fn_geo3k import compute_score
    monkeypatch.setenv("GEO3K_FORMAT_BONUS", "0.1")
    assert compute_score("x", "reasoning here...</think>\n\\boxed{5}", "5")["format"] == 1.0   # Thinking 模板
    assert compute_score("x", "</think>\\boxed{5}", "5")["format"] == 0.0                       # 空推理
    assert compute_score("x", "no think \\boxed{5}", "5")["format"] == 0.0


def test_classify_env_errors():
    from multiagent.trainer_agent import classify_crash
    assert classify_crash("ImportError: cannot import name 'flash_attn_func' from 'flash_attn'") == "env_error"
    assert classify_crash("omegaconf.errors.ConfigAttributeError: Key 'foo' is not in struct") == "config_error"


def test_sdpa_overrides(dry_env, monkeypatch):
    from multiagent.architect_agent import ArchitectAgent, PRESETS
    from multiagent.common import Blackboard, Bus, load_config
    from multiagent.trainer_agent import TrainerAgent
    monkeypatch.setenv("FAKE_NO_FLASH_ATTN", "1")
    cfg = load_config()
    ArchitectAgent(cfg, Bus()).probe_verl()
    tr = TrainerAgent(cfg, Bus(), Blackboard())
    plan = tr.make_plan(1, {"preset": "grpo", **PRESETS["grpo"], "max_response_length": 4096},
                        {"train_file": "t", "val_file": "v", "max_prompt_length": 768}, {"params_b": 4.4})
    args, _ = tr.overrides(plan)
    assert plan["attn_implementation"] == "sdpa"
    assert "actor_rollout_ref.model.use_remove_padding=False" in args
    assert "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" in args


def test_classify_vllm_wakeup_oom():
    from multiagent.trainer_agent import classify_crash
    msg = "Exception: Call to wake_up method failed: Worker failed with error 'CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:112'"
    assert classify_crash(msg) == "gpu_oom"


# ---------------------------------------------------------------- 旁路监控（wandb / TensorBoard 同步）
class _RecSink:
    name = "rec"

    def __init__(self):
        self.rows, self.evals_seen, self.events_seen, self.finished = [], [], [], []

    def round_rows(self, rnd, rows, meta):
        self.rows += [(rnd, r["step"], meta["verl"].get("data.train_batch_size")) for r in rows]

    def evals(self, table):
        self.evals_seen.append(sorted(table))

    def events(self, evs, start):
        self.events_seen += [(start + i, e["type"]) for i, e in enumerate(evs)]

    def finish_round(self, rnd):
        self.finished.append(rnd)

    def close(self):
        pass


def test_wandb_sync_incremental(tmp_path, monkeypatch):
    from multiagent import wandb_sync as ws
    monkeypatch.setattr(ws, "RUN_DIR", tmp_path)
    monkeypatch.setattr(ws, "MON_DIR", tmp_path / "monitor")
    monkeypatch.setattr(ws, "STATE_PATH", tmp_path / "monitor" / "sync_state.json")
    for r in (1, 2):
        d = tmp_path / "rounds" / f"round_{r}"
        d.mkdir(parents=True)
        (d / "plan.json").write_text(json.dumps({"preset": "grpo", "round": r}))
        (d / "command.sh").write_text("GEO3K_STRICT_BOXED=1 python3 -m verl.trainer.main_ppo data.train_batch_size=64 "
                                      "+actor_rollout_ref.model.override_config.attn_implementation=sdpa\n")
    m1 = tmp_path / "rounds" / "round_1" / "metrics.jsonl"
    m1.write_text("".join(json.dumps({"t": 1, "step": s, "critic/score/mean": 0.3, "x": float("nan")}) + "\n"
                          for s in (1, 2)))
    m2 = tmp_path / "rounds" / "round_2" / "metrics.jsonl"
    m2.write_text(json.dumps({"t": 1, "step": 0, "val-core/acc": 0.52}) + "\n" + '{"t": 2, "step"')  # 末行写到一半
    (tmp_path / "events.jsonl").write_text(json.dumps({"type": "decision", "content": {"decision": "x"}}) + "\n")
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "base_val.json").write_text(json.dumps({"summary": {"accuracy": 50.0}}))
    (tmp_path / "state.json").write_text(json.dumps({"phase": "train_round", "current": {"plan": {"round": 2}}}))

    syn = ws.Syncer({"wandb": "off", "tensorboard": False})
    rec = _RecSink()
    syn.sinks = [rec]
    assert syn.sync_once() is False
    assert rec.rows == [(1, 1, "64"), (1, 2, "64"), (2, 0, "64")]
    assert rec.finished == [1]                       # 已结束的轮次收尾，当前轮保持打开
    assert rec.events_seen == [(0, "decision")] and rec.evals_seen == [["base_val"]]
    # 第二次：只推新增内容；写完的半行被补上
    with m2.open("a") as f:
        f.write(': 1, "critic/score/mean": 0.4}\n')
    syn.sync_once()
    assert rec.rows[-1] == (2, 1, "64") and len(rec.rows) == 4
    assert len(rec.evals_seen) == 1 and len(rec.events_seen) == 1
    # 重启进程后从 sync_state.json 接着推，不重复
    syn2 = ws.Syncer({"wandb": "off", "tensorboard": False})
    rec2 = _RecSink()
    syn2.sinks = [rec2]
    (tmp_path / "state.json").write_text(json.dumps({"phase": "done"}))
    assert syn2.sync_once() is True
    assert rec2.rows == [] and rec2.finished == [2]


def test_wandb_sync_parse_command_and_numeric_row(tmp_path):
    from multiagent.wandb_sync import numeric_row, parse_command
    p = tmp_path / "command.sh"
    p.write_text("A=1 python3 -m verl.trainer.main_ppo trainer.logger=[console] 'x.y=a b' --cfg job\n")
    assert parse_command(p) == {"A": "1", "trainer.logger": "[console]", "x.y": "a b"}
    assert numeric_row({"t": 1, "step": 3, "a": 1.5, "b": float("inf"), "c": "s", "d": True}) == {"step": 3, "a": 1.5}


def test_disk_slots_account_for_verl_save_before_delete(dry_env, monkeypatch):
    """keep=1 时 verl 先写新检查点再删旧的：磁盘只够 1 个轮转位时计划要带磁盘警告。"""
    from multiagent import trainer_agent as ta
    from multiagent.architect_agent import ArchitectAgent, PRESETS
    from multiagent.common import Blackboard, Bus, load_config
    cfg = load_config()
    ArchitectAgent(cfg, Bus()).probe_verl()
    tr = ta.TrainerAgent(cfg, Bus(), Blackboard())
    prop = {"preset": "grpo", **PRESETS["grpo"], "max_response_length": 4096}
    data = {"train_file": "t", "val_file": "v", "max_prompt_length": 768}
    w = ta.ckpt_size_gb(4.4, False)
    monkeypatch.setattr(ta, "disk_free_gb", lambda *_: 5 + w + 1.5 * w)   # 本轮情形：钉住 + 1.5 个位
    plan = tr.make_plan(1, prop, data, {"params_b": 4.4})
    assert plan["keep_ckpts"] == 1 and not plan["save_optimizer"] and "写满磁盘" in plan["warning"]
    monkeypatch.setattr(ta, "disk_free_gb", lambda *_: 5 + w + 2.2 * w)
    plan = tr.make_plan(1, prop, data, {"params_b": 4.4})
    assert plan["keep_ckpts"] == 2 and not plan["disk_warning"]

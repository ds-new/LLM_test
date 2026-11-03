"""旁路监控：把多智能体 loop 的训练过程同步到 wandb 与 TensorBoard。

不改训练命令、不需要重启训练：只读 loop 已经写下的文件，增量推送：
  * rounds/round_N/metrics.jsonl  → wandb run「roundN_<preset>」：verl 每步的全部指标（reward、熵、
    回答长度、截断率、KL、梯度范数、各阶段耗时）与第 0/8/16… 步的验证准确率；横轴 global_step。
    断点续训后重复的步也会画出来（attempt 指标标出是第几次启动）。run 的 config 是 plan.json 与
    command.sh 里的全部 verl 参数。
  * eval/*.json                    → wandb run「agents」：基线、各轮导出模型、最终 test 的离线评估汇总表。
  * events.jsonl                   → wandb run「agents」：五个智能体之间的指令、报告与决策（表格）。
TensorBoard 写到 /root/tf-logs/…（AutoDL 控制台 → AutoPanel → TensorBoard 直接可看）；wandb 被墙或未登录时
TensorBoard 照常写，之后可在能访问 wandb 的机器上 `wandb sync --sync-tensorboard <目录>` 一次补传。

进度记在 RUN_DIR/monitor/sync_state.json，进程重启后接着推，wandb run 用固定 id 续写（resume="allow"）。

用法（工程根目录）：
  python -m multiagent.wandb_sync                 # 常驻（run_loop.sh 会自动拉起）
  python -m multiagent.wandb_sync --once          # 只同步一次
  python -m multiagent.wandb_sync --check         # 只检查 wandb 是否已登录、能否连通
  python -m multiagent.wandb_sync --export_offline DIR   # 连不上 wandb 时导出离线 run，换台机器 `wandb sync`
"""

import argparse
import json
import math
import os
import re
import shlex
import sys
import time
from pathlib import Path

from .common import DEFAULT_CONFIG, DRY_RUN, RUN_DIR, deep_merge, read_json, write_json

MON_DIR = RUN_DIR / "monitor"
STATE_PATH = MON_DIR / "sync_state.json"
EVENT_TEXT_MAX = 600


def log(msg):
    print(f"[{time.strftime('%F %T')}] wandb_sync: {msg}", flush=True)


def monitor_config():
    cfg = deep_merge(DEFAULT_CONFIG, read_json(RUN_DIR / "config.json", {}) or {})
    mon = dict(cfg.get("monitor") or {})
    if DRY_RUN:
        mon["tensorboard_dir"] = str(RUN_DIR / "tf-logs")
    return mon


# ---------------------------------------------------------------------- 读取 loop 产物
def numeric_row(d):
    """metrics.jsonl 的一行 → 只含有限数值的指标字典（去掉时间戳）。"""
    out = {}
    for k, v in d.items():
        if k in ("t",) or isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and math.isfinite(v):
            out[k] = v
    return out


def read_new_lines(path, start):
    """返回 (新行列表, 新的行计数)；只计完整的行，写到一半的末行留到下次。"""
    if not path.exists():
        return [], start
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    complete = [l for l in lines if l.endswith("\n")]
    rows = []
    for l in complete[start:]:
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            continue
    return rows, len(complete)


def parse_command(path):
    """command.sh → {key: value}：verl 的 Hydra 覆盖项与环境变量，作为 wandb config。"""
    if not path.exists():
        return {}
    try:
        toks = shlex.split(path.read_text(encoding="utf-8"), comments=True)
    except ValueError:
        toks = path.read_text(encoding="utf-8").split()
    out = {}
    for t in toks:
        if "=" in t and not t.startswith("-"):
            k, _, v = t.lstrip("+").partition("=")
            if re.fullmatch(r"[A-Za-z_][\w.]*", k):
                out[k] = v
    return out


def round_dirs():
    rd = RUN_DIR / "rounds"
    if not rd.exists():
        return []
    dirs = [p for p in rd.glob("round_*") if p.is_dir() and p.name.split("_")[-1].isdigit()]
    return sorted(dirs, key=lambda p: int(p.name.split("_")[-1]))


def loop_state():
    return read_json(RUN_DIR / "state.json", {}) or {}


def current_round(state):
    cur = state.get("current") or {}
    return (cur.get("plan") or {}).get("round")


def eval_summary(path):
    d = read_json(path, {}) or {}
    s = d.get("summary") or {}
    keep = {}
    for k, v in s.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            keep[k] = v
        elif k in ("backend", "model_path", "time", "input_file"):
            keep[k] = v
    return keep


def event_text(ev):
    c = ev.get("content")
    if isinstance(c, dict):
        head = c.get("decision") or c.get("action") or c.get("task") or c.get("phase") or ""
        why = c.get("why") or c.get("rationale") or ""
        rest = {k: v for k, v in c.items() if k not in ("decision", "action", "task", "why", "rationale")}
        text = " ".join(x for x in [str(head), str(why)] if x)
        if rest:
            text += " " + json.dumps(rest, ensure_ascii=False)
    else:
        text = str(c)
    return text[:EVENT_TEXT_MAX]


# ---------------------------------------------------------------------- 输出端
class TensorBoardSink:
    name = "tensorboard"

    def __init__(self, root):
        try:  # 延迟导入：只在需要时依赖 torch/tensorboard
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            from tensorboardX import SummaryWriter

        self._SW = SummaryWriter
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.writers = {}

    def _w(self, key):
        if key not in self.writers:
            self.writers[key] = self._SW(log_dir=str(self.root / key), flush_secs=30)
        return self.writers[key]

    def round_rows(self, rnd, rows, meta):
        w = self._w(f"round_{rnd}")
        if meta and not meta.get("_tb_config_logged"):
            w.add_text("config", "```\n" + json.dumps(meta, ensure_ascii=False, indent=1)[:60000] + "\n```", 0)
            meta["_tb_config_logged"] = True
        for r in rows:
            step = int(r.get("step", 0))
            for k, v in numeric_row(r).items():
                if k != "step":
                    w.add_scalar(k, v, step)
        w.flush()

    def evals(self, table):
        w = self._w("agents")
        for name, s in table.items():
            for k, v in s.items():
                if isinstance(v, (int, float)):
                    w.add_scalar(f"eval/{name}/{k}", v, 0)
        md = "| eval | acc | boxed | truncated | max_new_tokens | model |\n|---|---|---|---|---|---|\n" + "\n".join(
            f"| {n} | {s.get('accuracy')} | {s.get('boxed_rate')} | {s.get('truncated_rate')} | "
            f"{s.get('max_new_tokens')} | {s.get('model_path')} |" for n, s in sorted(table.items()))
        w.add_text("eval_summary", md, 0)
        w.flush()

    def events(self, evs, start_index):
        w = self._w("agents")
        for i, ev in enumerate(evs):
            w.add_text("events", f"`{ev.get('time')}` **{ev.get('from')} → {ev.get('to')}** ({ev.get('type')}): "
                                 f"{event_text(ev)}", start_index + i)
        w.flush()

    def finish_round(self, rnd):
        w = self.writers.pop(f"round_{rnd}", None)
        if w:
            w.close()

    def close(self):
        for w in self.writers.values():
            w.close()
        self.writers = {}


def wandb_logged_in():
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = Path(os.environ.get("NETRC", Path.home() / ".netrc"))
    try:
        return "api.wandb.ai" in netrc.read_text() or bool(os.environ.get("WANDB_BASE_URL"))
    except OSError:
        return False


class WandbSink:
    name = "wandb"

    def __init__(self, project, entity, state, offline_dir=None):
        import wandb

        self.wandb = wandb
        self.project, self.entity = project, entity
        self.offline_dir = offline_dir  # 非空：离线模式，写到该目录，之后在能连 wandb 的机器上 `wandb sync`
        self.state = state  # 共享的 sync_state["wandb"]（存 run id）
        self.runs = {}
        self.prefix = re.sub(r"[^A-Za-z0-9]+", "-", RUN_DIR.name).strip("-")[:20] or "run"
        self.events_rows = None
        self.last_events_upload = 0.0

    def _run(self, key, name, config=None, job_type=None):
        if key in self.runs:
            return self.runs[key]
        ids = self.state.setdefault("ids", {})
        rid = ids.get(key) or f"{self.prefix}-{key}-{self.wandb.util.generate_id()}"
        # id 只在建 run 成功后才记下：失败的 id 会被本进程的 wandb 占用（"run ID is in use"），重试须换新 id
        extra = ({"mode": "offline", "dir": str(self.offline_dir)} if self.offline_dir
                 else {"resume": "allow", "dir": str(MON_DIR)})
        run = self.wandb.init(project=self.project, entity=self.entity or None, id=rid, name=name,
                              group=RUN_DIR.name, job_type=job_type, config=config or {}, reinit="create_new",
                              settings=self.wandb.Settings(init_timeout=120, silent=True, console="off"), **extra)
        ids[key] = rid
        run.define_metric("global_step")
        run.define_metric("*", step_metric="global_step")
        self.runs[key] = run
        return run

    def round_rows(self, rnd, rows, meta):
        plan = meta.get("plan") or {}
        run = self._run(f"round{rnd}", f"round{rnd}_{plan.get('preset', 'train')}",
                        config={"plan": plan, "verl": meta.get("verl", {})}, job_type="train")
        for r in rows:
            m = numeric_row(r)
            m["global_step"] = m.pop("step", 0)
            run.log(m)

    def evals(self, table):
        run = self._run("agents", "agents", job_type="agents")
        cols = ["eval", "accuracy", "boxed_rate", "truncated_rate", "format_rate", "n_items", "n_samples",
                "max_new_tokens", "backend", "model_path", "time"]
        t = self.wandb.Table(columns=cols)
        for name, s in sorted(table.items()):
            t.add_data(name, *[s.get(c) for c in cols[1:]])
            for k in ("accuracy", "boxed_rate", "truncated_rate"):
                if k in s:
                    run.summary[f"eval/{name}/{k}"] = s[k]
        run.log({"eval_summary": t})

    def events(self, evs, start_index):
        run = self._run("agents", "agents", job_type="agents")
        if self.events_rows is None:
            self.events_rows = []
        for i, ev in enumerate(evs):
            self.events_rows.append([start_index + i, ev.get("time"), ev.get("from"), ev.get("to"), ev.get("type"),
                                     event_text(ev)])
            if ev.get("type") == "decision":
                run.summary["last_decision"] = event_text(ev)
        self.flush_events(force=False)

    def flush_events(self, force):
        # wandb 表格不可增量追加；整表重传，最多 10 分钟一次（结束时强制一次）
        if not self.events_rows or "agents" not in self.runs:
            return
        if not force and time.time() - self.last_events_upload < 600:
            return
        t = self.wandb.Table(columns=["index", "time", "from", "to", "type", "content"], data=self.events_rows)
        self.runs["agents"].log({"agent_events": t})
        self.last_events_upload = time.time()

    def finish_round(self, rnd):
        run = self.runs.pop(f"round{rnd}", None)
        if run:
            run.finish()

    def close(self):
        self.flush_events(force=True)
        for run in self.runs.values():
            run.finish()
        self.runs = {}


# ---------------------------------------------------------------------- 主循环
class Syncer:
    def __init__(self, mon, use_wandb=True, use_tb=True):
        self.mon = mon
        MON_DIR.mkdir(parents=True, exist_ok=True)
        self.state = read_json(STATE_PATH, {}) or {}
        self.sinks = []
        self.want_wandb = use_wandb and str(mon.get("wandb", "auto")).lower() not in ("off", "false", "0", "none")
        self.next_wandb_try = 0.0
        if use_tb and mon.get("tensorboard", True):
            try:
                self.sinks.append(TensorBoardSink(mon.get("tensorboard_dir") or str(RUN_DIR / "tf-logs")))
                log(f"TensorBoard → {mon.get('tensorboard_dir')}")
            except Exception as e:  # noqa: BLE001
                log(f"TensorBoard 不可用：{e}")
        self.meta_cache = {}

    def _try_wandb(self):
        if not self.want_wandb or any(s.name == "wandb" for s in self.sinks) or time.time() < self.next_wandb_try:
            return
        self.next_wandb_try = time.time() + 1800  # 失败后 30 分钟再试
        if not wandb_logged_in():
            log("wandb 未登录（在终端运行 `wandb login` 或设置 WANDB_API_KEY）；暂只写 TensorBoard，30 分钟后再检查")
            return
        try:
            sink = WandbSink(self.mon.get("project"), self.mon.get("entity"), self.state.setdefault("wandb", {}))
            sink._run("agents", "agents", job_type="agents")  # 先建一个 run 验证连通与登录
            self.sinks.append(sink)
            log(f"wandb 已连通：project={self.mon.get('project')}")
        except Exception as e:  # noqa: BLE001
            log(f"wandb 连接失败（{type(e).__name__}: {str(e)[:300]}）；暂只写 TensorBoard，30 分钟后重试")

    def _counters(self, sink):
        return self.state.setdefault("progress", {}).setdefault(sink.name, {})

    def _meta(self, rdir):
        if rdir.name not in self.meta_cache:
            self.meta_cache[rdir.name] = {"plan": read_json(rdir / "plan.json", {}) or {},
                                          "verl": parse_command(rdir / "command.sh")}
        return self.meta_cache[rdir.name]

    def sync_once(self):
        self._try_wandb()
        st = loop_state()
        cur = current_round(st)
        done = st.get("phase") == "done"
        for sink in list(self.sinks):
            c = self._counters(sink)
            try:
                for rdir in round_dirs():
                    rnd = int(rdir.name.split("_")[-1])
                    key = f"metrics_{rnd}"
                    rows, n = read_new_lines(rdir / "metrics.jsonl", c.get(key, 0))
                    if rows:
                        sink.round_rows(rnd, rows, self._meta(rdir))
                    c[key] = n
                    finished = c.setdefault("finished_rounds", [])
                    if (cur != rnd or done) and rnd not in finished and (rdir / "metrics.jsonl").exists():
                        sink.finish_round(rnd)
                        finished.append(rnd)
                evs, n = read_new_lines(RUN_DIR / "events.jsonl", c.get("events", 0))
                if evs:
                    sink.events(evs, c.get("events", 0))
                c["events"] = n
                table, mt = {}, c.setdefault("eval_mtime", {})
                changed = False
                for p in sorted((RUN_DIR / "eval").glob("*.json")):
                    table[p.stem] = eval_summary(p)
                    if mt.get(p.stem) != p.stat().st_mtime:
                        mt[p.stem] = p.stat().st_mtime
                        changed = True
                if changed and table:
                    sink.evals(table)
            except Exception as e:  # noqa: BLE001  单个输出端出错不影响其他输出端与训练
                log(f"{sink.name} 同步出错：{type(e).__name__}: {str(e)[:300]}")
                if sink.name == "wandb":
                    try:
                        sink.close()
                    except Exception:  # noqa: BLE001
                        pass
                    self.sinks.remove(sink)
                    self.next_wandb_try = time.time() + 600
        write_json(STATE_PATH, self.state)
        return done

    def close(self):
        for s in self.sinks:
            try:
                s.close()
            except Exception as e:  # noqa: BLE001
                log(f"{s.name} 关闭出错：{e}")
        write_json(STATE_PATH, self.state)


def export_offline(out_dir):
    """把整个运行（各轮训练曲线、离线评估、智能体决策）写成 wandb 离线 run，供之后 `wandb sync` 补传。

    用于实例连不上 wandb 的情况：在实例上导出、打包下载，再在能访问 wandb 的电脑上运行
    `wandb sync <out_dir>/wandb/offline-run-*`。不需要登录即可导出。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_MODE"] = "offline"
    mon = monitor_config()
    sink = WandbSink(mon.get("project"), mon.get("entity"), {}, offline_dir=out)
    table = {p.stem: eval_summary(p) for p in sorted((RUN_DIR / "eval").glob("*.json"))}
    n_rows = 0
    for rdir in round_dirs():
        rnd = int(rdir.name.split("_")[-1])
        rows, _ = read_new_lines(rdir / "metrics.jsonl", 0)
        if not rows:
            continue
        meta = {"plan": read_json(rdir / "plan.json", {}) or {}, "verl": parse_command(rdir / "command.sh")}
        sink.round_rows(rnd, rows, meta)
        run = sink.runs.get(f"round{rnd}")
        ev = table.get(f"round{rnd}_val") or {}
        if run is not None and ev:
            for k in ("accuracy", "boxed_rate", "truncated_rate"):
                if k in ev:
                    run.summary[f"offline_val/{k}"] = ev[k]
        n_rows += len(rows)
        sink.finish_round(rnd)
    evs, _ = read_new_lines(RUN_DIR / "events.jsonl", 0)
    if evs:
        sink.events(evs, 0)
    if table:
        sink.evals(table)
    final = (loop_state().get("final") or {})
    if final and "agents" in sink.runs:
        for k in ("best_round", "val_acc", "test_acc"):
            if k in final:
                sink.runs["agents"].summary[f"final/{k}"] = final[k]
    sink.close()
    runs = sorted(p.name for p in (out / "wandb").glob("offline-run-*"))
    log(f"离线导出完成：{n_rows} 行训练指标、{len(evs)} 条事件、{len(table)} 份评估 → {out}（{len(runs)} 个 run）")
    print(f"补传：wandb sync {out}/wandb/offline-run-*")
    return 0


def check():
    mon = monitor_config()
    print(f"project={mon.get('project')} entity={mon.get('entity')} 已登录={wandb_logged_in()}")
    if not wandb_logged_in():
        print("请先运行 `wandb login`（key 见 https://wandb.ai/authorize）")
        return 1
    import wandb

    try:
        viewer = wandb.Api(timeout=30).viewer
        print(f"wandb 连通，账号：{getattr(viewer, 'username', viewer)}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"wandb 连接失败：{type(e).__name__}: {str(e)[:400]}")
        return 2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--once", action="store_true", help="同步一次后退出")
    ap.add_argument("--check", action="store_true", help="只检查 wandb 登录与连通")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--no_tensorboard", action="store_true")
    ap.add_argument("--export_offline", metavar="DIR", default=None,
                    help="导出为 wandb 离线 run（实例连不上 wandb 时用），之后 `wandb sync DIR/wandb/offline-run-*`")
    args = ap.parse_args(argv)
    if args.check:
        return check()
    if args.export_offline:
        return export_offline(args.export_offline)
    mon = monitor_config()
    pid_path = MON_DIR / "wandb_sync.pid"
    MON_DIR.mkdir(parents=True, exist_ok=True)
    syncer = Syncer(mon, use_wandb=not args.no_wandb, use_tb=not args.no_tensorboard)
    if not args.once:
        pid_path.write_text(str(os.getpid()))
    try:
        while True:
            done = syncer.sync_once()
            if args.once or done:
                if done:
                    log("loop 已完成，最后一次同步后退出")
                break
            time.sleep(float(mon.get("poll_seconds", 60)))
    except KeyboardInterrupt:
        pass
    finally:
        syncer.close()
        if not args.once:
            pid_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

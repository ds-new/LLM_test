"""测试评估智能体：
  * 训练中（每 30 秒被训练监控回调一次）：某一步 N 出现评选集分数、且 global_step_N 已写完时，
    若分数创新高就把它复制到系统盘 /root/r1zero_best/global_step_N（最佳 checkpoint）；
    然后删掉数据盘上除最新以外的旧 checkpoint —— 数据盘上始终只有「最新」一份（写新的时短暂两份）。
  * 训练结束后：用完整 val.parquet（AIME24/25、AMC23 各 avg@16，MATH-500 全部）评估最佳和最终模型。

分数 = 评选集按样本数加权的准确率（MATH-500 200 / AMC23 80 / AIME24 60 / AIME25 60）。
2026-11-16 审计后：分数改用修正版判分 reward_v2 对验证生成重新打分（ma/rescore.py；原判分会把约 2.4% 的正确答案判错），
验证生成文件不可用时才退回训练日志里的原判分。最终评测同时给出 v2 和"去掉泄露题"两个口径（含 AIME 2026 留出集）。
"""
import json
import os
import shutil
import threading
import time
from pathlib import Path

from . import rescore

from .common import RUN, now, write_json
from .framework_agent import CKPT_DIR, plan

NAME = "eval"
BEST_ROOT = Path(os.environ.get("R1Z_BEST", "/root/r1zero_best"))
WEIGHTS = {"math500": 200, "amc23": 80, "aime24": 60, "aime25": 60}


def select_score(vals):
    tot = w = 0.0
    for k, v in vals.items():
        parts = k.split("/")
        if len(parts) >= 4 and parts[0] == "val-core" and parts[2] == "acc" and parts[1] in WEIGHTS:
            tot += WEIGHTS[parts[1]] * v
            w += WEIGHTS[parts[1]]
    return tot / w if w else None


def dir_gb(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 2**30


def ckpt_steps(ckpt_dir=CKPT_DIR):
    return sorted(int(d.name.split("_")[-1]) for d in Path(ckpt_dir).glob("global_step_*") if d.name.split("_")[-1].isdigit())


def ckpt_complete(d):
    """检查一个 global_step_N 是否完整、能载入。返回 (是否完整, 问题列表)。
    * HF 权重：model.safetensors.index.json 里列出的每个分片都在，且文件长度 ≥ 8 + 头长度 + 头里声明的最大数据偏移（能发现缺分片和写了一半的分片）；
    * dist_ckpt：metadata.json 在（verl 续训时先用它确认这是 Megatron 分布式 checkpoint）；
    * data.pt：数据读取进度在。"""
    import struct
    d = Path(d)
    hf, problems = d / "actor" / "huggingface", []
    idx = hf / "model.safetensors.index.json"
    if not idx.exists():
        return False, ["缺 model.safetensors.index.json"]
    try:
        shards = sorted(set(json.loads(idx.read_text())["weight_map"].values()))
    except Exception as e:  # noqa: BLE001
        return False, [f"index 读不了：{e!r}"]
    for name in shards:
        f = hf / name
        if not f.exists():
            problems.append(f"缺分片 {name}")
            continue
        try:
            with f.open("rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
            end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
            if f.stat().st_size < 8 + n + end:
                problems.append(f"分片不完整 {name}（{f.stat().st_size} < {8 + n + end} 字节）")
        except Exception as e:  # noqa: BLE001
            problems.append(f"分片读不了 {name}：{e!r}"[:200])
    if not (d / "actor" / "dist_ckpt" / "metadata.json").exists():
        problems.append("缺 dist_ckpt/metadata.json")
    if not (d / "data.pt").exists():
        problems.append("缺 data.pt")
    return not problems, problems


def latest_saved(ckpt_dir=CKPT_DIR):
    try:
        return int((Path(ckpt_dir) / "latest_checkpointed_iteration.txt").read_text().strip())
    except Exception:  # noqa: BLE001
        return -1


CKPT_GB_DEFAULT = 15.5  # Qwen3-8B bf16 的一份 checkpoint


def save_space(ckpt_dir=CKPT_DIR, margin_gb=1.5):
    """下一次存盘前的空间核算：新 checkpoint 写完并验证后才删旧的，所以要能和旧的并存。
    返回 (够不够, 可用 GB, 一份 checkpoint 的 GB)。2026-11-16 16:02 就是差 1.4 GB，把 step 50 写坏、训练中断。"""
    d = Path(ckpt_dir)
    real = [d / f"global_step_{k}" for k in ckpt_steps(d) if not (d / f"global_step_{k}").is_symlink()] if d.is_dir() else []
    need = max([dir_gb(p) for p in real] + [CKPT_GB_DEFAULT])
    probe = d if d.is_dir() else d.parent
    free = shutil.disk_usage(probe).free / 2**30
    return free >= need + margin_gb, free, need


class EvalAgent:
    def __init__(self, board, ckpt_dir=CKPT_DIR, start_path=None, since_key="round2_start"):
        """ckpt_dir：本阶段训练写 checkpoint 的目录；start_path：新阶段的起点 checkpoint（不在 ckpt_dir 里），
        起点步的验证分数用它对应（stage16k 的起点就是第二轮的最佳，已在系统盘上，不再复制）。"""
        self.board = board
        self.ckpt_dir = Path(ckpt_dir)
        self.start_path = Path(start_path) if start_path else None
        self.since = board.get(since_key, "")
        self.seen = set(board.get("eval_seen", []))
        self.best = board.get("best")
        self.copying = None  # 正在复制的线程
        self.pending_extra = {}
        try:
            self.since_ts = time.mktime(time.strptime(self.since[:19], "%Y-%m-%d %H:%M:%S")) if self.since else None
        except Exception:  # noqa: BLE001
            self.since_ts = None
        self._best_to_v2()

    def _best_to_v2(self):
        """沿用的最佳若是按原判分记的，用它的验证生成重新打分，保证和新分数可比。"""
        b = self.best
        if not b or b.get("metric") == "v2":
            return
        try:
            f = rescore.step_file(int(b["step"]))
            if f is None:
                self.board.post(NAME, "warning", {"why": "最佳的验证生成文件不在，无法换算成 v2 分数，新旧分数口径不一致", "step": b["step"]})
                return
            r = rescore.score_file(f)
            self.best = {**b, "score_v1": b["score"], "score": r["score_v2"], "score_v2_clean": r["score_v2_clean"], "metric": "v2"}
            self.board.set(best=self.best)
            self.board.post(NAME, "best_rescored_v2", {"step": b["step"], "v1": b["score"], "v2": r["score_v2"]})
        except Exception as e:  # noqa: BLE001
            self.board.post(NAME, "rescore_error", {"step": b.get("step"), "err": repr(e)[:300]})

    def val_rows(self, mfile):
        rows = {}
        if not Path(mfile).exists():
            return rows
        for line in open(mfile, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("t", "") < self.since:
                continue  # 只看第二轮的结果（第一轮的验证集不同，分数不可比）
            vals = {k: v for k, v in r.items() if k.startswith("val-core/")}
            if vals and "step" in r:
                rows.setdefault(int(r["step"]), {}).update(vals)
        return rows

    def busy(self):
        return self.copying is not None and self.copying.is_alive()

    def tick(self, mfile):
        if self.busy():
            return
        saved = latest_saved(self.ckpt_dir)
        if self.start_path is not None and saved < 0:
            saved = int(self.start_path.name.split("_")[-1])  # 新阶段还没存过盘：起点步可以先评
        for step, vals in sorted(self.val_rows(mfile).items()):
            if step in self.seen or step > saved:
                continue  # 已处理，或这一步的 checkpoint 还没写完
            gp = rescore.VALGEN_DIR / f"{step}.jsonl"
            if gp.exists() and gp.stat().st_mtime >= (self.since_ts or 0) and time.time() - gp.stat().st_mtime < 30:
                continue  # 验证生成刚写完，下个 tick 再用 v2 打分
            self.seen.add(step)
            self.board.set(eval_seen=sorted(self.seen))
            s = select_score(vals)
            extra = {"metric": "v1", "score_v1": None if s is None else round(s, 4)}
            try:
                f = rescore.step_file(step, self.since_ts)
                if f is not None:
                    r2 = rescore.score_file(f)
                    if r2["score_v2"] is not None:
                        s = r2["score_v2"]
                        extra = {"metric": "v2", "score_v1": r2["score_v1"], "score_v2_clean": r2["score_v2_clean"], "v2": r2["v2"]}
            except Exception as e:  # noqa: BLE001
                self.board.post(NAME, "rescore_error", {"step": step, "err": repr(e)[:300]})
            src = self.ckpt_dir / f"global_step_{step}"
            if not (src / "actor").is_dir() and self.start_path is not None and self.start_path.name == f"global_step_{step}":
                src = self.start_path
            have = (src / "actor" / "huggingface").is_dir()
            hist = {"step": step, "score": None if s is None else round(s, 4), "ckpt": have, **extra,
                    **{k.split("/")[1]: round(v, 4) for k, v in vals.items() if k.endswith("/acc/mean@1")}}
            with (RUN / "select_history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"t": now(), **hist}, ensure_ascii=False) + "\n")
            self.board.post(NAME, "select", hist | {"best": self.best and self.best["score"]})
            if s is not None and have and (not self.best or s > self.best["score"]):
                ok, problems = ckpt_complete(src)
                if not ok:
                    self.board.post(NAME, "best_skipped_incomplete", {"step": step, "score": round(s, 4), "problems": problems[:5]})
                    continue
                self.pending_extra = {k: v for k, v in extra.items() if k != "v2"}
                self.copying = threading.Thread(target=self.promote, args=(step, s, vals, src), daemon=True)
                self.copying.start()
                return  # 复制完成前不清理
        self.prune(saved)

    def promote(self, step, score, vals, src):
        try:
            BEST_ROOT.mkdir(parents=True, exist_ok=True)
            if BEST_ROOT.resolve() in Path(src).resolve().parents:
                # 起点 checkpoint 本身就是系统盘上的最佳：只更新分数（按本阶段的验证设置重新评过），不复制也不删
                meta = {"step": step, "score": score, "metrics": vals, "path": str(src), "t": now(), **self.pending_extra}
                write_json(BEST_ROOT / "best.json", meta)
                self.best = meta
                self.board.set(best=meta)
                self.board.post(NAME, "new_best", {"step": step, "score": round(score, 4), "path": str(src), "rescored": True})
                return
            # 系统盘放不下两份：先删旧的最佳。此刻新的最佳还在数据盘上，删旧的不会造成两头都没有
            for d in list(BEST_ROOT.glob("global_step_*")) + list(BEST_ROOT.glob(".tmp_*")):
                shutil.rmtree(d, ignore_errors=True)
            need, free = dir_gb(src), shutil.disk_usage(BEST_ROOT).free / 2**30
            if free < need + 1:
                self.board.post(NAME, "best_copy_failed", {"step": step, "need_gb": round(need, 1), "free_gb": round(free, 1)})
                return
            tmp, dst = BEST_ROOT / f".tmp_global_step_{step}", BEST_ROOT / f"global_step_{step}"
            shutil.copytree(src, tmp)
            tmp.rename(dst)
            meta = {"step": step, "score": score, "metrics": vals, "path": str(dst), "t": now(), **self.pending_extra}
            write_json(BEST_ROOT / "best.json", meta)
            self.best = meta
            self.board.set(best=meta)
            self.board.post(NAME, "new_best", {"step": step, "score": round(score, 4), "path": str(dst)})
        except Exception as e:  # noqa: BLE001
            self.board.post(NAME, "best_copy_failed", {"step": step, "error": repr(e)[:400]})

    def prune(self, saved=None):
        """删掉数据盘上除最新以外的 checkpoint。
        每 4 步验证、每 2 步存盘时，不验证那一步的 checkpoint 永远不会被打分；而某一步的验证结果和存盘写在同一行日志里，
        最新 checkpoint 前进之后，更早的步不会再有分数，所以可以直接删。唯一例外：最佳正在从它复制。"""
        saved = latest_saved(self.ckpt_dir) if saved is None else saved
        if saved < 0 or self.busy() or not (self.ckpt_dir / f"global_step_{saved}").is_dir():
            return
        older = [k for k in ckpt_steps(self.ckpt_dir) if k < saved]
        if not older:
            # 只剩最新一份、也没有正在写的新目录时，核算下一次存盘的空间；不够就提前告警（每步一次），由 Claude 处理
            if not [k for k in ckpt_steps(self.ckpt_dir) if k > saved] and self.board.get("disk_warned") != saved:
                ok, free, need = save_space(self.ckpt_dir)
                if not ok:
                    self.board.set(disk_warned=saved)
                    self.board.post(NAME, "warning", {"why": "disk_low：下一次存盘放不下", "free_gb": round(free, 1), "ckpt_gb": round(need, 1), "latest": saved})
            return
        ok, problems = ckpt_complete(self.ckpt_dir / f"global_step_{saved}")
        if not ok:
            # 最新的不完整：保留旧的（续训时回退到它），只告警一次
            if self.board.get("incomplete_warned") != saved:
                self.board.set(incomplete_warned=saved)
                self.board.post(NAME, "ckpt_incomplete", {"step": saved, "problems": problems[:5], "kept": older})
            return
        for k in ckpt_steps(self.ckpt_dir):
            if k >= saved:
                # 只删比最新更旧的。verl 先写 global_step_N 目录、几秒后才更新 latest 标记，
                # 比 latest 新的目录可能正在写——2026-11-16 00:56 就在这个空档里删掉了正在写的 step 56
                continue
            if self.best and self.best["step"] == k and not Path(self.best["path"]).is_dir():
                continue  # 最佳还没复制成功，别删源头
            target = self.ckpt_dir / f"global_step_{k}"
            if target.is_symlink():
                target.unlink()  # 指向系统盘最佳的链接：只解除链接，绝不删它指向的目录
            else:
                shutil.rmtree(target, ignore_errors=True)
            self.board.post(NAME, "pruned", {"step": k, "latest": saved, "symlink": target.is_symlink()})

    def rollback_to_best(self, mfile):
        """最新 checkpoint 损坏时：删掉 ckpt_dir 里的 checkpoint，把系统盘上的最佳复制回来作为最新，
        最佳之后那段作废的指标/评选记录挪到 *_rollback_<时间>.jsonl。返回回退到的步数，没有可用的最佳时返回 None。"""
        import subprocess
        import time as _t
        best = self.best
        if not best or not (Path(best["path"]) / "actor" / "huggingface").is_dir():
            return None
        step, tag = int(best["step"]), _t.strftime("%Y%m%d_%H%M")
        for k in ckpt_steps(self.ckpt_dir):
            shutil.rmtree(self.ckpt_dir / f"global_step_{k}", ignore_errors=True)
        dst = self.ckpt_dir / f"global_step_{step}"
        subprocess.run(["cp", "-a", best["path"], str(dst)], check=True)
        (self.ckpt_dir / "latest_checkpointed_iteration.txt").write_text(str(step))
        cut = best.get("t", "")
        for path in (Path(mfile), RUN / "select_history.jsonl"):
            if not path.exists():
                continue
            keep, gone = [], []
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    keep.append(line)
                    continue
                (gone if (r.get("t", "") >= cut and int(r.get("step", -1)) > step) else keep).append(line)
            path.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
            if gone:
                (path.parent / f"{path.stem}_rollback_{tag}.jsonl").write_text("\n".join(gone) + "\n", encoding="utf-8")
        self.seen = {s for s in self.seen if s <= step}
        self.board.set(eval_seen=sorted(self.seen))
        self.board.post(NAME, "rollback_to_best", {"step": step, "from": str(best["path"])})
        return step

    def wait(self):
        if self.copying is not None:
            self.copying.join()

    def final_eval(self, trainer, tag, path, level, kind="eval"):
        """用完整验证集评估一个 checkpoint（verl val_only）。kind=eval16k 时按 16K 生成长度评估。返回 {数据集: acc} 或 None。"""
        p = plan(kind, level, exp=f"final_eval_{tag}", resume_path=path)
        ok, err = trainer.validate(p)
        if not ok:
            self.board.post(NAME, "final_eval_failed", {"tag": tag, "why": "config", "err": err[-500:]})
            return None
        r = trainer.train(p, hang_min=90)
        res = {}
        mfile = Path(r["metrics"])
        if mfile.exists():
            for line in mfile.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                res.update({k.split("/")[1]: v for k, v in row.items() if k.startswith("val-core/") and k.endswith("/acc/mean@1")})
        out = {"tag": tag, "path": str(path), "status": r["status"], **res}
        try:  # v2 判分 + 去掉泄露题的口径（含 AIME 2026 留出集）
            f = rescore.newest_file(p["valgen_dir"]) if p.get("valgen_dir") else None
            if f is not None:
                r2 = rescore.score_file(f)
                out.update(v1=r2["v1"], v2=r2["v2"], v2_clean=r2["v2_clean"], n=r2["n"], valgen=str(f))
        except Exception as e:  # noqa: BLE001
            self.board.post(NAME, "rescore_error", {"tag": tag, "err": repr(e)[:300]})
        self.board.post(NAME, "final_eval", out)
        return out if res else None

"""结果分析智能体：
  * 训练中：每出现新的一步就做健康检查，异常发告警到事件总线；梯度/损失连续 2 步出现 NaN/Inf 判为致命，由决策智能体停训。
  * 结束时：汇总两轮指标、评选集分数、最佳 checkpoint、完整验证集结果，写 run/REPORT_round2.md 和曲线图 run/curves_round2.png。
"""
import json
import math
import statistics
from pathlib import Path

from .common import RUN, now

NAME = "analysis"
VALGEN_DIR = Path("/root/r1zero_valgen")


def load(mfile, since=""):
    out = []
    if Path(mfile).exists():
        for line in Path(mfile).read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("t", "") >= since:
                out.append(r)
    return out


def bad(v):
    return v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))


class AnalysisAgent:
    def __init__(self, board, since_key="round2_start", mfile=None, title="r1zero 第二轮（从 step 40 续训）报告", out="REPORT_round2"):
        self.board = board
        self.since = board.get(since_key, "")
        self.mfile = Path(mfile) if mfile else RUN / "metrics_r1zero_qwen3_8b_dapo.jsonl"
        self.title, self.out = title, out
        self.checked = set(board.get("analysis_checked", []))
        self.nan_streak = 0
        self.warned = {}

    def warn(self, kind, step, **info):
        last = self.warned.get(kind, -99)
        if step - last >= 10:  # 同类告警 10 步内只发一次
            self.warned[kind] = step
            self.board.post(NAME, "warning", {"kind": kind, "step": step, **info})

    def check(self, mfile):
        """返回致命原因（字符串）或 None。"""
        rows = [r for r in load(mfile, self.since) if "critic/score/mean" in r]
        hist = [r for r in load(mfile) if "critic/score/mean" in r]
        for r in rows:
            s = int(r["step"])
            if s in self.checked:
                continue
            self.checked.add(s)
            g, l = r.get("actor/grad_norm"), r.get("actor/pg_loss")
            self.nan_streak = self.nan_streak + 1 if (bad(g) or bad(l)) else 0
            if self.nan_streak >= 2:
                return f"第 {s} 步：grad_norm/pg_loss 连续 2 步为 NaN/Inf"
            ent = r.get("actor/entropy", 1)
            prev5 = [x for x in hist if int(x["step"]) < s][-5:]
            if len(prev5) >= 3:
                med_ent = statistics.median(x.get("actor/entropy", 0) for x in prev5)
                med_len = statistics.median(x.get("response_length/mean", 0) for x in prev5)
                if med_ent > 0 and ent > 3 * med_ent and r.get("response_length/mean", 0) < 0.5 * med_len:
                    # 2026-11-15 实测：续训后优化器主参数没同步，第一次更新就把模型重置回基座，表现正是熵猛涨、长度骤降
                    return (f"第 {s} 步疑似权重被重置回基座：熵 {ent:.3f}（前 5 步中位数 {med_ent:.3f}），"
                            f"长度 {r.get('response_length/mean', 0):.0f}（中位数 {med_len:.0f}）")
            # 2026-11-17（第三轮 A 把每批更新次数提到 8 次）：更新步子过大的两个信号，连续 2 步就暂停等 Claude 判断
            cf = r.get("actor/pg_clipfrac", 0) or 0
            self.clip_streak = getattr(self, "clip_streak", 0) + 1 if cf > 0.2 else 0
            if self.clip_streak >= 2:
                return f"第 {s} 步：裁剪比例连续 2 步超过 20%（{cf:.3f}），更新步子过大"
            if cf > 0.08:
                self.warn("clipfrac_high", s, pg_clipfrac=round(cf, 4))
            low_ent = len(prev5) >= 3 and med_ent > 0 and ent < 0.5 * med_ent
            self.ent_streak = getattr(self, "ent_streak", 0) + 1 if low_ent else 0
            if self.ent_streak >= 2:
                return f"第 {s} 步：熵连续 2 步低于前 5 步中位数的一半（{ent:.3f}），疑似熵塌缩"
            if ent < 0.03:
                self.warn("entropy_collapse", s, entropy=round(ent, 4))
            # 2026-11-18：第三轮后期熵从 0.2 升到 0.3 以上（目前效果是变好的）。超过 0.5 告警，连续 2 步超过 1.0 视为熵爆炸，暂停等 Claude
            if ent > 0.5:
                self.warn("entropy_high", s, entropy=round(ent, 4))
            self.ent_hi_streak = getattr(self, "ent_hi_streak", 0) + 1 if ent > 1.0 else 0
            if self.ent_hi_streak >= 2:
                return f"第 {s} 步：熵连续 2 步超过 1.0（{ent:.3f}），疑似熵爆炸"
            ngb = r.get("train/num_gen_batches", 0)
            if ngb >= 4:  # 动态采样要生成 4 批以上才凑够 64 个有效题组：题目对当前模型太简单或太难
                self.warn("many_gen_batches", s, num_gen_batches=int(ngb))
            clip = r.get("response_length/clip_ratio", 0)
            if clip > 0.25:
                self.warn("too_many_truncated", s, clip_ratio=round(clip, 3))
            prev = [x for x in hist if int(x["step"]) < s][-10:]
            if len(prev) >= 5:
                med = statistics.median(x.get("timing_s/step", 0) for x in prev)
                if med and r.get("timing_s/step", 0) > 1.6 * med:
                    self.warn("slow_step", s, step_s=round(r["timing_s/step"]), median_s=round(med))
                base = statistics.mean(x["critic/score/mean"] for x in prev)
                if r["critic/score/mean"] < base - 0.35:
                    self.warn("reward_drop", s, score=round(r["critic/score/mean"], 3), prev10=round(base, 3))
            if s % 10 == 0:
                self.board.post(NAME, "summary", {"step": s, "score": round(r["critic/score/mean"], 3),
                                                  "entropy": round(ent, 3), "len": round(r.get("response_length/mean", 0)),
                                                  "clip": round(clip, 3), "step_min": round(r.get("timing_s/step", 0) / 60, 1)})
        self.board.set(analysis_checked=sorted(self.checked))
        return None

    def valgen_stats(self):
        """最近一次验证样本：截断（无 Answer 行）比例、明显重复的比例。"""
        files = sorted(VALGEN_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not files:
            return None
        n = noans = rep = 0
        for line in files[-1].read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            out = str(r.get("output", ""))
            n += 1
            noans += "Answer:" not in out[-600:]
            tailtxt = out[-2000:]
            rep += len(tailtxt) > 1000 and len(set(tailtxt[i:i + 50] for i in range(0, len(tailtxt) - 50, 50))) < 15
        return {"file": files[-1].name, "n": n, "no_answer": round(noans / max(n, 1), 3), "repetitive": round(rep / max(n, 1), 3)}

    def plot(self, rows, sel, out):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:  # noqa: BLE001
            return False
        tr = [r for r in rows if "critic/score/mean" in r]
        x = [r["step"] for r in tr]
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        ax[0][0].plot(x, [(r["critic/score/mean"] + 1) / 2 for r in tr], ".-"); ax[0][0].set_title("train reward → (score+1)/2")
        ax[0][1].plot(x, [r.get("response_length/mean", 0) for r in tr], ".-"); ax[0][1].set_title("response length (mean)")
        ax[1][0].plot(x, [r.get("actor/entropy", 0) for r in tr], ".-"); ax[1][0].set_title("entropy")
        if sel:
            ax[1][1].plot([s["step"] for s in sel], [s["score"] for s in sel], "o-"); ax[1][1].set_title("select-set score")
        for a in ax.flat:
            a.grid(alpha=.3); a.set_xlabel("step")
        fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
        return True

    def report(self, final_status, reason, final_eval=None):
        st = self.board.state
        all_rows = load(self.mfile)
        r2 = [r for r in all_rows if r.get("t", "") >= self.since and "critic/score/mean" in r]
        sel = [json.loads(l) for l in (RUN / "select_history.jsonl").read_text().splitlines()] if (RUN / "select_history.jsonl").exists() else []
        L = [f"# {self.title}", "", f"- 结束：{now()}，状态：**{final_status}**（{reason}）",
             f"- 设置：{self.board.get('settings', '每 2 步存盘、每 4 步验证（第 52 步起；之前每 2 步验证）；评选集 400 条；DAPO 超长软惩罚（>6144 token 线性扣分）；只存模型权重')}", ""]
        best = st.get("best")
        if best:
            L += [f"- 最佳 checkpoint：step {best['step']}，评选集分数 {best['score']:.4f}，位置 {best['path']}", ""]
        if final_eval:
            L += ["## 完整验证集（AIME/AMC avg@16，MATH-500 全量）", "", "| 模型 | aime24 | aime25 | amc23 | math500 |", "| --- | --- | --- | --- | --- |",
                  "| step 0（基座） | 0.046 | 0.044 | 0.280 | 0.506 |", "| step 25（第一轮） | 0.144 | 0.150 | 0.547 | 0.752 |"]
            for e in final_eval:
                if e:
                    L.append(f"| {e['tag']} ({Path(e['path']).name}) | " + " | ".join(f"{e.get(k, float('nan')):.3f}" for k in ("aime24", "aime25", "amc23", "math500")) + " |")
            L.append("")
        if sel:
            L += ["## 评选集分数（第 52 步前每 2 步、之后每 4 步）", "", "| 步 | 分数 | math500 | amc23 | aime24 | aime25 |", "| --- | --- | --- | --- | --- | --- |"]
            L += [f"| {s['step']} | {s['score']} | {s.get('math500')} | {s.get('amc23')} | {s.get('aime24')} | {s.get('aime25')} |" for s in sel]
            L.append("")
        if r2:
            L += ["## 训练指标（第二轮，score 已含超长惩罚）", "", "| 步 | score | 长度 | 截断率 | 熵 | 每步分钟 |", "| --- | --- | --- | --- | --- | --- |"]
            pick = r2 if len(r2) <= 40 else r2[::max(1, len(r2) // 40)] + r2[-1:]
            L += [f"| {r['step']} | {r['critic/score/mean']:.3f} | {r.get('response_length/mean', 0):.0f} | {r.get('response_length/clip_ratio', 0):.3f} | "
                  f"{r.get('actor/entropy', 0):.3f} | {r.get('timing_s/step', 0) / 60:.1f} |" for r in pick]
            L.append("")
        vg = self.valgen_stats()
        if vg:
            L += [f"- 最近一次验证样本：无 Answer 行 {vg['no_answer']:.1%}，明显重复 {vg['repetitive']:.1%}（{vg['file']}，{vg['n']} 条）", ""]
        warns = [json.loads(l) for l in (RUN / "events.jsonl").read_text().splitlines() if '"from": "analysis", "kind": "warning"' in l and l > '{"t": "' + self.since]
        if warns:
            L += ["## 告警", ""] + [f"- {w['t']} step {w['step']}：{w['kind']} {json.dumps({k: v for k, v in w.items() if k not in ('t', 'from', 'kind', 'step')}, ensure_ascii=False)}" for w in warns] + [""]
        png = self.out.replace("REPORT", "curves") + ".png"
        if self.plot(all_rows, sel, RUN / png):
            L += [f"![curves]({png})", ""]
        (RUN / f"{self.out}.md").write_text("\n".join(L), encoding="utf-8")
        return RUN / f"{self.out}.md"

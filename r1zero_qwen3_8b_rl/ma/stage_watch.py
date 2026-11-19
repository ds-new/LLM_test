"""第三轮 B（upd8dyn）收尾旁路脚本（2026-11-17 用户要求：评测结束后自动跑配对比较，把结论写进文件，然后立刻关机）。

B 跑完 -> 流水线自动完整评测 -> 暂停（state.json：stage=upd8dyn、phase=waiting_claude、waiting.stage_done=true）。这时本脚本：
  1. 用完整评测的生成结果（/root/r1zero_valgen/final_final_eval_r3_upd8dyn_*）和 step 96（B 的起点）做同题配对比较；
  2. 按下面事先定好的标准写结论：run/audit/STAGE_B_RESULT.md（和 stage_b_result.json）；
  3. 立刻关机（不等看门狗的 30 分钟宽限）。
事先定好的判定标准（2026-11-17 20:35，看结果前定）：
  * 主比较：B 终点 step 116 vs B 起点 step 96（和 A 一样都是 20 步，不涉及挑选）；
  * 主指标：MATH-500 / AMC23 / AIME24 / AIME25 四个评测集等权平均的差值，按题目配对 bootstrap 10000 次的 95% 区间；
    区间下限 > 0 为显著提升，上限 < 0 为显著下降，否则无显著差异；
  * AIME 2026 留出集单独报告；B 期间评选分最高的 checkpoint 只作参考（有挑选偏差）。
只读评测结果和日志，不改任何 checkpoint。训练中途出错暂停（不是 stage_done）时不动，交给看门狗。
启动：cd /root/autodl-tmp/r1zero && setsid nohup nice python ma/stage_watch.py >> run/logs/stage_watch.log 2>&1 < /dev/null &
测试：python ma/stage_watch.py --dry-run   （用 step 76 -> 96 的现有结果写一份测试报告到 /tmp，不关机）
停用：touch run/STOP_STAGE_WATCH
"""
import collections
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "run"
DATA = RUN / "data"
VG = Path("/root/r1zero_valgen")
def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


# 2026-11-17 起通用于第三轮各阶段：--stage 阶段名（默认 B=upd8dyn），--base 起点完整评测的生成文件/目录，--base-label 起点名
STAGE = arg("--stage", "upd8dyn")
OUT_MD = RUN / "audit" / ("STAGE_B_RESULT.md" if STAGE == "upd8dyn" else "STAGE_%s_RESULT.md" % STAGE)
OUT_JSON = RUN / "audit" / ("stage_b_result.json" if STAGE == "upd8dyn" else "stage_%s_result.json" % STAGE)
LABELS = {"upd8dyn": "动态采样（B）", "lr2": "学习率 2e-6（C）", "lr2dyn": "学习率 2e-6，保留动态采样（C）", "frontier": "边界题（D）"}
# 自动接续（用户 2026-11-17 同意 B -> C -> D 连续跑；2026-11-19 起 D 结束后不关机，--then sft 时直接启动蒸馏 SFT（E），见 start_e）
NEXT = {"upd8dyn": lambda v: "lr2dyn" if v == "显著提升" else "lr2", "lr2": lambda v: "frontier", "lr2dyn": lambda v: "frontier"}


def result_json(stage):
    return RUN / "audit" / ("stage_b_result.json" if stage == "upd8dyn" else "stage_%s_result.json" % stage)
LABEL = LABELS.get(STAGE, STAGE)
BASE46 = VG / "final_final_eval_r2_best_v2"
BASE76 = VG / "final_final_eval_r2_last_v2"
BASE96 = VG / "final_final_eval_r3_upd8_best"
SETS4 = ["math500", "amc23", "aime24", "aime25"]
ALL5 = SETS4 + ["aime26"]
NAMES = {"math500": "MATH-500", "amc23": "AMC23", "aime24": "AIME24", "aime25": "AIME25", "aime26": "AIME 2026（留出）"}
A_LINE = ("A（step 76 -> 96，只把每批更新次数改成 8）：四个原评测集平均 -0.32，95% 区间 [-1.84, +1.14]；"
          "AIME 2026 +1.1，区间 [-0.9, +3.9]；无显著差异。")


def log(msg):
    print(time.strftime("%F %T"), msg, flush=True)


_SRC = None
_CACHE = {}


def src_map():
    global _SRC
    if _SRC is None:
        import pandas as pd
        _SRC = {}
        for f in ("val.parquet", "test_aime26.parquet"):
            if (DATA / f).exists():
                for _, r in pd.read_parquet(DATA / f).iterrows():
                    _SRC[r["prompt"][0]["content"][-200:]] = r["data_source"]
    return _SRC


def newest(p):
    if not p:
        return None
    p = Path(p)
    if p.is_file():
        return p
    if p.is_dir():
        fs = sorted(p.glob("*.jsonl"), key=lambda f: f.stat().st_mtime)
        return fs[-1] if fs else None
    return None


def load(p):
    """返回 ({题目: (数据集, 该题多次采样的平均正确率)}, 文件)。"""
    f = newest(p)
    if f is None:
        return None, None
    if str(f) in _CACHE:
        return _CACHE[str(f)], f
    per = collections.defaultdict(list)
    for line in open(f, encoding="utf-8"):
        d = json.loads(line)
        per[d["input"]].append(float(d.get("acc", 0)) > 0.5)
    sm = src_map()
    out = {k: (next((s for t, s in sm.items() if t in k), "unknown"), float(np.mean(v))) for k, v in per.items()}
    _CACHE[str(f)] = out
    return out, f


def set_acc(D):
    by = collections.defaultdict(list)
    for s, a in D.values():
        by[s].append(a)
    return {s: 100 * float(np.mean(v)) for s, v in by.items()}


def compare(A, B, seed=0):
    rng = np.random.default_rng(seed)
    keys = sorted(set(A) & set(B))
    by = collections.defaultdict(list)
    for k in keys:
        by[A[k][0]].append(k)
    res = {"n_pairs": len(keys), "sets": {}}
    boots = {}
    for s in ALL5:
        ks = by.get(s)
        if not ks:
            continue
        d = np.array([B[k][1] - A[k][1] for k in ks])
        bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
        boots[s] = bs
        res["sets"][s] = dict(n=len(ks), a=100 * float(np.mean([A[k][1] for k in ks])), b=100 * float(np.mean([B[k][1] for k in ks])),
                              diff=100 * float(d.mean()), lo=100 * float(np.percentile(bs, 2.5)), hi=100 * float(np.percentile(bs, 97.5)),
                              p=float((bs > 0).mean()))
    for name, sets in (("macro4", SETS4), ("macro5", ALL5)):
        sets = [s for s in sets if s in boots]
        if len(sets) < 2:
            continue
        bs = np.mean([boots[s] for s in sets], axis=0)
        res[name] = dict(sets=sets, diff=float(np.mean([res["sets"][s]["diff"] for s in sets])),
                         lo=100 * float(np.percentile(bs, 2.5)), hi=100 * float(np.percentile(bs, 97.5)), p=float((bs > 0).mean()))
    return res


def verdict(m):
    if not m:
        return "无法判断（缺少评测结果）"
    if m["lo"] > 0:
        return "显著提升"
    if m["hi"] < 0:
        return "显著下降"
    return "无显著差异"


def cmp_table(title, c, la, lb):
    L = ["### " + title, "", "配对题数 %d。" % c["n_pairs"], "",
         "| 数据集 | 题数 | %s | %s | 差值 | 95%% 区间 | 提升的概率 |" % (la, lb), "|---|---|---|---|---|---|---|"]
    for s in ALL5:
        x = c["sets"].get(s)
        if x:
            L.append("| %s | %d | %.1f%% | %.1f%% | %+.1f | [%+.1f, %+.1f] | %.2f |" % (NAMES[s], x["n"], x["a"], x["b"], x["diff"], x["lo"], x["hi"], x["p"]))
    for name, lab in (("macro4", "**四个原评测集平均**"), ("macro5", "五个评测集平均（含 AIME 2026）")):
        m = c.get(name)
        if m:
            L.append("| %s | | | | %+.2f | [%+.2f, %+.2f] | %.3f |" % (lab, m["diff"], m["lo"], m["hi"], m["p"]))
    return L + [""]


def train_summary(since):
    rows, sel, warn = [], [], []
    for line in open(RUN / "metrics_r1zero_qwen3_8b_dapo.jsonl", encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if d.get("t", "") >= since and "timing_s/step" in d and "step" in d:
            rows.append(d)
    for line in open(RUN / "select_history.jsonl", encoding="utf-8"):
        try:
            x = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if x.get("t", "") >= since:
            sel.append((x.get("step"), x.get("score")))
    bad = ("warning", "ckpt_incomplete", "best_skipped_incomplete", "rescore_error", "best_copy_failed", "exception", "final_eval_failed")
    for line in open(RUN / "events.jsonl", encoding="utf-8"):
        try:
            x = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if x.get("t", "") >= since and (x.get("kind") in bad or (x.get("kind") == "result" and x.get("status") not in ("done", None))):
            warn.append(json.dumps(x, ensure_ascii=False)[:200])
    t = dict(n=len(rows), select=sel, warnings=warn)
    if rows:
        g = lambda k: [float(r[k]) for r in rows if k in r] or [0.0]  # noqa: E731
        t.update(first=int(rows[0]["step"]), last=int(rows[-1]["step"]), step_min=float(np.mean(g("timing_s/step"))) / 60,
                 gen_batches={str(k): v for k, v in sorted(collections.Counter(int(x) for x in g("train/num_gen_batches")).items())},
                 ent_min=min(g("actor/entropy")), ent_max=max(g("actor/entropy")), clip_max=max(g("actor/pg_clipfrac")),
                 trunc=float(np.mean(g("response_length/clip_ratio"))), length=float(np.mean(g("response_length/mean"))))
        if any("rollout_corr/kl" in r for r in rows):
            t.update(rc_kl=float(np.mean(g("rollout_corr/kl"))), rc_kl_max=max(g("rollout_corr/kl")), rc_chi2=float(np.mean(g("rollout_corr/chi2_token"))))
    return t


def b_candidates(st):
    out, seen = [], set()
    for fe in st.get("final_eval") or []:
        if not fe:
            continue
        path = str(fe.get("path") or "")
        step = int(path.rstrip("/").rsplit("_", 1)[-1]) if "global_step_" in path else None
        out.append(dict(tag=fe.get("tag"), step=step, valgen=fe.get("valgen")))
        seen.add(str(newest(fe.get("valgen")) or ""))
    for d in sorted(VG.glob("final_final_eval_r3_%s_*" % STAGE)):  # 状态里没记下的（例如评测返回空）也找出来
        f = newest(d)
        if f is not None and str(f) not in seen:
            out.append(dict(tag=d.name.replace("final_final_eval_", ""), step=int(f.stem) if f.stem.isdigit() else None, valgen=str(f)))
    return out


def context_lines():
    L = ["- " + A_LINE]
    for stg in ("upd8dyn", "lr2", "lr2dyn"):
        jf = result_json(stg)
        if stg == STAGE or not jf.exists():
            continue
        try:
            b = json.loads(jf.read_text())
            m = (b.get("main") or {}).get("macro4")
            if m:
                L.append("- %s（step %s 起）：四个原评测集平均 %+.2f，95%% 区间 [%+.2f, %+.2f]；%s。"
                         % (LABELS.get(stg, stg), b.get("start_step", "?"), m["diff"], m["lo"], m["hi"], b.get("verdict")))
        except Exception:  # noqa: BLE001
            pass
    return L


def suggestion(v):
    if STAGE == "upd8dyn":
        if v == "显著提升":
            return "动态采样有效：后续实验保留它。按计划下一步是学习率 2e-6，配置用 lr2dyn（B 的配置 + 学习率 2e-6）。"
        return ("动态采样%s：后续实验去掉它（它让每步慢约 35%%）。按计划下一步是学习率 2e-6，配置用 lr2（A 的配置 + 学习率 2e-6）。"
                % ("反而让效果变差" if v == "显著下降" else "没有可测的效果"))
    if STAGE in ("lr2", "lr2dyn"):
        head = {"显著提升": "学习率 2e-6 有效：之后保留。", "显著下降": "学习率 2e-6 让效果变差：之后回到 1e-6。"}.get(
            v, "学习率 2e-6 没有可测的效果：之后保持原来的 1e-6（更稳）。")
        return head + "按计划下一步是边界题实验（D）：先离线测训练题的通过率，只训边界题，每题 n=32。"
    if STAGE == "frontier":
        head = {"显著提升": "只训边界题有效。", "显著下降": "只训边界题让效果变差。"}.get(v, "只训边界题没有可测的效果。")
        return (head + "纯 R1-Zero 这条线到此结束：下一步整理 R1-Zero 阶段报告，然后按计划做蒸馏 SFT（E）。"
                "开始 E 前需要先下载并去重蒸馏数据，并确认数据盘空间（建议扩容约 50 GB）。")
    return "开机后由 Claude 判断下一步。"


def passk(f):
    """AMC/AIME（每题 16 次）的平均正确率、多数投票、至少对一次。"""
    import reward_v2 as R
    sm = src_map()
    per = collections.defaultdict(list)
    for line in open(f, encoding="utf-8"):
        d = json.loads(line)
        a, _ = R.final_answer(d["output"])
        per[d["input"]].append((R._norm_txt(a) if a else None, float(d.get("acc", 0)) > 0.5))
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for k, v in per.items():
        if len(v) < 16:
            continue
        src = next((s for t, s in sm.items() if t in k), "unknown")
        votes = collections.defaultdict(list)
        for ans, ok in v:
            if ans:
                votes[ans].append(ok)
        maj = max(votes.values(), key=len) if votes else [False]
        res[src]["mean"].append(np.mean([ok for _, ok in v]))
        res[src]["maj"].append(np.mean(maj) > 0.5)
        res[src]["pass"].append(any(ok for _, ok in v))
    return {s: tuple(100 * float(np.mean(r[x])) for x in ("mean", "maj", "pass")) for s, r in res.items()}


def build(st, cands, base_dir, base_label, target, out_md, out_json, extra_base=None):
    since = st.get("round3_start", "")
    base, _ = load(base_dir)
    if base is None:
        raise RuntimeError("找不到基准评测结果：%s" % base_dir)
    got = []
    for c in cands:
        D, f = load(c.get("valgen"))
        if D is None and c.get("tag"):
            D, f = load(VG / ("final_final_eval_" + c["tag"]))
        if D is None or any(g["f"] == str(f) for g in got):
            continue
        step = c.get("step") if c.get("step") is not None else (int(f.stem) if f.stem.isdigit() else None)
        got.append(dict(tag=c.get("tag"), step=step, D=D, f=str(f)))
    primary = next((g for g in got if g["step"] == target), None)
    if primary is None and got:
        primary = max(got, key=lambda g: g["step"] or -1)
    others = [g for g in got if g is not primary]
    main = compare(base, primary["D"]) if primary else None
    vs_extra = None
    if primary and extra_base is not None:
        E, _ = load(extra_base)
        vs_extra = compare(E, primary["D"]) if E is not None else None
    oth = [(g, compare(base, g["D"])) for g in others]
    m4 = (main or {}).get("macro4")
    v = verdict(m4) if primary else "无法判断（没有找到 B 的完整评测结果）"
    sug = suggestion(v) if (primary and m4) else "开机后由 Claude 检查完整评测为什么没有结果，再手动比较。"
    plabel = "step %s" % (primary["step"] if primary else "?")
    L = ["# 第三轮 %s 结果" % LABEL, "", "自动生成于 %s（ma/stage_watch.py，生成后立即关机）。" % time.strftime("%F %T"), "", "## 结论", ""]
    if primary and m4:
        a26 = main["sets"].get("aime26")
        L.append("**%s（step %s -> %s）：%s。** 四个原评测集平均 %+.2f 个百分点，95%% 区间 [%+.2f, %+.2f]"
                 % (LABEL, st.get("start_step"), primary["step"], v, m4["diff"], m4["lo"], m4["hi"])
                 + ("；AIME 2026 留出集 %+.1f，区间 [%+.1f, %+.1f]。" % (a26["diff"], a26["lo"], a26["hi"]) if a26 else "。"))
    else:
        L.append("**%s**" % v)
    L += ["", "前面各阶段："] + context_lines() + ["", "建议：" + sug + "（这是脚本按事先定好的标准给出的，你回来后 Claude 会再核对一遍。）", ""]
    L += ["## 完整评测（v2 判分，平均正确率）", "", "| checkpoint | MATH-500 | AMC23 | AIME24 | AIME25 | AIME 2026 |", "|---|---|---|---|---|---|"]
    rows_abs = [("step 46（第二轮评选最佳）", load(BASE46)[0]), ("step 76（第二轮最后）", load(BASE76)[0]), ("step 96（A 结束 / B 起点）", load(BASE96)[0])]
    if str(newest(base_dir)) not in {str(newest(x)) for x in (BASE46, BASE76, BASE96)}:
        rows_abs.append(("%s（本阶段起点）" % base_label, base))
    for g in got:
        lab = "step %s" % g["step"] + ("（本阶段结束）" if (g is primary and g["step"] == target) else ("（本阶段评选分最高，有挑选偏差）" if g is not primary else "（本阶段）"))
        rows_abs.append((lab, g["D"]))
    for lab, D in rows_abs:
        if D is None:
            continue
        acc = set_acc(D)
        L.append("| %s | " % lab + " | ".join(("%.1f%%" % acc[s]) if s in acc else "—" for s in ALL5) + " |")
    try:
        pk = [(base_label + "（起点）", passk(newest(base_dir)))] + ([(plabel + "（终点）", passk(primary["f"]))] if primary else [])
        L += ["", "## AMC / AIME：平均正确率 / 多数投票 / 16 次里至少对一次", "",
              "| checkpoint | AMC23 | AIME24 | AIME25 | AIME 2026 |", "|---|---|---|---|---|"]
        for lab, r in pk:
            L.append("| %s | " % lab + " | ".join(("%.1f / %.1f / %.1f" % r[s]) if s in r else "—" for s in ["amc23", "aime24", "aime25", "aime26"]) + " |")
    except Exception as e:  # noqa: BLE001
        L += ["", "（多数投票表生成失败：%r）" % e]
    L += ["", "## 同题配对比较", ""]
    if main:
        L += cmp_table("%s vs %s（主比较）" % (plabel, base_label), main, base_label, plabel)
    if vs_extra:
        L += cmp_table("%s vs step 76（第三轮累计）" % plabel, vs_extra, "step 76", plabel)
    for g, c in oth:
        L += cmp_table("step %s vs %s（本阶段评选分最高，有挑选偏差，仅供参考）" % (g["step"], base_label), c, base_label, "step %s" % g["step"])
    t = train_summary(since)
    L += ["## 本阶段训练过程", ""]
    if t.get("n"):
        L.append("- 步数：step %d–%d，共 %d 步，平均每步 %.1f 分钟" % (t["first"], t["last"], t["n"], t["step_min"]))
        L.append("- 动态采样每步生成的批数：" + "，".join("%s 批 %d 步" % (k, v2) for k, v2 in t["gen_batches"].items()))
        L.append("- 熵 %.3f–%.3f；最大裁剪比例 %.4f；平均截断 %.1f%%；平均长度 %.0f token" % (t["ent_min"], t["ent_max"], t["clip_max"], 100 * t["trunc"], t["length"]))
    if "rc_kl" in t:
        L.append("- 生成端和训练端的概率差异：KL 平均 %.5f、最大 %.5f；token 级 χ² 平均 %.4f" % (t["rc_kl"], t["rc_kl_max"], t["rc_chi2"]))
    L.append("- 评选分（400 条，误差约 ±0.024）：" + ("，".join("step %s %s" % (s, sc) for s, sc in t["select"]) or "无"))
    L.append("- 告警：" + ("无" if not t["warnings"] else "%d 条" % len(t["warnings"])))
    L += ["    " + w for w in t["warnings"]]
    lat = RUN / "ckpt" / "r1zero_qwen3_8b_dapo" / "latest_checkpointed_iteration.txt"
    latest = lat.read_text().strip() if lat.exists() else "?"
    try:
        best = json.loads(Path("/root/r1zero_best/best.json").read_text())
    except Exception:  # noqa: BLE001
        best = {}
    L += ["", "## checkpoint", "", "- 数据盘最新：step %s（run/ckpt/r1zero_qwen3_8b_dapo/global_step_%s）" % (latest, latest),
          "- 系统盘最佳（按 400 条评选分选的）：step %s，%s" % (best.get("step"), best.get("path")), "",
          "## 判定标准（看结果前定的）", "",
          "- 主比较：B 终点 vs B 起点 step 96（和 A 一样都是 20 步，不涉及挑选）。",
          "- 主指标：MATH-500 / AMC23 / AIME24 / AIME25 四个评测集等权平均的差值；按题目配对 bootstrap 10000 次取 95% 区间。"
          "区间下限 > 0 为显著提升，上限 < 0 为显著下降，否则为无显著差异。",
          "- AIME 2026 留出集单独报告；B 期间评选分最高的 checkpoint 只作参考。", ""]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L), encoding="utf-8")
    r = dict(t=time.strftime("%F %T"), stage=STAGE, start_step=st.get("start_step"), verdict=v, suggestion=sug, primary=dict(step=primary["step"], file=primary["f"]) if primary else None,
             main=main, vs76=vs_extra, others=[dict(step=g["step"], file=g["f"], cmp=c) for g, c in oth], train=t)
    out_json.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


def ready(st):
    w = st.get("waiting") or {}
    return st.get("stage") == STAGE and st.get("phase") == "waiting_claude" and bool(w.get("stage_done"))


def shutdown(reason):
    try:
        with open(RUN / "logs" / "watchdog.log", "a") as fo:
            fo.write("%s shutdown (stage_watch): %s\n" % (time.strftime("%F %T"), reason))
    except Exception:  # noqa: BLE001
        pass
    try:
        sys.path.insert(0, str(ROOT))
        from ma.common import Board
        Board().post("stage_watch", "shutdown", {"reason": reason})
    except Exception as e:  # noqa: BLE001
        log("board post failed: %r" % e)
    log("shutdown: " + reason)
    subprocess.run("sync; /usr/bin/shutdown", shell=True)


def start_e(reason):
    """用户 2026-11-19 09:03：D 之后的蒸馏 SFT 不用再确认，直接开始 -> D 结束后不关机，启动 sft/start_e.sh（它自己负责最后关机）。"""
    try:
        with open(OUT_MD, "a", encoding="utf-8") as fo:
            fo.write("\n## 自动接续\n\n按用户 2026-11-19 09:03 的指示：D 结束后不关机，直接开始蒸馏 SFT（E）。"
                     "总控脚本 sft/start_e.sh，日志 run/logs/e_stage.log，结论 run/audit/STAGE_E_RESULT.md。\n")
    except Exception:  # noqa: BLE001
        pass
    log(reason + "；不关机，自动接续 -> E（蒸馏 SFT）")
    subprocess.Popen(["setsid", "bash", str(ROOT / "sft" / "start_e.sh")], cwd=str(ROOT), stdin=subprocess.DEVNULL,
                     stdout=open(RUN / "logs" / "start_e.out", "a"), stderr=subprocess.STDOUT)


def main():
    if "--dry-run" in sys.argv:
        assert ready({"stage": STAGE, "phase": "waiting_claude", "waiting": {"stage_done": True}})
        assert not ready({"stage": STAGE, "phase": "waiting_claude", "waiting": {"reason": "crash"}})
        assert not ready({"stage": STAGE, "phase": "train"})
        assert not ready({"stage": "upd8", "phase": "waiting_claude", "waiting": {"stage_done": True}})
        st = json.loads((RUN / "state.json").read_text())
        out = Path("/tmp/stage_%s_test.md" % STAGE)
        r = build(st, [dict(tag="dryrun", step=96, valgen=str(newest(BASE96)))], BASE76, "step 76", 96, out, out.with_suffix(".json"))
        print(out.read_text())
        print("verdict:", r["verdict"])
        return
    log("start: waiting for stage=%s stage_done" % STAGE)
    while True:
        if (RUN / "STOP_STAGE_WATCH").exists():
            log("STOP_STAGE_WATCH present, exit")
            return
        try:
            st = json.loads((RUN / "state.json").read_text())
        except Exception:  # noqa: BLE001
            time.sleep(30)
            continue
        if st.get("phase") in ("done", "failed"):
            log("phase=%s, exit" % st.get("phase"))
            return
        if st.get("stage") != STAGE:
            log("stage=%s (not %s), exit" % (st.get("stage"), STAGE))
            return
        if ready(st):
            log("B stage done: building result")
            time.sleep(20)
            try:
                st = json.loads((RUN / "state.json").read_text())
                base, base_label = BASE96, "step 96"
                if STAGE != "upd8dyn":  # 之后的阶段：起点 = 上一阶段的终点；没给 --base 时从上一阶段的结论文件里读
                    prev = ((st.get("prev_stages") or [{}])[-1] or {}).get("stage")
                    jb = result_json(prev) if prev else None
                    prim = (json.loads(jb.read_text()).get("primary") or {}) if (jb and jb.exists()) else {}
                    base = arg("--base", prim.get("file"))
                    base_label = arg("--base-label", "step %s" % prim.get("step", st.get("start_step")))
                r = build(st, b_candidates(st), base, base_label, int(st.get("target_steps") or 116), OUT_MD, OUT_JSON, extra_base=BASE76)
                reason = "%s 阶段完成，结论：%s，见 %s" % (STAGE, r["verdict"], OUT_MD)
                if STAGE == "frontier" and arg("--then") == "sft":
                    start_e(reason)
                    return
                if arg("--then") == "auto" and STAGE in NEXT and r["verdict"] in ("显著提升", "无显著差异", "显著下降"):
                    # 用户 2026-11-17 同意：B -> C -> D 连续跑，按事先定好的规则选下一阶段的配置
                    nxt = NEXT[STAGE](r["verdict"])
                    with open(OUT_MD, "a", encoding="utf-8") as fo:
                        fo.write("\n## 自动接续\n\n按事先定好的规则（用户 2026-11-17 同意），已自动开始下一阶段 %s：%s，"
                                 "从 step %s 起训 20 步；结束后同样自动做配对比较、写结论（run/audit/STAGE_%s_RESULT.md）并关机。\n"
                                 % (nxt, LABELS[nxt], (r.get("primary") or {}).get("step"), nxt))
                    log(reason + "；自动接续 -> " + nxt)
                    subprocess.Popen(["setsid", "bash", str(ROOT / "start_next_stage.sh"), nxt], cwd=str(ROOT), stdin=subprocess.DEVNULL,
                                     stdout=open(RUN / "logs" / "start_next_stage.log", "a"), stderr=subprocess.STDOUT)
                    return
            except Exception as e:  # noqa: BLE001
                OUT_MD.parent.mkdir(parents=True, exist_ok=True)
                OUT_MD.write_text("# 第三轮 %s 结果\n\n自动生成结论时出错" % STAGE + "（训练和评测结果都还在，开机后由 Claude 手动比较）：\n\n"
                                  + "\n".join("    " + x for x in traceback.format_exc()[-3000:].splitlines()) + "\n", encoding="utf-8")
                reason = "%s 阶段完成，自动比较出错：%r" % (STAGE, e)
                if STAGE == "frontier" and arg("--then") == "sft":
                    start_e(reason)
                    return
            log(reason)
            shutdown(reason)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()

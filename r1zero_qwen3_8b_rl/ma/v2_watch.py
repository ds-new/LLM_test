"""第二轮旁路监控（只读，不动训练和 checkpoint）：第二轮的评测智能体是按原判分挑最佳的（进程已在跑，改代码要重启训练）。
这里每出一批验证生成就用 reward_v2 重新打分，写 run/select_history_v2.jsonl；
若 v2 下的最佳和现行最佳（/root/r1zero_best/best.json）不是同一步，写 run/v2_flag.json，由 Claude 在每小时检查时决定怎么处理。
第二轮结束（phase=done/failed）或出现 run/STOP_V2WATCH 时退出。"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ma import rescore  # noqa: E402

RUN = Path(__file__).resolve().parent.parent / "run"
HIST, FLAG = RUN / "select_history_v2.jsonl", RUN / "v2_flag.json"
CUTOFF = time.mktime(time.strptime("2026-11-16 14:11:00", "%Y-%m-%d %H:%M:%S"))  # 第二轮从 step 46 续训的时间；更早的 48+ 文件属于已作废的那条训练线
BASE_STEP = 46  # 续训起点（同一条训练线）

def done_steps():
    if not HIST.exists():
        return {}
    return {r["step"]: r for r in map(json.loads, HIST.read_text().splitlines())}

while True:
    try:
        st = json.loads((RUN / "state.json").read_text())
        if st.get("phase") in ("done", "failed") or (RUN / "STOP_V2WATCH").exists():
            break
        have = done_steps()
        for p in sorted(rescore.VALGEN_DIR.glob("*.jsonl"), key=lambda p: int(p.stem)):
            s = int(p.stem)
            if s in have or (s != BASE_STEP and p.stat().st_mtime < CUTOFF) or time.time() - p.stat().st_mtime < 60:
                continue
            r = rescore.score_file(p)
            row = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "step": s, **{k: r[k] for k in ("score_v1", "score_v2", "score_v2_clean", "v2")}}
            with HIST.open("a") as f:
                f.write(json.dumps(row) + "\n")
            have[s] = row
        if have:
            b2 = max(have.values(), key=lambda r: (r["score_v2"], -r["step"]))
            best = json.loads(Path("/root/r1zero_best/best.json").read_text())
            if b2["step"] != best["step"] and b2["score_v2"] > have.get(best["step"], {}).get("score_v2", -1):
                ck = RUN / "ckpt" / "r1zero_qwen3_8b_dapo" / f"global_step_{b2['step']}"
                FLAG.write_text(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "v2_best_step": b2["step"], "v2_best": b2["score_v2"],
                                            "current_best_step": best["step"], "current_best_v2": have.get(best["step"], {}).get("score_v2"),
                                            "ckpt_exists": ck.is_dir()}, indent=1))
            elif FLAG.exists():
                FLAG.unlink()
    except Exception as e:  # noqa: BLE001
        print(time.strftime("%H:%M:%S"), "error", repr(e)[:300], flush=True)
    time.sleep(60)
rescore.reward_v2.V1._kill_pool()

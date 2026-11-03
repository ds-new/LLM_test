"""数据清洗智能体：获取 geometry3k、转换格式、清洗、测 prompt 长度、按难度过滤训练集。

清洗规则（训练集严格；val/test 只修复不删题，保证与公开结果可比，删除的情况单独报告）：
  1. 图像能解码；超大图按最长边缩放（控制视觉 token），过小图记录（processor 会放大）；
  2. prompt 恰好一个 <image>（verl 要求占位符数 == 图像数，否则训练直接报错）；
  3. 标准答案非空，且判分器认为"标准答案自身正确"（否则这道题奖励恒为 0，只会浪费 rollout）；
  4. 无数字且无法求值的答案（如选项字母）无法可靠判分，剔除（可配置）；
  5. 训练集内部去重；与 val/test 同题同图（文本 + 图像均值哈希）的训练样本剔除（防泄漏）；
  6. 难度过滤：基线模型对每道训练题采样 k 次，全对的题 GRPO 组内优势恒为 0，只保留少量做锚点。
"""

import json
import random
from collections import Counter
from pathlib import Path

from geo3k_data import (average_hash, build_prompt_text, clean_question, load_raw, make_synthetic_raw, open_image,
                        png_bytes, read_split, record_image_bytes, split_ext, write_split)
from preprocess_geo3k import convert
from reward_fn_geo3k import latex_value, normalize_answer, self_check_ground_truth

from .common import DRY_RUN, RUN_DIR, now, read_json, stamp, write_json

NAME = "data"
SPLITS = ("train", "val", "test")


class DataAgent:
    def __init__(self, config, bus):
        self.config = config
        self.dcfg = config["data"]
        self.bus = bus
        self.dir = RUN_DIR / "data"
        self.ext = split_ext()

    def path(self, stage, split):
        return self.dir / stage / f"{split}{self.ext}"

    def report(self, action, content):
        self.bus.post(NAME, "decision", "report", {"action": action, **content})

    # ------------------------------------------------------------------
    def acquire(self):
        source = self.dcfg["source"]
        if DRY_RUN and not Path(source).exists():
            make_synthetic_raw(source)
        if all(self.path("converted", s).exists() for s in SPLITS):
            stats = {s: len(read_split(self.path("converted", s))) for s in SPLITS}
            self.report("acquire", {"reused": True, "counts": stats})
            return stats
        t0 = now()
        raw = load_raw(source)
        splits = convert(raw)
        counts = {}
        for s in SPLITS:
            write_split(splits[s], self.path("converted", s))
            counts[s] = len(splits[s])
        out = {"source": source, "raw_counts": {k: len(v) for k, v in raw.items()}, "counts": counts,
               "seconds": round(now() - t0, 1), "format": self.ext}
        write_json(self.dir / "acquire_report.json", out)
        self.report("acquire", out)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def answer_kind(ans):
        if latex_value(ans) is not None:
            return "numeric"
        norm = normalize_answer(ans)
        if not norm:
            return "empty"
        return "symbolic" if any(c.isdigit() for c in norm) else "opaque"

    def _fix_record(self, rec, issues):
        """修复图像与 <image> 占位符；无法修复返回 None（附原因）。"""
        data = record_image_bytes(rec)
        try:
            img = open_image(data)
        except Exception:
            issues["bad_image"] += 1
            return None, "bad_image"
        w, h = img.size
        side = self.dcfg["max_image_side"]
        if max(w, h) > side:
            r = side / max(w, h)
            img = img.resize((max(1, int(w * r)), max(1, int(h * r))))
            rec["images"] = [{"bytes": png_bytes(img), "path": None}]
            issues["resized_large"] += 1
        if min(img.size) < self.dcfg["min_image_side"]:
            issues["tiny_image"] += 1
        content = rec["prompt"][0]["content"]
        if content.count("<image>") != 1 or not content.startswith("<image>"):
            rec["prompt"] = [{"role": "user", "content": build_prompt_text(rec["extra_info"]["question"])}]
            issues["fixed_image_tag"] += 1
        return rec, average_hash(img)

    def clean(self):
        report = {"time": stamp(), "splits": {}}
        keys = {}
        cleaned = {}
        for split in ("val", "test", "train"):  # 先建 val/test 的指纹，训练集据此去泄漏
            rows = read_split(self.path("converted", split))
            issues = Counter()
            kept, seen = [], set()
            kinds = Counter()
            for rec in rows:
                rec, info = self._fix_record(rec, issues)
                if rec is None:
                    continue
                key = (clean_question(rec["extra_info"]["question"]).lower(), info)
                gt = rec["reward_model"]["ground_truth"]
                kind = self.answer_kind(gt)
                kinds[kind] += 1
                if split == "train":
                    if kind == "empty" or not self_check_ground_truth(gt):
                        issues["unverifiable_gt"] += 1
                        continue
                    if kind == "opaque" and self.dcfg["drop_opaque_answers"]:
                        issues["opaque_answer"] += 1
                        continue
                    if key in seen:
                        issues["duplicate"] += 1
                        continue
                    if key in keys.get("val", set()) or key in keys.get("test", set()):
                        issues["leak_to_eval"] += 1
                        continue
                else:
                    if kind == "empty" or not self_check_ground_truth(gt):
                        issues["unverifiable_gt_kept"] += 1  # 不删题，只报告
                    if key in seen:
                        issues["duplicate_kept"] += 1
                seen.add(key)
                rec["extra_info"]["answer_kind"] = kind
                kept.append(rec)
            keys[split] = seen
            cleaned[split] = kept
            report["splits"][split] = {"input": len(rows), "kept": len(kept), "issues": dict(issues),
                                       "answer_kinds": dict(kinds)}
        for split, rows in cleaned.items():
            write_split(rows, self.path("clean", split))
        report["prompt_tokens"] = self.measure_prompt_tokens(cleaned["train"] + cleaned["val"])
        write_json(self.dir / "clean_report.json", report)
        self.report("clean", report)
        return report

    def measure_prompt_tokens(self, rows, limit=400):
        """用模型 processor 实测 prompt token 数（含视觉 token），给出 max_prompt_length 建议。"""
        model_path = self.config["model"]["path"]
        if DRY_RUN or not Path(model_path, "config.json").exists():
            return {"measured": False, "reason": "dry run 或模型尚未就绪"}
        try:
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            return {"measured": False, "reason": f"{type(e).__name__}: {e}"}
        rng = random.Random(0)
        sample = rows if len(rows) <= limit else rng.sample(rows, limit)
        lens = []
        for rec in sample:
            text = rec["prompt"][0]["content"].replace("<image>", "")
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
            prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = proc(text=[prompt], images=[open_image(record_image_bytes(rec))], return_tensors="pt")["input_ids"]
            lens.append(int(ids.shape[1]))
        lens.sort()
        mx = lens[-1]
        rec_len = max(512, ((int(mx * 1.15) + 255) // 256) * 256)
        return {"measured": True, "n": len(lens), "p50": lens[len(lens) // 2], "p99": lens[int(len(lens) * 0.99) - 1],
                "max": mx, "recommended_max_prompt_length": rec_len}

    # ------------------------------------------------------------------
    def difficulty_filter(self, passrate_json):
        res = read_json(passrate_json)
        rows = read_split(self.path("clean", "train"))
        rate = {it["index"]: it["pass_rate"] for it in res["items"]}
        rng = random.Random(20261101)
        kept, dropped_easy, missing = [], 0, 0
        for rec in rows:
            r = rate.get(rec["extra_info"]["index"])
            if r is None:
                missing += 1
                kept.append(rec)
                continue
            rec["extra_info"]["base_pass_rate"] = r
            if r >= 1.0 and rng.random() > self.dcfg["keep_easy_fraction"]:
                dropped_easy += 1
                continue
            kept.append(rec)
        hist = Counter(round(rec["extra_info"].get("base_pass_rate", -1), 3) for rec in kept)
        out = {"input": len(rows), "kept": len(kept), "dropped_all_correct": dropped_easy, "no_rate": missing,
               "all_wrong_kept": sum(1 for r in kept if r["extra_info"].get("base_pass_rate") == 0),
               "pass_rate_hist": {str(k): v for k, v in sorted(hist.items())}}
        if len(kept) < self.dcfg["min_train_items"]:
            out["applied"] = False
            out["reason"] = f"过滤后只剩 {len(kept)} 题（< {self.dcfg['min_train_items']}），保留清洗后的完整训练集"
        else:
            write_split(kept, self.path("filtered", "train"))
            out["applied"] = True
        write_json(self.dir / "difficulty_report.json", out)
        self.report("difficulty_filter", out)
        return out

    def train_file(self):
        f = self.path("filtered", "train")
        return f if f.exists() else self.path("clean", "train")

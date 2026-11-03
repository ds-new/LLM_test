"""geometry3k 数据公共层：提示词、记录格式、读写（parquet / jsonl）、合成数据。

预处理脚本、多智能体里的数据清洗智能体和评估脚本都只通过这里读写数据，保证三处
看到的 prompt 和 ground_truth 完全一致（之前评估脚本单独复制了一份指令文本）。

读写后端：
  * .parquet —— AutoDL 上的正式格式（verl RLHFDataset 直接读取）。优先用 `datasets`，
    否则用 pandas + pyarrow；图像列为 [{"bytes": PNG, "path": None}]，与 datasets.Image 一致。
  * .jsonl   —— 无 pyarrow 的机器（例如在 Mac 上干跑多智能体 loop）使用，图像 base64 编码。
"""

import base64
import io
import json
import os
import random
from pathlib import Path

DATA_SOURCE = "hiyouga/geometry3k"
IMAGE_TOKEN = "<image>"

# R1-Zero 风格指令：先在 <think></think> 内推理，最终答案放入 \boxed{}
INSTRUCTION = (
    "You FIRST think about the reasoning process as an internal monologue "
    "and then provide the final answer.\n"
    "The reasoning process MUST be enclosed within <think> </think> tags.\n"
    "The final answer MUST be put in \\boxed{}."
)


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------
def clean_question(problem: str) -> str:
    """去掉所有 <image> 占位符与多余空白，得到纯文本题干。"""
    return " ".join(str(problem).replace(IMAGE_TOKEN, " ").split())


def build_prompt_text(problem: str) -> str:
    """训练用 prompt：恰好一个 <image>（verl 要求占位符数量 == 图像数量），后接题干与指令。"""
    return IMAGE_TOKEN + clean_question(problem) + "\n" + INSTRUCTION


def build_eval_messages(question: str):
    """评估用 chat 消息，与 verl 把 "<image>文本" 切分成 [image, text] 的结果一致。"""
    return [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": clean_question(question) + "\n" + INSTRUCTION},
    ]}]


# ---------------------------------------------------------------------------
# 图像
# ---------------------------------------------------------------------------
def image_bytes_of(obj):
    """把 PIL.Image / {'bytes':..} / bytes / base64 字符串统一成原始字节。"""
    if obj is None:
        return None
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if isinstance(obj, dict):
        if obj.get("bytes"):
            return bytes(obj["bytes"])
        if obj.get("path") and os.path.isfile(obj["path"]):
            return Path(obj["path"]).read_bytes()
        return None
    if isinstance(obj, str):
        try:
            return base64.b64decode(obj)
        except Exception:
            return None
    if hasattr(obj, "save"):  # PIL.Image
        buf = io.BytesIO()
        obj.save(buf, format="PNG")
        return buf.getvalue()
    return None


def open_image(data: bytes):
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    img.load()
    return img.convert("RGB")


def png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def average_hash(img, size: int = 16) -> str:
    """16x16 均值哈希（256 位），用于发现同图不同编码的重复题（跨划分泄漏检查）。
    geometry3k 有大量题干相同（如 "Find x."）而图不同的题，8x8 哈希会误判为重复，所以用 16x16。"""
    g = img.convert("L").resize((size, size))
    px = list(g.get_flattened_data() if hasattr(g, "get_flattened_data") else g.getdata())
    mean = sum(px) / len(px)
    bits = "".join("1" if p >= mean else "0" for p in px)
    return "%0*x" % (size * size // 4, int(bits, 2))


# ---------------------------------------------------------------------------
# 记录
# ---------------------------------------------------------------------------
def to_verl_record(problem, answer, image_bytes, split, index, data_source=DATA_SOURCE, extra=None):
    info = {"split": split, "index": int(index), "answer": str(answer), "question": clean_question(problem)}
    info.update(extra or {})
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": build_prompt_text(problem)}],
        "images": [{"bytes": image_bytes, "path": None}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer)},
        "extra_info": info,
    }


def record_image_bytes(rec):
    imgs = rec.get("images") or []
    return image_bytes_of(imgs[0]) if len(imgs) else None


# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------
def _has(module):
    try:
        __import__(module)
        return True
    except Exception:
        return False


def write_split(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix == ".jsonl":
        with tmp.open("w", encoding="utf-8") as f:
            for r in records:
                r = dict(r)
                r["images"] = [{"b64": base64.b64encode(image_bytes_of(i) or b"").decode(), "path": None}
                               for i in r.get("images") or []]
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    elif _has("datasets"):
        import datasets
        ds = datasets.Dataset.from_list(list(records))
        try:
            ds = ds.cast_column("images", datasets.Sequence(datasets.Image()))
        except Exception:
            pass
        ds.to_parquet(str(tmp))
    else:
        import pandas as pd
        pd.DataFrame(list(records)).to_parquet(str(tmp), index=False)
    os.replace(tmp, path)
    return path


def read_split(path):
    """返回 list[dict]（verl 记录格式，images 中为 {'bytes':...}）。"""
    path = Path(path)
    if path.suffix == ".jsonl":
        out = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                r["images"] = [{"bytes": base64.b64decode(i.get("b64", "")), "path": None} for i in r.get("images") or []]
                out.append(r)
        return out
    if _has("datasets"):
        import datasets
        ds = datasets.Dataset.from_parquet(str(path))
        if "images" in ds.column_names:
            try:  # 存成 Image 特征时取原始字节，避免逐张解码
                ds = ds.cast_column("images", datasets.Sequence(datasets.Image(decode=False)))
            except Exception:
                pass
        return [dict(r) for r in ds]
    import pandas as pd
    df = pd.read_parquet(str(path))
    out = []
    for r in df.to_dict(orient="records"):
        r["images"] = [dict(i) if isinstance(i, dict) else {"bytes": image_bytes_of(i)} for i in list(r.get("images", []))]
        out.append(r)
    return out


def split_ext():
    """正式运行用 parquet；缺少 datasets 和 pyarrow 时退回 jsonl（干跑）。"""
    return ".parquet" if (_has("datasets") or _has("pyarrow")) else ".jsonl"


# ---------------------------------------------------------------------------
# 原始数据加载（Hub 名称 / load_from_disk / parquet 目录 / 合成 jsonl 目录）
# ---------------------------------------------------------------------------
RAW_SPLITS = ("train", "validation", "test")


def load_raw(source):
    """返回 {split: list[{'problem','answer','image'(bytes)}]}，split 为 train/validation/test。"""
    src = Path(str(source))
    if src.is_dir() and any((src / f"{s}.jsonl").exists() for s in RAW_SPLITS):
        out = {}
        for s in RAW_SPLITS:
            p = src / f"{s}.jsonl"
            if p.exists():
                rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
                out[s] = [{"problem": r["problem"], "answer": r["answer"],
                           "image": base64.b64decode(r["image_b64"]) if r.get("image_b64") else None} for r in rows]
        return out
    import datasets
    ds = None
    errors = []
    loaders = []
    if src.is_dir():
        loaders.append(lambda: datasets.load_from_disk(str(src)))
        files = {s: str(src / f"{s}.parquet") for s in RAW_SPLITS if (src / f"{s}.parquet").exists()}
        if files:
            loaders.append(lambda: datasets.load_dataset("parquet", data_files=files))
    loaders.append(lambda: datasets.load_dataset(str(source)))
    for fn in loaders:
        try:
            ds = fn()
            if isinstance(ds, datasets.DatasetDict):
                break
        except Exception as e:  # noqa
            errors.append(f"{type(e).__name__}: {e}")
            ds = None
    if ds is None:
        raise RuntimeError(f"无法加载 geometry3k: {source}\n" + "\n".join(errors))
    out = {}
    for s in RAW_SPLITS:
        if s not in ds:
            continue
        part = ds[s]
        try:
            part = part.cast_column("images", datasets.Sequence(datasets.Image(decode=False)))
        except Exception:
            pass
        rows = []
        for r in part:
            imgs = r.get("images") or []
            rows.append({"problem": r["problem"], "answer": r["answer"],
                         "image": image_bytes_of(imgs[0]) if imgs else None})
        out[s] = rows
    return out


# ---------------------------------------------------------------------------
# 合成数据（干跑 / 单元测试）：带有若干刻意注入的缺陷，用来检验清洗逻辑
# ---------------------------------------------------------------------------
def make_synthetic_raw(out_dir, n_train=64, n_val=16, n_test=24, seed=0):
    from PIL import Image, ImageDraw
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def draw(a, b, salt):
        img = Image.new("RGB", (160 + salt % 40, 120), "white")
        d = ImageDraw.Draw(img)
        d.polygon([(10, 110), (150, 110), (10 + salt % 100, 10)], outline="black")
        d.text((60, 112 - 14), f"{a}", fill="black")
        d.text((12, 50), f"{b}+{salt}", fill="black")
        return png_bytes(img)

    def item(i):
        a, b = rng.randint(2, 30), rng.randint(2, 30)
        kind = i % 4
        if kind == 0:
            return f"<image>Find x. Side {a}, side {b}.", str(a + b), draw(a, b, i)
        if kind == 1:
            return f"<image>Find the area. a={a}", f"{a} \\sqrt {{ 3 }}", draw(a, b, i)
        if kind == 2:
            return f"<image>Find y. ratio {a}/{b}", f"\\frac {{ {a} }} {{ {b} }}", draw(a, b, i)
        return f"<image>Find m \\angle 1. {a}", str(a * 3), draw(a, b, i)

    splits = {"train": [item(i) for i in range(n_train)],
              "validation": [item(1000 + i) for i in range(n_val)],
              "test": [item(2000 + i) for i in range(n_test)]}
    tr = splits["train"]
    # 注入缺陷：训练集内重复、训练/测试泄漏、空答案、损坏图像、缺少 <image>、无法校验的答案
    tr.append(tr[0])
    tr.append(splits["test"][0])
    tr.append(("<image>Find x.", "", draw(1, 1, 7)))
    tr.append(("<image>Find z.", "5", b"not-an-image"))
    tr.append(("Find w without tag.", "7", draw(2, 2, 9)))
    tr.append(("<image>Which option?", "\\mathrm{??}", draw(3, 3, 11)))
    for s, rows in splits.items():
        with (out_dir / f"{s}.jsonl").open("w", encoding="utf-8") as f:
            for p, a, img in rows:
                f.write(json.dumps({"problem": p, "answer": a,
                                    "image_b64": base64.b64encode(img).decode()}, ensure_ascii=False) + "\n")
    return out_dir

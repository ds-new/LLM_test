"""M6 step 1: download Qwen3.5-0.8B (the exact revision the edits were made on) and the benchmarks, with curl.

  python fetch.py model  <dir>      # every file of the repo at revision REV, resumable; complete = sizes match the API
  python fetch.py bench  <dir>      # MATH-500 and GSM8K test as jsonl, in the authors' row format

curl instead of huggingface_hub / datasets: on AutoDL images where `pip install -U vllm` pulled a newer httpx2, the
hub client's decompression breaks ("process() takes no keyword arguments"), and large files stalled behind the academic
proxy. curl asks for no compression, resumes partial files (-C -) and retries. hf-mirror is domestic: no proxy.
Base tensors are hash-checked later by build.py, so a wrong file cannot slip through.
"""
import json, os, subprocess, sys

REPO, REV = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
EP = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")


def curl(url, dst=None, tries=20):
    cmd = ["curl", "-fL", "--retry", str(tries), "--retry-all-errors", "--retry-delay", "5", "--connect-timeout", "20",
           "--speed-limit", "100000", "--speed-time", "60", "-sS"]
    if dst:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        for attempt in range(tries):   # --speed-limit aborts a stalled transfer; -C - resumes it
            r = subprocess.run(cmd + ["-C", "-", "-o", dst, url])
            if r.returncode in (0, 33):   # 33: server does not support ranges (file already complete)
                return
            print(f"curl rc {r.returncode} for {url}, attempt {attempt + 1}; resuming", flush=True)
        raise RuntimeError(f"download failed: {url}")
    return subprocess.run(cmd + [url], capture_output=True, text=True, check=True).stdout


def model(out):
    info = json.loads(curl(f"{EP}/api/models/{REPO}/revision/{REV}?blobs=true"))
    files = [(s["rfilename"], s.get("size") or (s.get("lfs") or {}).get("size")) for s in info["siblings"]]
    print(f"{REPO}@{REV[:8]}: {len(files)} files, {sum(x or 0 for _, x in files) / 1e9:.2f} GB", flush=True)
    for name, size in files:
        dst = os.path.join(out, name)
        if size and os.path.exists(dst) and os.path.getsize(dst) == size:
            continue
        if os.path.exists(dst) and size and os.path.getsize(dst) > size:
            os.remove(dst)
        print(f"get {name} ({(size or 0) / 1e6:.1f} MB)", flush=True)
        ms = f"https://www.modelscope.cn/models/{REPO}/resolve/master/{name}"
        if size and size > 100e6 and os.environ.get("M6_MODELSCOPE", "1") == "1":   # ModelScope is ~10x faster on AutoDL
            try:                                                     # (a partial file is resumed with -C -)
                curl(ms, dst, tries=3)
            except Exception as e:  # noqa: BLE001
                print("modelscope failed:", e, flush=True)
            if not (os.path.exists(dst) and os.path.getsize(dst) == size):
                print("modelscope copy unusable (size differs from the HF revision); falling back to", EP, flush=True)
                if os.path.exists(dst):
                    os.remove(dst)
        if not (size and os.path.exists(dst) and os.path.getsize(dst) == size):
            curl(f"{EP}/{REPO}/resolve/{REV}/{name}", dst)
        if size:
            assert os.path.getsize(dst) == size, f"{name}: size {os.path.getsize(dst)} != {size}"
    json.dump({"source": EP, "repo": REPO, "revision": REV, "files": len(files)},
              open(os.path.join(out, "_source.json"), "w"))
    print("model complete", flush=True)


def bench(out):
    os.makedirs(out, exist_ok=True)

    def dump(name, rows):
        with open(os.path.join(out, f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rows), flush=True)

    raw = os.path.join(out, "_raw")
    if not os.path.exists(os.path.join(out, "math500.jsonl")):
        p = os.path.join(raw, "math500_test.jsonl")
        curl(f"{EP}/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl", p)
        dump("math500", [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]}
                         for r in map(json.loads, open(p))])
    if not os.path.exists(os.path.join(out, "gsm8k.jsonl")):
        p = os.path.join(raw, "gsm8k_test.parquet")
        curl(f"{EP}/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet", p)
        import pandas as pd
        df = pd.read_parquet(p)
        dump("gsm8k", [{"id": f"gsm8k_test_{i}", "problem": q, "answer": a.split("####")[-1].strip().replace(",", "")}
                       for i, (q, a) in enumerate(zip(df["question"], df["answer"]))])


if __name__ == "__main__":
    {"model": model, "bench": bench}[sys.argv[1]](sys.argv[2])

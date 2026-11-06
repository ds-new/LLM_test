"""M6 step 2: build the evaluated checkpoints from the public release, bit for bit.

  python build.py <release_dir> <out_root> [p50c p10c p25c]

<out_root>/base  : the release as is (safetensors linked), with the authors' chat template
<out_root>/<run> : the release with the run's edited tensors rebuilt by the authors' code (apply_edit.rebuild) and every
                   base and edited tensor checked against the sha256 in the run's manifest. Works for a single
                   model.safetensors or a sharded release.
"""
import glob, hashlib, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "authors"))
import st_numpy as S  # noqa: E402
from apply_edit import rebuild  # noqa: E402

import numpy as np  # noqa: E402

TEMPLATE = os.path.join(HERE, "authors", "Qwen3.5-0.8B-text", "chat_template.jinja")


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def link_tree(src, dst, skip=()):
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f in skip or f.startswith("."):
            continue
        s, d = os.path.join(src, f), os.path.join(dst, f)
        if os.path.isfile(s) and not os.path.exists(d):
            os.symlink(os.path.abspath(s), d)
    shutil.copyfile(TEMPLATE, os.path.join(dst, "chat_template.jinja"))   # the exact template of the evaluation


def build(release, root, runs):
    shards = sorted(glob.glob(os.path.join(release, "*.safetensors")))
    assert shards, f"no safetensors in {release}"
    link_tree(release, os.path.join(root, "base"), skip=("chat_template.jinja",))
    report = {}
    for run in runs:
        edir = os.path.join(HERE, "authors", "edits", run)
        man = json.load(open(os.path.join(edir, "manifest.json")))
        z = np.load(os.path.join(edir, "edit.npz"))
        want = {t["name"]: (i, t) for i, t in enumerate(man["tensors"])}
        out = os.path.join(root, run)
        done = {}
        changed_shards = set()
        for sh in shards:
            head, _ = S.read_header(sh)
            if want.keys() & head.keys():
                changed_shards.add(os.path.basename(sh))
        link_tree(release, out, skip=tuple(changed_shards) + ("chat_template.jinja",))
        for sh in shards:
            if os.path.basename(sh) not in changed_shards:
                continue
            tensors = S.load(sh)
            for name, (i, t) in want.items():
                if name not in tensors:
                    continue
                dt, b = tensors[name]
                assert dt == "BF16" and list(b.shape) == t["shape"], (name, dt, b.shape)
                assert sha(b) == t["base_sha256"], f"{name}: base tensor differs from the one the edit was made on " \
                                                   f"(wrong model revision?)"
                e = rebuild(b, t, z, i)
                assert sha(e) == t["edited_sha256"], f"{name}: rebuilt tensor does not match the evaluated one"
                tensors[name] = ("BF16", e)
                done[name] = int(t["changed_entries"])
            dst = os.path.join(out, os.path.basename(sh))
            if os.path.islink(dst):
                os.unlink(dst)
            S.save(dst, tensors, {"format": "pt"})
        missing = set(want) - set(done)
        assert not missing, f"{run}: tensors not found in the release: {sorted(missing)[:3]}"
        report[run] = {"tensors": len(done), "changed_entries": sum(done.values()), "hash_checked": True}
        print(f"{run}: {len(done)} tensors rebuilt, {sum(done.values()):,} entries changed, all sha256 match")
    json.dump(report, open(os.path.join(root, "build_report.json"), "w"), indent=1)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], sys.argv[3:] or ["p50c"])

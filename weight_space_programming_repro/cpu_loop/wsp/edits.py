"""The small "edits" language the subject writes weights in (exp5; the same shape of language as exp4).

    ```edits
    W1[12, 0] = 2.5
    W1[0:6, 3] = [1, 0, -1, 0, 2, 0]
    b1[0:4] = -1
    W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]
    W3[:, 2] *= 2
    ```
Only the LAST edits block of a reply is read; statements apply in order; a statement that cannot be read is skipped.
"""
import ast, re
import numpy as np

BLOCK = re.compile(r"```edits[ \t]*\n(.*?)(?:```|\Z)", re.S)
STMT = re.compile(r"^([A-Za-z_][\w\.]*)\s*\[([^\]]*)\]\s*(=|\*=|\+=|-=)\s*(.+)$", re.S)
MAX_ABS = 1e6


def _index(s, shape):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != len(shape):
        raise ValueError(f"{len(parts)} indices for a {len(shape)}-D tensor")
    idx = []
    for p, n in zip(parts, shape):
        if ":" in p:
            a = [x.strip() for x in p.split(":")]
            if len(a) > 3:
                raise ValueError(f"bad range {p}")
            a += [""] * (3 - len(a))
            idx.append(slice(*[int(x) if x else None for x in a]))
        else:
            i = int(p)
            if not -n <= i < n:
                raise IndexError(f"index {i} out of range {n}")
            idx.append(i)
    return tuple(idx)


def statements(block):
    """Split a block into statements: '#' comments removed, a statement continues while brackets are open."""
    out, buf, depth = [], "", 0
    for line in block.splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line.strip() and depth == 0:
            continue
        buf = (buf + " " + line.strip()) if buf else line.strip()
        depth = buf.count("[") - buf.count("]")
        if depth <= 0:
            out.append(buf)
            buf, depth = "", 0
    if buf:
        out.append(buf)
    return out


def apply_edits(reply, tensors):
    """Apply the last edits block of `reply` to `tensors` (dict name -> float32 array, changed in place)."""
    blocks = BLOCK.findall(reply)
    rep = {"block": bool(blocks), "statements": 0, "applied": 0, "errors": [], "tensors_touched": set()}
    if not blocks:
        return rep
    for st in statements(blocks[-1]):
        rep["statements"] += 1
        try:
            m = STMT.match(st)
            if not m:
                raise ValueError("not a statement")
            name, idx_s, op, val_s = m.groups()
            if name not in tensors:
                raise KeyError(f"unknown tensor {name}")
            t = tensors[name]
            idx = _index(idx_s, t.shape)
            v = np.asarray(ast.literal_eval(val_s.strip()), dtype=np.float64)
            if not np.all(np.isfinite(v)) or np.any(np.abs(v) > MAX_ABS):
                raise ValueError("value not finite or |v| > 1e6")
            sel = t[idx]
            if v.ndim and np.shape(sel) != v.shape:
                if v.size == np.size(sel):
                    v = v.reshape(np.shape(sel))
                else:
                    raise ValueError(f"{v.size} values for {np.size(sel)} entries")
            if op == "=":
                t[idx] = v
            elif op == "*=":
                t[idx] = sel * v
            elif op == "+=":
                t[idx] = sel + v
            else:
                t[idx] = sel - v
            rep["applied"] += 1
            rep["tensors_touched"].add(name)
        except Exception as e:  # a statement that cannot be read is skipped; the others still apply
            rep["errors"].append(f"{st[:120]} -> {type(e).__name__}: {e}")
    rep["tensors_touched"] = sorted(rep["tensors_touched"])
    return rep

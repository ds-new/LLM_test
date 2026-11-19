
p = "/root/autodl-tmp/r1zero/ma/loop.py"
s = open(p).read()
old = '''    shutdown(board, reason, delay=60)
    return 0 if status == "completed" else 1'''
new = '''    if status == "completed":
        shutdown(board, reason, delay=60)
    else:
        # 报错停止：不立即关机，交给 watchdog —— 10 分钟内没人开始修复（run/REPAIRING）才关机
        board.post(NAME, "failure_grace", {"reason": reason, "shutdown_in_min": 10, "hold_file": str(RUN / "REPAIRING")})
    return 0 if status == "completed" else 1'''
if "failure_grace" not in s:
    assert old in s
    s = s.replace(old, new); open(p, "w").write(s)
print("loop patched")

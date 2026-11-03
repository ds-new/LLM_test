#!/usr/bin/env bash
# 捕获 Python 的实际退出码；系统杀进程时仍保留前后 cgroup 计数。
set +e
set -uo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1
PYTHON="${PYTHON:-python3}"
mkdir -p output/logs || exit 1
TOKENIZER_LOG="output/logs/bbpe-train-$(date +%Y%m%d-%H%M%S)-$$.log"
{
    echo "时间：$(date)"
    echo "解释器：$PYTHON"
    if command -v free >/dev/null 2>&1; then free -h; fi
    df -h .
    echo "训练前资源快照："
    "$PYTHON" -S tokenizer_training/runtime_info.py
} >"$TOKENIZER_LOG" 2>&1
echo "完整日志：${TOKENIZER_LOG}"
"$PYTHON" -S -X faulthandler -u tokenizer_training/train_tokenizer.py "$@" 2>&1 | tee -a "$TOKENIZER_LOG"
TOKENIZER_STATUSES=("${PIPESTATUS[@]}")
TOKENIZER_EXIT="${TOKENIZER_STATUSES[0]}"
{
    echo "训练后资源快照（独立进程；训练进程 RSS 见上方训练日志）："
    "$PYTHON" -S tokenizer_training/runtime_info.py
    echo "训练 Python 退出码：${TOKENIZER_EXIT}；日志：${TOKENIZER_LOG}"
} | tee -a "$TOKENIZER_LOG"
if [[ "$TOKENIZER_EXIT" != 0 ]]; then
    if [[ "$TOKENIZER_EXIT" == 137 ]]; then
        echo "Python 收到 SIGKILL；检查日志中 memory.events 的 oom_kill 变化及系统日志，不能仅凭退出码认定 OOM。" >&2
    fi
    echo "训练未成功完成，暂勿运行 inspect_tokenizer.py。" >&2
    exit "$TOKENIZER_EXIT"
fi
if [[ "${TOKENIZER_STATUSES[1]}" != 0 ]]; then
    echo "训练日志写入失败，请检查磁盘空间。" >&2
    exit 1
fi

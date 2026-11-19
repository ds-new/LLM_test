#!/usr/bin/env bash
# AutoDL 一键准备并启动多智能体预训练（在“有卡模式”下运行）。
#   cd /root/autodl-tmp/verl_megatron_sglang && bash decoder_from_scratch/a800/autodl/setup.sh
# 可选：NO_START=1 只做检查不启动；NO_SHUTDOWN=1 结束后不关机（调试用）。
set -euo pipefail
cd "$(dirname "$0")/../../.."
PROJECT=$(pwd)
RUN=${A8_RUN_DIR:-/root/autodl-tmp/a800_pretrain_v1}
echo "== 工程目录 $PROJECT，运行目录 $RUN"
mkdir -p "$RUN/logs" /root/a800_scratch

echo "== GPU / 磁盘 / 内存"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || { echo "没有 GPU：请先在 AutoDL 控制台切换到有卡模式"; exit 1; }
df -h /root/autodl-tmp / | cat
free -g | head -2; echo "CPU 核数 $(nproc)"

echo "== Python 依赖（独立 venv：继承系统包，只把 numpy 升到 2.x，修复 torch.from_numpy 与 numpy 1.26 不兼容）"
VENV=/root/a800_venv
[ -x "$VENV/bin/python" ] || python -m venv --system-site-packages "$VENV"
export PYTHON="$VENV/bin/python"
"$PYTHON" -c "import numpy,sys; sys.exit(0 if int(numpy.__version__.split('.')[0]) >= 2 else 1)" || "$VENV/bin/pip" install -q "numpy>=2.1,<2.4"
"$PYTHON" -c "import torch,numpy; torch.from_numpy(numpy.zeros(2))"
"$PYTHON" -c "import torch,sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda); assert torch.cuda.is_available()"
"$PYTHON" -c "import pyarrow, tokenizers" 2>/dev/null || "$VENV/bin/pip" install -q pyarrow tokenizers
"$PYTHON" -c "import numpy, pyarrow, tokenizers; print('numpy', numpy.__version__, 'pyarrow', pyarrow.__version__, 'tokenizers', tokenizers.__version__)"

echo "== 自检（清洗、学习率、采样器、ModelScope 连通性）"
A8_RUN_DIR=$(mktemp -d) "$PYTHON" -m decoder_from_scratch.a800.selfcheck

if [ "${NO_SHUTDOWN:-0}" = "1" ]; then touch "$RUN/NO_SHUTDOWN"; echo "已放置 NO_SHUTDOWN：结束后不关机"; fi
if [ "${NO_START:-0}" = "1" ]; then echo "NO_START=1，未启动"; exit 0; fi
if [ -f "$RUN/supervisor.pid" ] && kill -0 "$(cat "$RUN/supervisor.pid")" 2>/dev/null; then
  echo "守护脚本已在运行（pid $(cat "$RUN/supervisor.pid")）"; exit 0
fi
rm -f "$RUN/STOP"
PYTHON="$PYTHON" nohup bash decoder_from_scratch/a800/run_loop.sh >/dev/null 2>&1 &
sleep 5
echo "== 已启动守护脚本 pid $(cat "$RUN/supervisor.pid" 2>/dev/null)"
echo "查看进度：$PYTHON -m decoder_from_scratch.a800.loop status"
echo "实时日志：tail -f $RUN/logs/loop.log"
echo "可读报告：cat $RUN/REPORT.md"

"""路线衔接启动器（由守护脚本调用，也可手动运行）：
  smoke [micro 缩小倍数]：路线 A 冒烟测试——用极小规模走完 数据 → 第 1 段（含中途检查探针）→ 重启 → 数据 → 第 2 段
                         （含检查点平均）→ 最终评估 → 结果分析。GPU 被别的任务占用时传 4，把 micro batch 缩小 4 倍。
  after_ab：第一批消融（学习率扫描 + 多种子）完成后 → 清理对照 / 消融的 token 数据 → 冒烟测试 → 正式启动路线 A；
            对照结论为"v2 明显更好"且差异超过种子噪声时，路线 A 开启中途效果检查（不达标自动转路线 B）。
  route_b ：路线 A 中途检查不达标时 → 释放路线 A 的数据 → 准备路线 B 运行目录（v2 tokenizer + v1 的结构与数据来源）→ 启动。
返回码非 0 表示失败。
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path('/root/autodl-tmp/verl_megatron_sglang')
PY = '/root/a800_venv/bin/python'
AB = Path('/root/autodl-tmp/a800_tok_ab')
ABL = Path('/root/autodl-tmp/a800_ablate')
V1 = Path('/root/autodl-tmp/a800_pretrain_v1')
CPT = Path('/root/autodl-tmp/a800_cpt_v1')
SMOKE = Path('/root/autodl-tmp/cpt_smoke')
SMOKE_SCRATCH = Path('/root/a800_scratch/cpt_smoke')
ROUTE_B = Path('/root/autodl-tmp/a800_pretrain_v2b')
STEP_TOKENS = 524288
OFF = {'weight_edit': {'enabled': False}, 'token_probe': {'enabled': False}, 'profile': {'enabled': False}}


def log(msg):
    print(time.strftime('%F %T'), msg, flush=True)


def env_for(run_dir, scratch):
    env = dict(os.environ)
    env.pop('OMP_NUM_THREADS', None)
    env.update(A8_RUN_DIR=str(run_dir), A8_SCRATCH_DIR=str(scratch), PYTHON=PY)
    return env


def launch_supervisor(run_dir, scratch):
    subprocess.Popen(['bash', 'decoder_from_scratch/a800/run_loop.sh'], cwd=PROJECT, env=env_for(run_dir, scratch),
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, start_new_session=True)


def write_config(run_dir, cfg):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def smoke(micro_div=1):
    shutil.rmtree(SMOKE, ignore_errors=True)
    shutil.rmtree(SMOKE_SCRATCH, ignore_errors=True)
    s1, s2 = {'tokens': 2 * STEP_TOKENS}, {'tokens': 3 * STEP_TOKENS}
    if micro_div > 1:
        s1.update(micro=32 // micro_div, accum=16 * micro_div)
        s2.update(micro=8 // micro_div, accum=16 * micro_div)
    write_config(SMOKE, {'mode': 'cpt', **OFF, 'passk': {'enabled': False},
                         'train': {'max_train_hours': 200, 'segment_minutes': 0.25},
                         'cpt': {'stage1': s1, 'stage2': s2, 'workers': 4, 'delete_v1_latest': False, 'avg_last_steps': 10,
                                 'midcheck': {'enabled': True, 'at_tokens': STEP_TOKENS, 'probe_tokens': 2 * STEP_TOKENS,
                                              'min_gain': 0.005, 'action': 'record'}}})
    log(f'路线 A 冒烟测试开始（micro 缩小 {micro_div} 倍）')
    rc = None
    with open(SMOKE / 'smoke.log', 'a') as out:
        for _ in range(6):  # 退出码 75 = 阶段之间请求重启（释放显存），与守护脚本的行为一致
            rc = subprocess.run([PY, '-u', '-m', 'decoder_from_scratch.a800.loop', 'run'], cwd=PROJECT,
                                env=env_for(SMOKE, SMOKE_SCRATCH), stdout=out, stderr=subprocess.STDOUT, timeout=3 * 3600).returncode
            if rc != 75:
                break
    st = read(SMOKE / 'state.json', {})
    avg = st.get('ckpt_avg') or {}
    checks = {'returncode_0': rc == 0, 'phase_done': st.get('phase') == 'done',
              'final_analysis': (SMOKE / 'analysis' / 'FINAL_ANALYSIS.md').exists(),
              'midcheck': (st.get('midcheck') or {}).get('probe_val') is not None,
              'long_context': (st.get('long_context') or {}).get('val_4096') is not None,
              'ckpt_avg': 'val_avg' in avg, 'baseline': (st.get('baseline') or {}).get('val_loss') is not None}
    log('冒烟测试检查：' + json.dumps(checks, ensure_ascii=False))
    log(f'  baseline={st.get("baseline")} midcheck={(st.get("midcheck") or {}).get("probe_val")} '
        f'final={(st.get("final") or {}).get("val_loss")} long_context={st.get("long_context")} ckpt_avg={avg}')
    ok = all(checks.values())
    if ok:
        shutil.rmtree(SMOKE, ignore_errors=True)
        shutil.rmtree(SMOKE_SCRATCH, ignore_errors=True)
    else:
        log('冒烟测试失败，保留 ' + str(SMOKE) + ' 供排查（last_error：' + str((st.get('last_error') or {}).get('error')) + '）')
    return ok


def after_ab():
    verdict = ((read(AB / 'state.json', {}).get('results') or {}).get('report') or {}).get('verdict', '')
    summ = read(ABL / 'stage1_summary.json')
    v2_better = '明显更好' in verdict and (summ is None or summ.get('v2_significant') is not False)
    log(f'对照结论：{verdict}；种子噪声检验：{None if summ is None else summ.get("v2_significant")}；'
        f'路线 A 中途检查：{"开启" if v2_better else "关闭"}')
    for path in (AB / 'data' / 'tokens', AB / 'scratch', ABL / 'scratch'):
        shutil.rmtree(path, ignore_errors=True)
    log('已清理对照实验 / 第一批消融的 token 数据与缓存（保留报告、tokenizer、对照模型、receipts）')
    if not smoke():
        return 1
    write_config(CPT, {'mode': 'cpt', **OFF, 'passk': {'enabled': True}, 'analysis': {'enabled': True},
                       'train': {'max_train_hours': 200, 'segment_minutes': 60, 'checkpoint_minutes': 30},
                       'cpt': {'delete_v1_latest': True,
                               'midcheck': {'enabled': v2_better, 'at_tokens': 2_000_000_000, 'probe_tokens': 500_000_000,
                                            'min_gain': 0.005, 'action': 'route_b'}}})
    launch_supervisor(CPT, Path('/root/a800_scratch'))
    log('路线 A 已启动：' + str(CPT))
    return 0


def route_b():
    log('路线 B 准备开始：释放路线 A 第 1 段数据（检查点保留，可随时恢复路线 A）')
    shutil.rmtree(CPT / 'data' / 'stage1' / 'tokens', ignore_errors=True)
    (Path('/root/a800_scratch/cpt_stage1') / 'prev.pt').unlink(missing_ok=True)
    rb = ROUTE_B
    (rb / 'tokenizer').mkdir(parents=True, exist_ok=True)
    (rb / 'data').mkdir(parents=True, exist_ok=True)
    for name in ('tokenizer_hf.json', 'bbpe.json'):
        shutil.copy2(AB / 'tokenizer_v2.json', rb / 'tokenizer' / name)
    shutil.copy2(V1 / 'data' / 'file_lists.json', rb / 'data' / 'file_lists.json')
    write_config(rb, read(V1 / 'config.json'))
    v1st = read(V1 / 'state.json')
    board = {'started': time.time(), 'started_text': time.strftime('%F %T'), 'phase': 'build_data', 'decisions': 0, 'route': 'B',
             'selftest': v1st.get('selftest'), 'benchmark': v1st.get('benchmark'), 'model_choice': v1st.get('model_choice'),
             'compile_disabled': v1st.get('compile_disabled', False),
             'tokenizer': {'name': 'v2', 'vocab_size': 32000, 'source': str(AB / 'tokenizer_v2.json'),
                           'note': 'Qwen2 风格预切分（数字逐位、前导空格并入单词），字节级 BPE'},
             'fast_tokenizer': {'use_fast': True, 'note': 'HF 原生 tokenizer，无需等价性检查'}}
    (rb / 'state.json').write_text(json.dumps(board, ensure_ascii=False, indent=2), encoding='utf-8')
    if (CPT / 'NEXT_ON_DONE.sh').exists():  # 排在路线 A 之后的任务改排到路线 B 之后
        shutil.copy2(CPT / 'NEXT_ON_DONE.sh', rb / 'NEXT_ON_DONE.sh')
    launch_supervisor(rb, Path('/root/a800_scratch/route_b'))
    log('路线 B 已启动：' + str(rb))
    return 0


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'smoke':
        sys.exit(0 if smoke(int(sys.argv[2]) if len(sys.argv) > 2 else 1) else 1)
    sys.exit({'after_ab': after_ab, 'route_b': route_b}[cmd]())

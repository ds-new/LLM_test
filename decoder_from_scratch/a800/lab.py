"""实验室总调度（第二、三批消融）：路线 A 结束后由 NEXT_ON_DONE.sh 启动，run_lab.sh 守护。

顺序按 11-17 确认的排期（plan/07），11-18 按用户要求精简（完整版备份在 lab.py.bak_full）：
  1. μP：坐标检查 → 宽度扫描（8 层，宽 128 / 256 / 512，SP 与 μP 各 7 档学习率，3 亿 token，cptmix 数据）
     → 深度 16、宽 256 的基准扫描（proxy）→ 1.25 亿参数（宽 768）迁移验证（×0.5 / ×2；×1 取自 scaling law）。
  2. scaling law：同一族结构宽 320 / 512 / 768（25% / 50% / 100% 预算，100% = 20 token/参数）+ 宽 1152（只做 25% / 50%），
     学习率用 μP 迁移；v1、路线 A 的参照值。
  3. 数据工程（代理模型 d768_l16、SP、学习率 1e-3、25 亿 token）：路线 A 退火配比；中文教育 4-5 分对 3-4 分；
     重复率高的来源做去重对照。基准直接取调度块的同配置分支。
  4. 学习率调度：恒定学习率主线（40 亿）+ 冷却分支（四个预算、三种形状）；余弦 ×3；全程线性到 0；Schedule-Free；
     seed 2 在 12.5 亿上复核。
  5. Muon / SOAP：12.5 亿 token 扫学习率 → 各自最优续到 25 亿；AdamW 的尺子来自第 4 块。
每个实验组在独立子进程里训练（隔离显存和编译缓存）；失败重试一次，仍失败记为 failed，依赖它的实验记为 skipped，继续后面的。
里程碑检查点放在内存盘（/dev/shm），用完即删；机器重启丢失时自动重跑对应的主线。除 proxy 外的数据集，用到它的实验全部结束后删除分片。
每完成一组就重写 REPORT.md。
"""

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

from .common import Bus, FatalError, StopRequested, control_signal, heartbeat, read_json, stamp, write_json, write_text
from . import lab_data
from .lab_core import LAB, LAB_CKPT, LAB_SCRATCH, PROXY_SHAPE, family_shape, milestone_path, param_count, run_dir

PROJECT = Path('/root/autodl-tmp/verl_megatron_sglang')
PY = os.environ.get('PYTHON', '/root/a800_venv/bin/python')
G = 1_000_000_000
TPS = 1024 * 64 * 4
LR_GRID = [2.0 ** -k for k in range(12, 5, -1)]
FAMILY = (320, 512, 768, 1152)
PHASES = ['prepare', 'experiments', 'report', 'done']
FINISHED = ('done', 'failed', 'skipped')
ABL_SUMMARY = Path('/root/autodl-tmp/a800_ablate/stage1_summary.json')
V1_MODEL = Path('/root/autodl-tmp/a800_pretrain_v1/model/final.pt')
CPT_MODEL = Path('/root/autodl-tmp/a800_cpt_v1/model/final.pt')
V1_SHAPE = {'width': 1024, 'depth': 40, 'heads': 16, 'kv_heads': 4, 'ffn': 2816}
V1_N, V1_D = 483_738_624, 25_000_000_000
MIX_ZH = {'zh_cci3': 0.35, 'zh_fwedu': 0.20, 'zh_industry': 0.15, 'en_fwedu': 0.11, 'en_cosmo': 0.04, 'code': 0.07, 'math': 0.05, 'arith': 0.03}
MIX_ANNEAL = {**{k: round(v * 0.97, 4) for k, v in lab_data.CPT_STAGE2_MIX.items()}, 'arith': 0.03}
Q_MIX = {'zh_fwedu': 0.30, **{k: round(0.67 * v / 0.86, 4) for k, v in lab_data.V1_MIX.items() if k != 'zh_fwedu'}, 'arith': 0.03}


class Ref:
    """运行时才能确定的值：best_lr（某些组里加权 bits/字节 最低的那组的学习率 × mult）、best_main（那组的主线名）。"""

    def __init__(self, kind, groups, mult=1.0):
        self.kind, self.groups, self.mult = kind, tuple([groups] if isinstance(groups, str) else groups), mult

    def __repr__(self):
        return f'Ref({self.kind}, {self.groups}, x{self.mult:g})'


class Pending(Exception):
    pass


class Unresolvable(Exception):
    pass


def T(name, block, group, **kw):
    j = {'type': 'train', 'name': name, 'block': block, 'group': group, 'seq_len': 1024, 'micro': 64, 'accum': 4, 'seed': 1234,
         'param': 'sp', 'shape': dict(PROXY_SHAPE), 'data': 'proxy', 'opt': {'name': 'adamw', 'lr': 1e-3, 'wd': 0.1},
         'eval': True, 'arith': False, 'compile': True}
    opt, lr, x = kw.pop('opt', None), kw.pop('lr', None), kw.pop('x', {})
    j.update(kw)
    if opt:
        j['opt'] = {**j['opt'], **opt}
    if lr is not None:
        j['opt'] = {**j['opt'], 'lr': lr}
    j['x'] = {**x, 'lr': j['opt']['lr']}
    return j


# ------------------------------------------------------------------ 实验清单 ----------
def family_budget(w):
    shape = family_shape(w)
    n = param_count(shape)
    B = 20 * n
    return shape, n, B, min(300, max(50, (B // TPS) // 10))


def sl_name(w, end):
    return f'sl_w{w}_D{end / G:.3g}B'


def transfer_point():
    """μP 迁移验证用 scaling law 里宽 768（m = 3）的 25% 预算：调度完全相同，×1 就是 scaling law 的那个分支。"""
    shape, n, B, warm = family_budget(768)
    p0 = B // 5
    return shape, p0, p0 * 5 // 4, warm


def plan_mup():
    """μP（精简版）：坐标检查 → 宽度扫描（8 层，宽 128 / 256 / 512，SP 与 μP 各 7 档学习率）→ 深度 16、宽 256 的基准扫描
    → 迁移验证：1.25 亿参数（宽 768）用基准最优学习率 ×0.5 / ×2 各跑一组，×1 取自 scaling law 同预算的分支。"""
    J = [{'type': 'coord', 'name': 'mup_coord', 'block': 'mup', 'group': 'mup_coord'}]
    for w in (128, 256, 512):
        for param in ('sp', 'mup'):
            for lr in LR_GRID:
                J.append(T(f'mupw_{param}_w{w}_lr{lr:.2e}', 'mup', f'mupw_{param}', data='cptmix', param=param, mup={'base_width': 256},
                           shape={'width': w, 'depth': 8}, lr=lr, tokens=300_000_000, x={'width': w, 'param': param}))
    for lr in LR_GRID:
        J.append(T(f'mup16_w256_lr{lr:.2e}', 'mup', 'mup16', param='mup', mup={'base_width': 256}, shape=family_shape(256),
                   lr=lr, tokens=390_000_000, x={'width': 256}))
    shape, p0, end, warm = transfer_point()
    for mult in (0.5, 2.0):
        J.append(T(f'mupx_w768_x{mult:g}', 'mup', 'mupx', param='mup', mup={'base_width': 256}, shape=shape,
                   lr=Ref('best_lr', 'mup16', mult), tokens=end,
                   sched={'kind': 'wsd', 'decay_from': p0, 'shape': 'linear', 'final': 0.1, 'warmup': warm}, x={'mult': mult}))
    return J


def plan_scaling():
    """scaling law（精简版）：宽 320 / 512 / 768 各 3 个预算（25% / 50% / 100%，100% = 20 token/参数），
    宽 1152（约 2.6 亿）只做 25% / 50%（它的 100% 一项就要约 22 小时）。学习率用 μP 迁移。"""
    J = []
    for w in FAMILY:
        shape, n, B, warm = family_budget(w)
        pts = [B // 5, 2 * B // 5] + ([] if w == 1152 else [4 * B // 5])
        main = f'sl_w{w}_main'
        J.append(T(main, 'scaling', 'sl_main', param='mup', mup={'base_width': 256}, shape=shape, lr=Ref('best_lr', 'mup16'),
                   tokens=pts[-1], sched={'kind': 'constant', 'warmup': warm}, milestones=pts, eval=False, eval_at_milestones=True,
                   x={'width': w, 'N': n}))
        for p in pts:
            end = p * 5 // 4
            J.append(T(sl_name(w, end), 'scaling', 'sl', param='mup', mup={'base_width': 256}, shape=shape,
                       lr=Ref('best_lr', 'mup16'), tokens=end, branch_from={'run': main, 'tokens': p},
                       sched={'kind': 'wsd', 'decay_from': p, 'shape': 'linear', 'final': 0.1, 'warmup': warm},
                       x={'width': w, 'N': n, 'D': end}))
    J.append({'type': 'ref', 'name': 'ref_v1', 'block': 'scaling', 'group': 'ref', 'model': 'v1'})
    J.append({'type': 'ref', 'name': 'ref_cpt', 'block': 'scaling', 'group': 'ref', 'model': 'cpt'})
    return J


def plan_data():
    """数据工程（精简版）：基准（v1 配比、WSD 最后 20% 降到 10%）与调度块的 sch_wsd_lin10_2.5B 完全同配置，直接取它，不重复跑。
    这里跑：路线 A 退火配比；中文教育 4-5 分对 3-4 分；重复率高的来源的去重对照。"""
    base = dict(arith=True, tokens=5 * G // 2)
    J = [T('de_mix_anneal', 'data', 'de_mix', weights=MIX_ANNEAL, x={'mix': '路线 A 退火配比'}, **base),
         T('de_q45', 'data', 'de_quality', data='zhfw45', weights=Q_MIX, x={'bucket': '4-5 分'}, **base),
         T('de_q34', 'data', 'de_quality', data='zhfw34', weights=Q_MIX, x={'bucket': '3-4 分'}, **base)]
    srcs, _ = lab_data.dedup_sources()
    for s in srcs:
        for v in ('raw', 'dedup'):
            J.append(T(f'de_dd_{s}_{v}', 'data', 'de_dedup', data=f'dd_{s}_{v}', tokens=G, x={'source': s, 'variant': v}))
    return J


def plan_sched():
    """学习率调度（精简版）：恒定学习率主线到 40 亿，分出四个预算的冷却（20%，线性到 0）和 25 亿下另两种形状；余弦到 10% ×3；
    全程线性到 0；Schedule-Free 一档；seed 2 在 12.5 亿预算上复核 WSD 对余弦。"""
    main, const = 'sch_main', {'kind': 'constant', 'warmup': 300}
    J = [T(main, 'sched', 'sch_main', tokens=4 * G, sched=const, milestones=[G // 2, G, 2 * G, 4 * G],
           eval=True, eval_at_milestones=True, x={'main': main})]

    def br(name, group, p, end, shape='linear', final=0.0, also=(), seed=1234, parent=main, arith=False, **x):
        return T(name, 'sched', group, tokens=end, seed=seed, branch_from={'run': parent, 'tokens': p}, also=list(also), arith=arith,
                 sched={'kind': 'wsd', 'decay_from': p, 'shape': shape, 'final': final, 'warmup': 300}, x={'main': parent, 'D': end, **x})

    for p in (G // 2, G, 2 * G, 4 * G):
        end = p * 5 // 4
        J.append(br(f'sch_wsd_lin0_{end / G:g}B', 'sch_budget', p, end, also=['opt_adamw_sweep'] if end == 5 * G // 4 else []))
    J.append(br('sch_wsd_lin10_2.5B', 'sch_shape', 2 * G, 5 * G // 2, final=0.1, arith=True, label='线性到 10%（也是数据工程块的基准）'))
    J.append(br('sch_wsd_sqrt0_2.5B', 'sch_shape', 2 * G, 5 * G // 2, shape='1-sqrt', label='1−√ 到 0'))
    for end in (5 * G // 8, 5 * G // 4, 5 * G // 2):
        J.append(T(f'sch_cos10_{end / G:g}B', 'sched', 'sch_cos', tokens=end, sched={'kind': 'cosine', 'final': 0.1, 'warmup': 300}, x={'D': end}))
    J.append(T('sch_d2z_2.5B', 'sched', 'sch_d2z', tokens=5 * G // 2, sched={'kind': 'linear', 'final': 0.0, 'warmup': 300}, x={'D': 5 * G // 2}))
    J.append(T('sch_sf_lr0.002', 'sched', 'sch_sf', opt={'name': 'sf_adamw'}, lr=2e-3, tokens=5 * G // 2, sched=const,
               milestones=[5 * G // 4], eval_at_milestones=True, x={'D': 5 * G // 2}))
    J.append(T('sch_main_s2', 'sched', 'sch_main_s2', seed=2, tokens=G, sched=const, milestones=[G], eval=False))
    J.append(br('sch_wsd_lin0_1.25B_s2', 'sch_rep', G, 5 * G // 4, seed=2, parent='sch_main_s2'))
    J.append(T('sch_cos10_1.25B_s2', 'sched', 'sch_rep', seed=2, tokens=5 * G // 4, sched={'kind': 'cosine', 'final': 0.1, 'warmup': 300},
               x={'D': 5 * G // 4}))
    return J


def adamw_best_is_base(R, J, state):
    b = best_in(R, J, ['opt_adamw_sweep'])
    return b is not None and abs(float(state['jobs'][b]['x']['lr']) - 1e-3) < 1e-12


def plan_opt():
    """Muon / SOAP（精简版）：12.5 亿 token 扫学习率（AdamW 补 2e-3，1e-3 取自调度块；Muon、SOAP 各 1e-3 / 2e-3 / 4e-3），
    各自最优值从 10 亿的里程碑续到 20 亿、冷却到 25 亿做最优对最优。AdamW 最优仍是 1e-3 时，它的 25 亿结果直接取自调度块。"""
    J = []
    sweeps = {'adamw': (2e-3,), 'muon': (1e-3, 2e-3, 4e-3), 'soap': (1e-3, 2e-3, 4e-3)}
    const = {'kind': 'constant', 'warmup': 300}

    def cool(p):
        return {'kind': 'wsd', 'decay_from': p, 'shape': 'linear', 'final': 0.0, 'warmup': 300}

    for o, lrs in sweeps.items():
        for lr in lrs:
            main = f'opt_{o}_lr{lr:g}_main'
            J.append(T(main, 'opt', f'opt_{o}_main', opt={'name': o}, lr=lr, tokens=G, sched=const, milestones=[G], eval=False))
            J.append(T(f'opt_{o}_lr{lr:g}_1.25B', 'opt', f'opt_{o}_sweep', opt={'name': o}, lr=lr, tokens=5 * G // 4,
                       branch_from={'run': main, 'tokens': G}, sched=cool(G), x={'main': main, 'D': 5 * G // 4}))
        lr = Ref('best_lr', f'opt_{o}_sweep')
        skip = 'adamw_base' if o == 'adamw' else None
        m2 = f'opt_{o}_best_main2'
        J += [T(m2, 'opt', f'opt_{o}_ext', opt={'name': o}, lr=lr, tokens=2 * G, sched=const, milestones=[2 * G], eval=False, skip_if=skip,
                branch_from={'run': Ref('best_main', f'opt_{o}_sweep'), 'tokens': G}),
              T(f'opt_{o}_best_2.5B', 'opt', f'opt_{o}_best', opt={'name': o}, lr=lr, tokens=5 * G // 2, branch_from={'run': m2, 'tokens': 2 * G},
                sched=cool(2 * G), arith=True, skip_if=skip, x={'D': 5 * G // 2})]
    return J


def plan_jobs():
    J = plan_mup() + plan_scaling() + plan_data() + plan_sched() + plan_opt()
    # 可选的加跑项（冷却长度、Schedule-Free 第二档、各优化器的 50 亿预算）统一排到最后：核心实验按排期先完成，加跑项随时可以用 STOP 截掉
    J = [j for j in J if not j.get('optional')] + [j for j in J if j.get('optional')]
    names = [j['name'] for j in J]
    if len(set(names)) != len(names):
        raise FatalError('实验名重复')
    for j in J:
        j['groups'] = [j['group']] + list(j.pop('also', []) or [])
    return J


# ------------------------------------------------------------------ 依赖与解析 ----------
def jstate(state, name):
    return state.setdefault('jobs', {}).setdefault(name, {'status': 'pending', 'attempts': 0})


def results(J, state):
    R = {}
    for j in J:
        if jstate(state, j['name'])['status'] == 'done':
            r = read_json(run_dir(j['name']) / 'result.json')
            if r:
                R[j['name']] = r
    return R


def group_finished(J, state, group):
    return all(jstate(state, j['name'])['status'] in FINISHED for j in J if group in j['groups'])


def best_in(R, J, groups):
    c = [(R[j['name']]['bpb_weighted'], j['name']) for j in J
         if set(j['groups']) & set(groups) and j['name'] in R and R[j['name']].get('bpb_weighted') is not None]
    return min(c)[1] if c else None


def resolve(v, J, state, R):
    if isinstance(v, Ref):
        if not all(group_finished(J, state, g) for g in v.groups):
            raise Pending(f'等待 {",".join(v.groups)} 完成')
        b = best_in(R, J, v.groups)
        if b is None:
            raise Unresolvable(f'{",".join(v.groups)} 没有成功的结果')
        x = jstate(state, b).get('x') or {}
        if v.kind == 'best_lr':
            return float(x['lr']) * v.mult
        if v.kind == 'best_main':
            return x['main']
        raise ValueError(v.kind)
    if isinstance(v, dict):
        return {k: resolve(x, J, state, R) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [resolve(x, J, state, R) for x in v]
    return v


def data_alias(state, name):
    return (state.get('data_alias') or {}).get(name, name)


def readiness(j, J, state, R):
    """返回 ('run', 解析后的配置) / 'wait' / ('skip', 原因)。"""
    try:
        spec = resolve({k: v for k, v in j.items() if k != 'skip_if'}, J, state, R)
    except Pending:
        return 'wait'
    except Unresolvable as exc:
        return ('skip', str(exc))
    if j.get('skip_if') == 'adamw_base' and adamw_best_is_base(R, J, state):
        return ('skip', 'AdamW 的最优学习率就是 1e-3：25 亿（两个种子）和 50 亿的结果直接取自调度块')
    data = spec.get('data')
    if spec['type'] == 'train' and data:
        data = spec['data'] = data_alias(state, data)
        if not lab_data.is_ready(data):
            if state.get('builder_finished') or data in (state.get('data_failed') or []):
                return ('skip', f'数据集 {data} 没有构建出来')
            return 'wait'
    bf = spec.get('branch_from')
    if bf:
        pj = jstate(state, bf['run'])
        ps = pj['status']
        if ps in ('failed', 'skipped'):
            return ('skip', f'起点 {bf["run"]} 状态为 {ps}')
        if not milestone_path(bf['run'], bf['tokens']).exists():
            if ps != 'done':
                return 'wait'
            if pj.get('reruns', 0) >= 1:
                return ('skip', f'起点 {bf["run"]} 的里程碑 {bf["tokens"]} 不存在（已重跑过一次）')
            # 里程碑在内存盘上，机器重启会丢：把主线改回待跑，重跑一遍
            pj.update(status='pending', reruns=pj.get('reruns', 0) + 1, attempts=0)
            (run_dir(bf['run']) / 'result.json').unlink(missing_ok=True)
            return 'wait'
    return ('run', spec)


def needed_milestones(J, state):
    """还没跑完的实验会用到的里程碑。起点是 Ref（扫描里最好的那条主线）时：能解析就只留解析出的那条；
    扫描还没结束就只留"当前最好的那条 + 还没跑完的候选"，其余随时可删（控制内存盘占用）。"""
    keep, R = set(), None
    for j in J:
        if jstate(state, j['name'])['status'] in FINISHED or not j.get('branch_from'):
            continue
        bf = j['branch_from']
        runs = [bf['run']]
        if isinstance(bf['run'], Ref):
            R = R if R is not None else results(J, state)
            try:
                runs = [resolve(bf['run'], J, state, R)]
            except Pending:
                members = [x for x in J if set(x['groups']) & set(bf['run'].groups)]
                runs = [(x.get('x') or {}).get('main') for x in members if jstate(state, x['name'])['status'] not in FINISHED]
                b = best_in(R, J, bf['run'].groups)
                if b:
                    runs.append((jstate(state, b).get('x') or {}).get('main'))
            except Unresolvable:
                runs = []
        for r in runs:
            if r:
                keep.add(str(milestone_path(r, bf['tokens'])))
    return keep


def cleanup_milestones(J, state):
    keep, freed = needed_milestones(J, state), 0
    for p in LAB_CKPT.glob('*/*.pt'):
        run = p.parent.name
        if jstate(state, run)['status'] in FINISHED and str(p) not in keep:
            freed += p.stat().st_size
            p.unlink()
    if freed:
        state.setdefault('freed_gb', 0)
        state['freed_gb'] = round(state['freed_gb'] + freed / 1e9, 1)


def cleanup_datasets(J, state):
    """除 proxy 外，一个数据集的实验全部结束后删除它自己的 token 分片（清单和回执保留）。cptmix 指向路线 A 第 2 段的分片，一并删除。"""
    gone = set(state.get('datasets_deleted') or [])
    use = {}
    for j in J:
        if j['type'] == 'train':
            use.setdefault(data_alias(state, j['data']), []).append(jstate(state, j['name'])['status'] in FINISHED)
    for d, fin in use.items():
        if d == 'proxy' or d in gone or not all(fin) or not lab_data.is_ready(d):
            continue
        man = read_json(LAB / 'data' / d / 'manifest.json') or {}
        own = (LAB / 'data' / d / 'tokens').resolve()
        freed = 0
        for e in (man.get('sources') or {}).values():
            for s in e.get('train', []):
                p = Path(s['path'])
                if p.exists() and (d == 'cptmix' or own in p.resolve().parents):
                    freed += p.stat().st_size
                    p.unlink()
        (LAB / 'data' / d / 'READY').rename(LAB / 'data' / d / 'DELETED')
        gone.add(d)
        state['datasets_deleted'] = sorted(gone)
        state.setdefault('freed_gb', 0)
        state['freed_gb'] = round(state['freed_gb'] + freed / 1e9, 1)


# ------------------------------------------------------------------ 子进程里执行一个实验 ----------
def child_env():
    env = dict(os.environ)
    env.pop('OMP_NUM_THREADS', None)
    env.update(LAB_RUN_DIR=str(LAB), A8_RUN_DIR=str(LAB), LAB_SCRATCH=str(LAB_SCRATCH), LAB_CKPT=str(LAB_CKPT), PYTHON=PY)
    return env


def run_job(name):
    """子进程入口：读 runs/<name>/spec.json，执行，结果写 result.json。退出码 0 成功，4 收到 STOP，其他为失败。"""
    from . import lab_core
    spec = read_json(run_dir(name) / 'spec.json')
    stop = lambda: control_signal() == 'stop'  # noqa: E731
    try:
        if spec['type'] == 'train':
            lab_core.train(spec, stop=stop)
        elif spec['type'] == 'coord':
            do_coord(spec)
        elif spec['type'] == 'ref':
            do_ref(spec)
        return 0
    except lab_core.StopJob:
        return 4


def do_coord(spec):
    import torch
    from . import lab_core
    widths = spec.get('widths', [128, 256, 512, 1024, 2048])
    data, slopes = lab_core.coord_check(widths, depth=spec.get('depth', 2), steps=spec.get('steps', 10), seeds=tuple(spec.get('seeds', (0, 1, 2))),
                                        lr=spec.get('lr', 2 ** -7))
    res = {'name': spec['name'], 'widths': widths, 'slopes': slopes, 'data': data}
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        kinds = list(slopes['sp'])
        fig, axes = plt.subplots(2, len(kinds), figsize=(3.2 * len(kinds), 6), squeeze=False)
        for r, param in enumerate(('sp', 'mup')):
            for c, kind in enumerate(kinds):
                ax = axes[r][c]
                for t in (0, 3, 6, 9):
                    ys = [data[param][w][kind][min(t, len(data[param][w][kind]) - 1)] for w in widths]
                    ax.plot(widths, ys, marker='o', label=f'step {t + 1}')
                ax.set_xscale('log', base=2)
                ax.set_yscale('log')
                ax.set_title(f'{param.upper()} {kind} (slope {slopes[param][kind]:+.2f})', fontsize=9)
                if r == 1:
                    ax.set_xlabel('width')
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        (LAB / 'figs').mkdir(parents=True, exist_ok=True)
        fig.savefig(LAB / 'figs' / 'mup_coord_check.png', dpi=110)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        res['plot_error'] = str(exc)
    write_json(run_dir(spec['name']) / 'result.json', res)
    torch.cuda.empty_cache()


def do_ref(spec):
    import torch
    from . import lab_core
    path = V1_MODEL if spec['model'] == 'v1' else CPT_MODEL
    if not path.exists():
        write_json(run_dir(spec['name']) / 'result.json', {'name': spec['name'], 'missing': str(path)})
        return
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck.get('model', ck) if isinstance(ck, dict) else ck
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    cfg = ck.get('config') if isinstance(ck, dict) else None
    shape = dict(V1_SHAPE)
    if isinstance(cfg, dict) and cfg.get('hidden_size'):
        shape = {'width': cfg['hidden_size'], 'depth': cfg['num_layers'], 'heads': cfg['num_heads'], 'kv_heads': cfg['num_kv_heads'],
                 'ffn': cfg['intermediate_size']}
    dev = torch.device('cuda')
    model = lab_core.LabLM(lab_core.shape_config(shape, max_seq_len=4096))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith('lm_head.weight')]
    if missing:
        raise RuntimeError(f'{path} 缺少参数：{missing[:5]}')
    model.to(dev)
    res = {'name': spec['name'], 'model': str(path), 'params': model.count_parameters(), **lab_core.evaluate(model, dev, arith=True)}
    write_json(run_dir(spec['name']) / 'result.json', res)


# ------------------------------------------------------------------ 调度器 ----------
def builder_alive(state):
    pid = state.get('builder_pid')
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            return b'lab_data' in f.read()
    except OSError:
        return False


def ensure_builder(state, bus):
    if state.get('builder_finished'):
        return
    if (LAB / 'data' / 'builder_done.json').exists():
        state['builder_finished'] = True
        return
    if builder_alive(state):
        return
    if state.get('builder_starts', 0) >= 4:
        state['builder_finished'] = True
        bus.post('data', 'decision', 'report', {'action': 'builder_gave_up', 'why': '数据构建进程连续失败 4 次'})
        return
    (LAB / 'logs').mkdir(parents=True, exist_ok=True)
    out = open(LAB / 'logs' / 'data_builder.out', 'a')
    p = subprocess.Popen([PY, '-u', '-m', 'decoder_from_scratch.a800.lab_data', 'build'], cwd=PROJECT, env=child_env(),
                         stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    state['builder_pid'] = p.pid
    state['builder_starts'] = state.get('builder_starts', 0) + 1
    bus.post('decision', 'data', 'directive', {'task': 'build_datasets', 'pid': p.pid, 'attempt': state['builder_starts']})


def phase_prepare(state, bus):
    for d in ('runs', 'data', 'logs', 'figs'):
        (LAB / d).mkdir(parents=True, exist_ok=True)
    LAB_CKPT.mkdir(parents=True, exist_ok=True)
    LAB_SCRATCH.mkdir(parents=True, exist_ok=True)
    try:
        lab_data.make_cptmix()
        bus.post('decision', 'all', 'decision', {'decision': 'cptmix', 'why': '路线 A 第 2 段分片取约 15 亿 token 给 μP 宽度扫描，其余分片删除'})
    except Exception as exc:  # noqa: BLE001
        state.setdefault('data_alias', {})['cptmix'] = 'proxy'
        bus.post('decision', 'all', 'decision', {'decision': 'cptmix_fallback', 'why': f'没有用上路线 A 的分片（{exc}），μP 宽度扫描改用 proxy'})
    ensure_builder(state, bus)
    J = plan_jobs()
    write_json(LAB / 'plan.json', [{k: (repr(v) if isinstance(v, Ref) else v) for k, v in j.items() if k != 'skip_if'} for j in J])


def launch(spec, state, bus):
    name = spec['name']
    write_json(run_dir(name) / 'spec.json', spec)
    js = jstate(state, name)
    js.update(status='running', attempts=js['attempts'] + 1, started=stamp(), x=spec.get('x', {}))
    write_json(LAB / 'state.json', state)
    bus.post('decision', 'trainer', 'directive', {'task': spec['type'], 'run': name, 'x': spec.get('x', {}), 'attempt': js['attempts']})
    (LAB / 'logs' / 'jobs').mkdir(parents=True, exist_ok=True)
    out = open(LAB / 'logs' / 'jobs' / f'{name}.log', 'a')
    p = subprocess.Popen([PY, '-u', '-m', 'decoder_from_scratch.a800.lab', 'job', name], cwd=PROJECT, env=child_env(), stdout=out,
                         stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while p.poll() is None:
        time.sleep(30)
        try:
            hb = float((LAB / 'heartbeat').read_text().split()[0])
        except Exception:  # noqa: BLE001
            hb = t0
        if time.time() - max(hb, t0) > 2400:
            os.killpg(p.pid, signal.SIGTERM)
            time.sleep(30)
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return 'hang'
        ensure_builder(state, bus)
    return p.returncode


def phase_experiments(state, bus):
    J = plan_jobs()
    while True:
        if control_signal() == 'stop':
            raise StopRequested('STOP file present')
        ensure_builder(state, bus)
        R = results(J, state)
        chosen, waiting = None, 0
        for j in J:
            js = jstate(state, j['name'])
            if js['status'] in FINISHED:
                continue
            r = readiness(j, J, state, R)
            if r == 'wait':
                waiting += 1
                continue
            if r[0] == 'skip':
                js.update(status='skipped', reason=r[1], finished=stamp())
                bus.post('decision', 'all', 'decision', {'decision': 'skip', 'run': j['name'], 'why': r[1]})
                continue
            chosen = r[1]
            break
        write_json(LAB / 'state.json', state)
        if chosen is None:
            if waiting == 0:
                return
            heartbeat(f'等待数据或依赖（{waiting} 组）')
            time.sleep(60)
            continue
        code = launch(chosen, state, bus)
        js = jstate(state, chosen['name'])
        if code == 4:
            js['status'] = 'pending'
            write_json(LAB / 'state.json', state)
            raise StopRequested('STOP file present')
        if code == 0 and (run_dir(chosen['name']) / 'result.json').exists():
            js.update(status='done', finished=stamp())
            r = read_json(run_dir(chosen['name']) / 'result.json')
            bus.post('evaluator', 'decision', 'report', {'action': 'run_done', 'run': chosen['name'],
                                                         'bpb_weighted': r.get('bpb_weighted'), 'diverged': r.get('diverged')})
        else:
            js['last_error'] = f'退出码 {code}，见 logs/jobs/{chosen["name"]}.log'
            js['status'] = 'failed' if js['attempts'] >= 2 else 'pending'
            if js['status'] == 'failed':
                js['finished'] = stamp()
            bus.post('decision', 'all', 'error', {'run': chosen['name'], 'code': code, 'attempts': js['attempts']})
            if js['status'] == 'pending':
                time.sleep(60)
        cleanup_milestones(J, state)
        try:
            cleanup_datasets(J, state)
        except Exception as exc:  # noqa: BLE001
            bus.post('data', 'decision', 'report', {'action': 'cleanup_failed', 'error': str(exc)[:300]})
        write_json(LAB / 'state.json', state)
        try:
            write_report(J, state)
        except Exception:  # noqa: BLE001
            write_text(LAB / 'REPORT_ERROR.txt', traceback.format_exc())


def phase_report(state, bus):
    J = plan_jobs()
    write_report(J, state, final=True)
    bus.post('analyst', 'decision', 'report', {'action': 'final_report', 'path': str(LAB / 'REPORT.md')})


PHASE_FN = {'prepare': phase_prepare, 'experiments': phase_experiments, 'report': phase_report}


def kill_orphan_jobs():
    """上一次调度进程异常退出时，它启动的实验子进程可能还在跑：先结束它们，避免同一组被两个进程同时训练。"""
    me = os.getpid()
    for d in Path('/proc').iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            cmd = (d / 'cmdline').read_bytes().replace(b'\0', b' ')
        except OSError:
            continue
        if b'decoder_from_scratch.a800.lab job' in cmd:
            try:
                os.kill(int(d.name), signal.SIGTERM)
            except OSError:
                pass


def run():
    LAB.mkdir(parents=True, exist_ok=True)
    kill_orphan_jobs()
    bus = Bus(LAB)
    state = read_json(LAB / 'state.json', {}) or {'phase': 'prepare', 'failures': 0, 'started': stamp()}
    for js in (state.get('jobs') or {}).values():
        if js.get('status') == 'running':
            js['status'] = 'pending'
    try:
        while state['phase'] != 'done':
            phase = state['phase']
            heartbeat(phase)
            bus.post('decision', 'all', 'phase', {'phase': phase})
            PHASE_FN[phase](state, bus)
            state['phase'] = PHASES[PHASES.index(phase) + 1]
            state['failures'] = 0
            write_json(LAB / 'state.json', state)
        bus.post('decision', 'all', 'decision', {'decision': 'complete', 'why': '实验室队列完成，报告见 REPORT.md'})
        return 0
    except StopRequested:
        write_json(LAB / 'state.json', state)
        return 4
    except FatalError as exc:
        err = {'time': stamp(), 'phase': state.get('phase'), 'error': str(exc), 'traceback': traceback.format_exc()[-4000:]}
        write_json(LAB / 'FAILED.json', err)
        return 2
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, flush=True)
        state['failures'] = state.get('failures', 0) + 1
        state['last_error'] = {'time': stamp(), 'phase': state.get('phase'), 'error': f'{type(exc).__name__}: {exc}', 'traceback': tb[-4000:]}
        write_json(LAB / 'state.json', state)
        if state['failures'] >= 6:
            write_json(LAB / 'FAILED.json', state['last_error'])
            return 2
        return 1


# ------------------------------------------------------------------ 报告 ----------
def sigma_bpb():
    s = read_json(ABL_SUMMARY) or {}
    return s.get('sigma_bpb')


def fmt(v, nd=4):
    return '-' if v is None else (f'{v:.{nd}f}' if isinstance(v, float) else str(v))


def fit_power(xs, ys):
    """L = E + A·x^(-α)，α 网格搜索 + 线性最小二乘。返回 (E, A, α, 残差均方根)。"""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    best = None
    for a in np.arange(0.05, 1.5, 0.005):
        X = np.stack([np.ones_like(xs), xs ** -a], 1)
        coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
        if coef[1] < 0:
            continue
        r = float(np.sqrt(np.mean((X @ coef - ys) ** 2)))
        if best is None or r < best[3]:
            best = (float(coef[0]), float(coef[1]), float(a), r)
    return best


def fit_chinchilla(N, D, L):
    """L = E + A·N^(-α) + B·D^(-β)：α、β 网格搜索，E、A、B 线性最小二乘（要求 A、B ≥ 0）。"""
    N, D, L = (np.asarray(v, float) for v in (N, D, L))
    best = None
    for a in np.arange(0.05, 1.2, 0.01):
        for b in np.arange(0.05, 1.2, 0.01):
            X = np.stack([np.ones_like(N), N ** -a, D ** -b], 1)
            coef, *_ = np.linalg.lstsq(X, L, rcond=None)
            if coef[1] < 0 or coef[2] < 0:
                continue
            r = float(np.sqrt(np.mean((X @ coef - L) ** 2)))
            if best is None or r < best['rmse']:
                best = {'E': float(coef[0]), 'A': float(coef[1]), 'B': float(coef[2]), 'alpha': float(a), 'beta': float(b), 'rmse': r}
    return best


def write_report(J, state, final=False):
    R = results(J, state)
    X = {n: (jstate(state, n).get('x') or {}) for n in R}
    sig = sigma_bpb()
    L = ['# 实验室报告：μP / scaling law / 数据工程 / 学习率调度 / Muon·SOAP', '', f'- 更新时间：{stamp()}' + ('（最终）' if final else ''),
         f'- 噪声参考：第一批消融 3 个种子的 σ = {fmt(sig)} bits/字节；两组独立训练之差超过 3·√2·σ = {fmt(3 * math.sqrt(2) * sig) if sig else "-"} 才算显著。'
         '本报告里同一块的对比组大多用同一个种子（初始化和数据顺序相同），实际噪声比这个小，复核组给出配对噪声。', '']
    blocks = [('mup', 'μP'), ('scaling', 'scaling law'), ('data', '数据工程'), ('sched', '学习率调度'), ('opt', 'Muon / SOAP'), ('optional', '加跑项（排在最后）')]
    L += ['## 进度', '', '| 块 | 完成 | 失败 | 跳过 | 待跑 |', '|---|---|---|---|---|']
    for b, title in blocks:
        st = [jstate(state, j['name'])['status'] for j in J if (j.get('optional') and b == 'optional') or (not j.get('optional') and j['block'] == b)]
        L.append(f'| {title} | {st.count("done")} | {st.count("failed")} | {st.count("skipped")} | {len(st) - st.count("done") - st.count("failed") - st.count("skipped")} |')
    L.append('')
    for fn in (report_mup, report_scaling, report_data, report_sched, report_opt):
        try:
            L += fn(J, state, R, X, sig)
        except Exception as exc:  # noqa: BLE001
            L += [f'（{fn.__name__} 生成出错：{type(exc).__name__}: {exc}）', '']
    bad = [(j['name'], jstate(state, j['name'])) for j in J if jstate(state, j['name'])['status'] in ('failed', 'skipped')]
    if bad:
        L += ['## 失败与跳过', ''] + [f'- {n}：{s["status"]}，{s.get("reason") or s.get("last_error")}' for n, s in bad] + ['']
    write_text(LAB / 'REPORT.md', '\n'.join(L) + '\n')


def bpb(R, n):
    return (R.get(n) or {}).get('bpb_weighted')


def report_mup(J, state, R, X, sig):
    L = ['## 1. μP', '']
    c = R.get('mup_coord')
    if c:
        L += ['### 坐标检查（2 层，10 步，3 个种子；斜率 = 最后一步 log(激活均值) 对 log(宽度) 的斜率，μP 应接近 0）', '',
              '| 层 | SP 斜率 | μP 斜率 |', '|---|---|---|']
        for k in c['slopes']['sp']:
            L.append(f'| {k} | {c["slopes"]["sp"][k]:+.2f} | {c["slopes"]["mup"][k]:+.2f} |')
        ok = all(abs(v) < 0.15 for v in c['slopes']['mup'].values())
        L += ['', f'- 判定：μP 实现{"通过" if ok else "没有通过"}（各类层斜率 {"都" if ok else "不全"}在 ±0.15 以内）；图：figs/mup_coord_check.png', '']
    rows = {}
    for n, x in X.items():
        if n.startswith('mupw_'):
            rows.setdefault((x['param'], x['width']), {})[x['lr']] = bpb(R, n)
    if rows:
        L += ['### 宽度扫描（8 层，3 亿 token）：加权 bits/字节', '', '| 参数化 | 宽度 | ' + ' | '.join(f'2^{int(round(math.log2(lr)))}' for lr in LR_GRID) + ' | 最优 |',
              '|---|---|' + '---|' * (len(LR_GRID) + 1)]
        best = {}
        for (param, w) in sorted(rows):
            vals = rows[(param, w)]
            ok = {lr: v for lr, v in vals.items() if v is not None}
            b = min(ok, key=ok.get) if ok else None
            best[(param, w)] = b
            L.append(f'| {param.upper()} | {w} | ' + ' | '.join(fmt(vals.get(lr)) for lr in LR_GRID) + f' | {"-" if b is None else f"2^{int(round(math.log2(b)))}"} |')
        for param in ('sp', 'mup'):
            bs = [best.get((param, w)) for w in (128, 256, 512, 1024) if best.get((param, w))]
            if len(bs) >= 2:
                span = math.log2(max(bs)) - math.log2(min(bs))
                L.append(f'- {param.upper()}：各宽度最优学习率在 2^{int(round(math.log2(min(bs))))} 到 2^{int(round(math.log2(max(bs))))} 之间（相差 {span:.0f} 档）')
        L.append('')
        plot_mup_sweep(rows)
    b16 = [(bpb(R, n), X[n]['lr']) for n in X if n.startswith('mup16_') and bpb(R, n) is not None]
    if b16:
        v, lr = min(b16)
        L += [f'- 深度 16、宽 256 的基准扫描：最优学习率 {lr:.2e}（2^{int(round(math.log2(lr)))}），加权 bits/字节 {v:.4f}；scaling law 各规模都用它', '']
    tx = [(X[n]['mult'], bpb(R, n), X[n]['lr']) for n in X if n.startswith('mupx_')]
    shape, p0, end, warm = transfer_point()
    one = sl_name(768, end)
    if one in R:
        tx.append((1.0, bpb(R, one), X[one]['lr']))
    if tx:
        L += [f'### 迁移验证（1.25 亿参数，宽 768，m = 3，{end / G:.3g}B token；×1 取自 scaling law 同预算的分支）', '',
              '| 学习率倍数 | 学习率 | 加权 bits/字节 |', '|---|---|---|']
        for m, v, lr in sorted(tx):
            L.append(f'| ×{m:g} | {lr:.2e} | {fmt(v)} |')
        okv = [(v, m) for m, v, lr in tx if v is not None]
        one_v = next((v for m, v, lr in tx if m == 1.0 and v is not None), None)
        if okv and one_v is not None:
            bv, bm = min(okv)
            if bm == 1.0:
                L.append('- 最优是 ×1：从宽 256 迁移来的学习率就是宽 768 的最优')
            else:
                L.append(f'- 最优是 ×{bm:g}（{bv:.4f}），×1 是 {one_v:.4f}，差 {one_v - bv:.4f}' +
                         (f'（{"超过" if one_v - bv > 3 * math.sqrt(2) * sig else "没超过"}显著线 {3 * math.sqrt(2) * sig:.4f}）' if sig else ''))
        L.append('')
    return L


def plot_mup_sweep(rows):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, param in zip(axes, ('sp', 'mup')):
            for w in (128, 256, 512, 1024):
                vals = rows.get((param, w)) or {}
                pts = sorted((lr, v) for lr, v in vals.items() if v is not None)
                if pts:
                    ax.plot([math.log2(p[0]) for p in pts], [p[1] for p in pts], marker='o', label=f'width {w}')
            ax.set_title(param.upper())
            ax.set_xlabel('log2 learning rate')
        axes[0].set_ylabel('weighted bits/byte')
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(LAB / 'figs' / 'mup_width_sweep.png', dpi=110)
        plt.close(fig)
    except Exception:  # noqa: BLE001
        pass


def report_scaling(J, state, R, X, sig):
    pts = [(X[n]['N'], X[n]['D'], bpb(R, n)) for n in X if n.startswith('sl_w') and 'D' in X[n] and bpb(R, n) is not None]
    L = ['## 2. scaling law', '']
    if not pts:
        return L + ['- 还没有结果', '']
    L += ['| 参数量 | token 数 | token/参数 | 加权 bits/字节 |', '|---|---|---|---|']
    for n, d, v in sorted(pts):
        L.append(f'| {n / 1e6:.0f}M | {d / 1e9:.2f}B | {d / n:.1f} | {v:.4f} |')
    L.append('')
    if len(pts) >= 8:
        f = fit_chinchilla(*zip(*pts))
        if f:
            pred = f['E'] + f['A'] * V1_N ** -f['alpha'] + f['B'] * V1_D ** -f['beta']
            L += [f'- 拟合：L = {f["E"]:.4f} + {f["A"]:.3g}·N^(-{f["alpha"]:.2f}) + {f["B"]:.3g}·D^(-{f["beta"]:.2f})，残差均方根 {f["rmse"]:.4f}',
                  f'- 外推到 v1（{V1_N / 1e6:.0f}M 参数、250 亿 token）：预测 {pred:.4f}' +
                  (f'；实测 {bpb(R, "ref_v1"):.4f}（差 {bpb(R, "ref_v1") - pred:+.4f}）' if bpb(R, 'ref_v1') else '') +
                  '。注意 v1 是 40 层、SP、学习率 4e-4，这一族是 16 层、μP，差异里包含结构和超参的影响',
                  f'- 按拟合式，计算量固定时参数量与 token 数的最优比例随 α/β = {f["alpha"] / f["beta"]:.2f} 变化（α>β 时数据更值钱，反之参数更值钱）']
            if bpb(R, 'ref_cpt'):
                L.append(f'- 参照：路线 A 最终模型在同一验证集上 {bpb(R, "ref_cpt"):.4f}')
            state['scaling_fit'] = f
    L.append('')
    return L


def report_data(J, state, R, X, sig):
    L = ['## 3. 数据工程对比（d768_l16，25 亿 token，学习率 1e-3，seed 1234）', '']
    rows = [(n, t) for n, t in (('sch_wsd_lin10_2.5B', 'v1 配比（基准，取自调度块同配置分支）'), ('de_mix_anneal', '路线 A 退火配比'),
                                ('de_q45', '中文教育只用 4-5 分（占 30%）'), ('de_q34', '中文教育只用 3-4 分（占 30%）')) if n in R]
    if not rows:
        return L + ['- 还没有结果', '']
    srcs = list((R[rows[0][0]].get('bpb') or {}).keys())
    L += ['| 组 | 设置 | 加权 bits/字节 | ' + ' | '.join(srcs) + ' | 运算 |', '|---|---|---|' + '---|' * (len(srcs) + 1)]
    for n, t in rows:
        r = R[n]
        L.append(f'| {n} | {t} | {fmt(r.get("bpb_weighted"))} | ' + ' | '.join(fmt((r.get('bpb') or {}).get(s)) for s in srcs) +
                 f' | {fmt(r.get("arith_mean"), 3)} |')
    thr = 3 * math.sqrt(2) * sig if sig else None
    base = R.get('sch_wsd_lin10_2.5B')
    if base and 'de_mix_anneal' in R:
        a = R['de_mix_anneal']
        d = a['bpb_weighted'] - base['bpb_weighted']
        L.append(f'- 退火配比对 v1 配比：加权 {d:+.4f}' + (f'（显著线 {thr:.4f}）' if thr else '') + '；分来源：' +
                 '，'.join(f'{s} {a["bpb"][s] - base["bpb"][s]:+.4f}' for s in srcs if s in a.get('bpb', {}) and s in base.get('bpb', {})))
    if 'de_q45' in R and 'de_q34' in R:
        a, b = R['de_q45'], R['de_q34']
        L.append(f'- 质量分档：4-5 分对 3-4 分，加权 {a["bpb_weighted"] - b["bpb_weighted"]:+.4f}，中文教育验证集 {a["bpb"].get("zh_fwedu", 0) - b["bpb"].get("zh_fwedu", 0):+.4f}，'
                 f'CCI3 验证集 {a["bpb"].get("zh_cci3", 0) - b["bpb"].get("zh_cci3", 0):+.4f}（负数 = 高分档更好）')
    dd = read_json(LAB / 'data' / 'dedup_plan.json') or {}
    if dd:
        L.append(f'- 去重对照：{dd.get("why")}' + ('' if dd.get('sources') else '；没有达到阈值的来源，不做'))
    for n in [n for n in R if n.startswith('de_dd_') and n.endswith('_raw')]:
        m = n[:-4] + '_dedup'
        if m in R:
            s = X[n]['source']
            L.append(f'- {s}：去重后对原样（各 10 亿 token），加权 {R[m]["bpb_weighted"] - R[n]["bpb_weighted"]:+.4f}，'
                     f'{s} 验证集 {R[m]["bpb"].get(s, 0) - R[n]["bpb"].get(s, 0):+.4f}')
    L.append('')
    return L


def report_sched(J, state, R, X, sig):
    L = ['## 4. 学习率调度（d768_l16，峰值 1e-3，seed 1234）', '']
    if not any(n.startswith('sch_') for n in R):
        return L + ['- 还没有结果', '']
    L += ['| 预算 | WSD 冷却 20% 线性到 0 | 余弦到 10% | 其他 |', '|---|---|---|---|']
    for d in (0.625, 1.25, 2.5, 5):
        other = []
        if d == 2.5:
            other = [f'全程线性到 0：{fmt(bpb(R, "sch_d2z_2.5B"))}', f'Schedule-Free（不衰减）：{fmt(bpb(R, "sch_sf_lr0.002"))}']
        if d == 1.25:
            me = ((R.get('sch_sf_lr0.002') or {}).get('train') or {}).get('milestone_evals', {}).get(str(5 * G // 4))
            if me:
                other.append(f'Schedule-Free 中途（不衰减）：{me["bpb_weighted"]:.4f}')
        L.append(f'| {d:g}B | {fmt(bpb(R, f"sch_wsd_lin0_{d:g}B"))} | {fmt(bpb(R, f"sch_cos10_{d:g}B"))} | {"；".join(other)} |')
    L += ['', '- 25 亿预算下的冷却形状：线性到 0 ' + fmt(bpb(R, 'sch_wsd_lin0_2.5B')) + '，线性到 10% ' + fmt(bpb(R, 'sch_wsd_lin10_2.5B')) +
          '，1−√ 到 0 ' + fmt(bpb(R, 'sch_wsd_sqrt0_2.5B'))]
    me = ((R.get('sch_main') or {}).get('train') or {}).get('milestone_evals') or {}
    gains = []
    for p, end in ((G // 2, '0.625'), (G, '1.25'), (2 * G, '2.5'), (4 * G, '5')):
        before, after = (me.get(str(p)) or {}).get('bpb_weighted'), bpb(R, f'sch_wsd_lin0_{end}B')
        if before and after:
            gains.append(f'{end}B：冷却前 {before:.4f} → 冷却后 {after:.4f}（{after - before:+.4f}）')
    if gains:
        L.append('- 冷却收益（主线在分支点的值 → 分支终点）：' + '；'.join(gains))
    a1, c1, a2, c2 = bpb(R, 'sch_wsd_lin0_1.25B'), bpb(R, 'sch_cos10_1.25B'), bpb(R, 'sch_wsd_lin0_1.25B_s2'), bpb(R, 'sch_cos10_1.25B_s2')
    if None not in (a1, c1, a2, c2):
        d1, d2 = a1 - c1, a2 - c2
        L.append(f'- 复核（WSD − 余弦，12.5 亿）：seed 1234 {d1:+.4f}，seed 2 {d2:+.4f}；两次差值之差 {abs(d1 - d2):.4f} 是配对噪声的量级，'
                 + ('方向一致' if d1 * d2 > 0 else '方向不一致，说明差异在噪声内'))
    L.append('')
    plot_sched(R)
    return L


def plot_sched(R):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, tpl in (('WSD (20%, linear to 0)', 'sch_wsd_lin0_{}B'), ('cosine to 10%', 'sch_cos10_{}B')):
            pts = [(d, bpb(R, tpl.format(f'{d:g}'))) for d in (0.625, 1.25, 2.5, 5)]
            pts = [p for p in pts if p[1] is not None]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', label=label)
        ax.set_xscale('log')
        ax.set_xlabel('tokens (B)')
        ax.set_ylabel('weighted bits/byte')
        ax.legend()
        fig.tight_layout()
        fig.savefig(LAB / 'figs' / 'schedules.png', dpi=110)
        plt.close(fig)
    except Exception:  # noqa: BLE001
        pass


def report_opt(J, state, R, X, sig):
    L = ['## 5. Muon / SOAP（d768_l16，WSD 冷却 20% 线性到 0）', '']
    sweep = {}
    for n in R:
        for o in ('adamw', 'muon', 'soap'):
            if n.startswith(f'opt_{o}_lr') and n.endswith('_1.25B'):
                sweep.setdefault(o, {})[X[n]['lr']] = bpb(R, n)
    if 'sch_wsd_lin0_1.25B' in R:
        sweep.setdefault('adamw', {})[1e-3] = bpb(R, 'sch_wsd_lin0_1.25B')
    if not sweep:
        return L + ['- 还没有结果', '']
    lrs = sorted({lr for v in sweep.values() for lr in v})
    L += ['### 学习率扫描（12.5 亿 token）', '', '| 优化器 | ' + ' | '.join(f'{lr:g}' for lr in lrs) + ' |', '|---|' + '---|' * len(lrs)]
    best = {}
    for o, v in sweep.items():
        L.append(f'| {o} | ' + ' | '.join(fmt(v.get(lr)) for lr in lrs) + ' |')
        ok = {lr: x for lr, x in v.items() if x is not None}
        if ok:
            best[o] = min(ok, key=ok.get)
    L.append('')
    adamw_base = best.get('adamw') == 1e-3

    def runs_for(o):
        if o == 'adamw' and adamw_base:
            return {0.625: 'sch_wsd_lin0_0.625B', 1.25: 'sch_wsd_lin0_1.25B', 2.5: 'sch_wsd_lin0_2.5B', 5: 'sch_wsd_lin0_5B'}, 'sch_wsd_lin0_2.5B_s2'
        b = best.get(o)
        d = {1.25: f'opt_{o}_lr{b:g}_1.25B' if b else None, 2.5: f'opt_{o}_best_2.5B', 5: f'opt_{o}_best_5B'}
        return d, f'opt_{o}_best_s2_2.5B'

    L += ['### 各自最优学习率', '', '| 优化器 | 学习率 | 12.5 亿 | 25 亿 | 吞吐 tok/s | 最大注意力 logit | 裁剪占比 |', '|---|---|---|---|---|---|---|']
    for o in ('adamw', 'muon', 'soap'):
        if o not in best:
            continue
        d, rep = runs_for(o)
        r25 = R.get(d.get(2.5)) or {}
        L.append(f'| {o} | {best[o]:g} | {fmt(bpb(R, d.get(1.25)))} | {fmt(bpb(R, d.get(2.5)))} | '
                 f'{(r25.get("train") or {}).get("tok_s", "-")} | {max(r25.get("max_attn_logit") or [0]) or "-"} | {(r25.get("train") or {}).get("clip_fraction", "-")} |')
    d, _ = runs_for('adamw')
    ruler = [(D * G, bpb(R, n)) for D, n in d.items() if n and bpb(R, n) is not None]
    if 'sch_wsd_lin0_0.625B' in R and adamw_base is False and best.get('adamw') is not None:
        L.append('- 注意：AdamW 最优学习率不是 1e-3，尺子只用该学习率自己的点')
    if len(ruler) >= 3:
        fit = fit_power(*zip(*ruler))
        if fit:
            E, A, a, rmse = fit
            L += ['', f'- AdamW 尺子：L(D) = {E:.4f} + {A:.3g}·D^(-{a:.3f})（{len(ruler)} 个预算点，残差 {rmse:.4f}）']
            for o in ('muon', 'soap'):
                dd, _ = runs_for(o)
                for D in (1.25, 2.5, 5):
                    v = bpb(R, dd.get(D))
                    if v is None:
                        continue
                    if v <= E:
                        L.append(f'- {o} {D:g}B：loss {v:.4f} 低于尺子的渐近值，无法换算')
                        continue
                    Deq = (A / (v - E)) ** (1 / a)
                    tok_o = ((R.get(dd.get(D)) or {}).get('train') or {}).get('tok_s')
                    tok_a = ((R.get(runs_for('adamw')[0].get(D)) or {}).get('train') or {}).get('tok_s')
                    wall = f'，按吞吐折算墙钟加速约 {Deq / (D * G) * tok_o / tok_a:.2f} 倍' if tok_o and tok_a else ''
                    L.append(f'- {o} {D:g}B：相当于 AdamW 训练 {Deq / G:.2f}B token，token 加速 {Deq / (D * G):.2f} 倍{wall}')
    L.append('')
    return L


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if cmd == 'run':
        sys.exit(run())
    if cmd == 'job':
        sys.exit(run_job(sys.argv[2]))
    if cmd == 'report':
        J = plan_jobs()
        st = read_json(LAB / 'state.json', {}) or {}
        write_report(J, st)
        print(open(LAB / 'REPORT.md', encoding='utf-8').read()[:3000])
    else:
        st = read_json(LAB / 'state.json', {}) or {}
        print(json.dumps({'phase': st.get('phase'), 'jobs': {k: v.get('status') for k, v in (st.get('jobs') or {}).items()}},
                         ensure_ascii=False)[:3000])

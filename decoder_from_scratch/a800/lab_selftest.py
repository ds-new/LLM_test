"""实验室代码自测。默认在 CPU 上跑（不占 GPU、不碰正式运行目录）：python -m decoder_from_scratch.a800.lab_selftest cpu
gpu：在 GPU 上用极小模型把四种优化器、μP、主线 + 分支、torch.compile 各走一遍（几分钟，显存占用很小）。
结果打印每一项 PASS / FAIL，并写到 <临时目录>/selftest.json。"""

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
TMP = Path(tempfile.mkdtemp(prefix=f'lab_selftest_{MODE}_', dir=os.environ.get('LAB_SELFTEST_DIR', '/root/autodl-tmp')))
os.environ['LAB_RUN_DIR'] = str(TMP)
os.environ['A8_RUN_DIR'] = str(TMP)
os.environ['LAB_SCRATCH'] = str(TMP / 'scratch')
os.environ['LAB_CKPT'] = str(TMP / 'ckpt')
if MODE == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np  # noqa: E402
import torch  # noqa: E402

from . import lab_core as C  # noqa: E402

RESULTS = []
DEV = torch.device('cuda' if MODE == 'gpu' else 'cpu')


def check(name, fn):
    t0 = __import__('time').time()
    try:
        info = fn()
        RESULTS.append({'test': name, 'ok': True, 'info': info, 's': round(__import__('time').time() - t0, 1)})
        print(f'PASS {name} {info if info is not None else ""}', flush=True)
    except Exception as exc:  # noqa: BLE001
        RESULTS.append({'test': name, 'ok': False, 'error': f'{type(exc).__name__}: {exc}', 'tb': traceback.format_exc()[-2000:]})
        print(f'FAIL {name}: {type(exc).__name__}: {exc}', flush=True)
        traceback.print_exc()


def tiny_run(**kw):
    run = {'shape': {'width': 64, 'depth': 2}, 'param': 'sp', 'opt': {'name': 'adamw', 'lr': 3e-3}, 'seed': 7}
    run.update(kw)
    return run


def batch(seed=0, n=4, t=64):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, 32000, (n, t + 1), generator=g)


# ------------------------------------------------------------------ 调度 ----------
def t_schedules():
    s = {'warmup': 10, 'total': 100, 'kind': 'wsd', 'decay_start': 80, 'shape': 'linear', 'final': 0.1}
    assert abs(C.lr_factor(0, s) - 0.1) < 1e-9 and C.lr_factor(9, s) == 1.0 and C.lr_factor(79, s) == 1.0
    assert abs(C.lr_factor(90, s) - 0.55) < 1e-9 and abs(C.lr_factor(100, s) - 0.1) < 1e-9
    s2 = dict(s, shape='1-sqrt', final=0.0)
    assert abs(C.lr_factor(85, s2) - (1 - 0.5)) < 1e-9
    c = {'warmup': 10, 'total': 110, 'kind': 'cosine', 'final': 0.1}
    assert abs(C.lr_factor(60, c) - 0.55) < 1e-9 and abs(C.lr_factor(110, c) - 0.1) < 1e-9
    lin = {'warmup': 10, 'total': 110, 'kind': 'linear', 'final': 0.0}
    assert abs(C.lr_factor(60, lin) - 0.5) < 1e-9
    run = {'sched': {'kind': 'wsd', 'decay_from': 2048, 'warmup': 3}, 'tokens': 4096}
    st = C.sched_steps(run, 8, 256)
    assert st['decay_start'] == 8 and st['total'] == 8
    st = C.sched_steps({'tokens': 1000 * 256}, 1000, 256)
    assert st['decay_start'] == 800 and st['warmup'] == 100 and st['final'] == 0.1
    return 'wsd / 1-sqrt / cosine / linear / 步数换算'


# ------------------------------------------------------------------ 优化器 ----------
def make(run):
    torch.manual_seed(run['seed'])
    model, m = C.build_model(run)
    model.to(DEV)
    return model, C.build_optimizer(model, run['opt'], m, DEV), m


def train_steps(model, opt, n, seed0=0, lr=None):
    losses = []
    for i in range(n):
        b = batch(seed0 + i).to(DEV)
        if lr is not None:
            opt.set_lr(lr)
        with C.amp(DEV):
            loss = model(b[:, :-1], targets=b[:, 1:]).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        losses.append(float(loss))
    return losses


def t_optimizer(name):
    def fn():
        run = tiny_run(opt={'name': name, 'lr': 3e-3})
        model, opt, _ = make(run)
        fixed = batch(123).to(DEV)
        l0 = None
        for i in range(30):
            opt.set_lr(3e-3)
            with C.amp(DEV):
                loss = model(fixed[:, :-1], targets=fixed[:, 1:]).loss
            l0 = float(loss) if l0 is None else l0
            loss.backward()
            opt.step()
            opt.zero_grad()
        opt.eval()
        with torch.no_grad(), C.amp(DEV):
            l1 = float(model(fixed[:, :-1], targets=fixed[:, 1:]).loss)
        opt.train()
        assert l1 < l0 - 0.5, f'loss 没有下降：{l0:.3f} → {l1:.3f}'
        # 存取一致：3 步 → 存 → 再 3 步 与 读回后 3 步 结果相同
        model, opt, _ = make(run)
        train_steps(model, opt, 3, 0, 3e-3)
        ck = {'model': {k: v.clone() for k, v in model.state_dict().items()}, 'opt': opt.state_dict()}
        path = TMP / f'opt_{name}.pt'
        torch.save(ck, path)
        train_steps(model, opt, 3, 100, 3e-3)
        a = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model2, opt2, _ = make(run)
        ck = torch.load(path, weights_only=False)
        model2.load_state_dict(ck['model'])
        opt2.load_state_dict(ck['opt'])
        train_steps(model2, opt2, 3, 100, 3e-3)
        diff = max(float((a[k].float() - v.float()).abs().max()) for k, v in model2.state_dict().items())
        assert diff < 1e-5, f'读回后继续训练与不中断不一致，最大差 {diff:.2e}'
        return f'loss {l0:.3f} → {l1:.3f}，读回差 {diff:.1e}'
    return fn


def t_muon_rms():
    w = torch.nn.Parameter(torch.zeros(64, 256))
    opt = torch.optim.Muon([w], lr=1.0, weight_decay=0.0, momentum=0.0, nesterov=False, adjust_lr_fn='match_rms_adamw')
    w.grad = torch.randn(64, 256)
    opt.step()
    rms = float(w.detach().pow(2).mean().sqrt())
    sv = torch.linalg.svdvals(w.detach() / (0.2 * 256 ** 0.5))
    assert 0.1 < rms < 0.3, f'更新 RMS {rms:.3f} 不在 0.2 附近'
    assert float(sv.min()) > 0.4 and float(sv.max()) < 1.6, f'正交化后奇异值 {float(sv.min()):.2f}–{float(sv.max()):.2f}'
    return f'更新 RMS {rms:.3f}，奇异值 {float(sv.min()):.2f}–{float(sv.max()):.2f}'


def t_soap_basis():
    run = tiny_run(opt={'name': 'soap', 'lr': 3e-3, 'freq': 3})
    model, opt, _ = make(run)
    train_steps(model, opt, 8, 0, 3e-3)
    so = opt.opts[0]
    worst = 0.0
    for p in so.param_groups[0]['params']:
        for q in so.state[p]['Q']:
            if q is not None:
                worst = max(worst, float((q.T @ q - torch.eye(q.shape[0], device=q.device)).abs().max()))
    assert worst < 1e-3, f'特征基不正交：{worst:.2e}'
    return f'特征基正交误差 {worst:.1e}'


def t_schedule_free_modes():
    run = tiny_run(opt={'name': 'sf_adamw', 'lr': 3e-3})
    model, opt, _ = make(run)
    train_steps(model, opt, 5, 0, 3e-3)
    y = {k: v.clone() for k, v in model.state_dict().items()}
    opt.eval()
    x = {k: v.clone() for k, v in model.state_dict().items()}
    opt.train()
    back = max(float((y[k] - v).abs().max()) for k, v in model.state_dict().items())
    moved = max(float((y[k] - x[k]).abs().max()) for k in y)
    assert back < 1e-6 and moved > 0, f'train/eval 切换不可逆（{back:.1e}）或没有移动（{moved:.1e}）'
    return f'eval/train 往返误差 {back:.1e}'


# ------------------------------------------------------------------ μP ----------
def t_mup():
    torch.manual_seed(1)
    sp, _ = C.build_model({'shape': {'width': 256, 'depth': 2}, 'param': 'sp'})
    torch.manual_seed(1)
    mu, m = C.build_model({'shape': {'width': 256, 'depth': 2}, 'param': 'mup', 'mup': {'base_width': 256}})
    same = all(torch.equal(x, y) for x, y in zip(sp.state_dict().values(), mu.state_dict().values()))
    assert m == 1.0 and same and mu.output_mult == 1.0, 'm = 1 时 μP 应与 SP 完全相同'
    torch.manual_seed(1)
    big, m4 = C.build_model({'shape': {'width': 1024, 'depth': 2}, 'param': 'mup', 'mup': {'base_width': 256}})
    q = big.layers[0].attention.q_proj.weight.std().item()
    o = big.layers[0].attention.o_proj.weight.std().item()
    e = big.embedding.weight.std().item()
    assert m4 == 4.0 and abs(q - 0.01) < 0.001 and abs(o - 0.01 / 2) < 0.0006 and abs(e - 0.02) < 0.001, (q, o, e)
    assert abs(big.output_mult - 0.25) < 1e-12
    opt = C.build_optimizer(big, {'name': 'adamw', 'lr': 1e-3}, m4, torch.device('cpu'))
    mults = {g['kind']: g['lr_mult'] for g in opt.param_groups}
    assert mults == {'hidden': 0.25, 'embed': 1.0, 'vector': 1.0}, mults
    return f'm=4：q std {q:.4f}，o std {o:.4f}，嵌入 std {e:.4f}，lr_mult {mults}'


def t_coord():
    out, slopes = C.coord_check([64, 128, 256, 512], depth=2, steps=4, seeds=(0, 1), lr=2 ** -7, micro=4, seq_len=128, device=DEV)
    worst_mup = max(abs(v) for v in slopes['mup'].values())
    worst_sp = max(slopes['sp'].values())
    assert worst_mup < worst_sp, f'μP 斜率（{slopes["mup"]}）没有比 SP（{slopes["sp"]}）更平'
    return {'sp': slopes['sp'], 'mup': slopes['mup']}


# ------------------------------------------------------------------ 端到端：数据集 → 主线 → 分支 → 断点续训 ----------
def make_dataset():
    _, streams = C.eval_streams(1024)
    arr = np.concatenate([w.numpy() for e in streams.values() for w in e['wins'][:40]]).astype('<u2')
    d = TMP / 'data' / 'tiny'
    (d / 'tokens').mkdir(parents=True, exist_ok=True)
    arr.tofile(d / 'tokens' / 'a.train.bin')
    half = len(arr) // 2
    arr[:half].tofile(d / 'tokens' / 'b.train.bin')
    man = {'sources': {'a': {'train': [{'path': str(d / 'tokens' / 'a.train.bin'), 'tokens': int(len(arr))}]},
                       'b': {'train': [{'path': str(d / 'tokens' / 'b.train.bin'), 'tokens': int(half)}]}},
           'weights': {'a': 0.7, 'b': 0.3}}
    (d / 'manifest.json').write_text(json.dumps(man))
    (d / 'READY').write_text('x')
    return 'tiny'


def base_run(name, **kw):
    r = {'name': name, 'shape': {'width': 64, 'depth': 2}, 'param': 'sp', 'opt': {'name': 'adamw', 'lr': 3e-3}, 'seed': 11,
         'data': 'tiny', 'seq_len': 64, 'micro': 4, 'accum': 2, 'tokens': 512 * 40, 'eval': True, 'compile': MODE == 'gpu'}
    r.update(kw)
    return r


def t_e2e():
    make_dataset()
    lim = 2
    main = base_run('main', sched={'kind': 'constant', 'warmup': 5}, milestones=[512 * 20, 512 * 40], eval_at_milestones=True)
    rm = C.train(main, device=DEV, eval_limit=lim)
    assert C.milestone_path('main', 512 * 20).exists() and C.milestone_path('main', 512 * 40).exists()
    cont = base_run('cont', sched={'kind': 'constant', 'warmup': 5}, branch_from={'run': 'main', 'tokens': 512 * 20}, milestones=[512 * 40])
    C.train(cont, device=DEV, eval_limit=lim)
    a = torch.load(C.milestone_path('main', 512 * 40), weights_only=False)['model']
    b = torch.load(C.milestone_path('cont', 512 * 40), weights_only=False)['model']
    diff = max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
    tol, dup = 1e-5, None
    if MODE == 'gpu':
        # GPU 上原子加法等不确定性会被 Adam 放大：同一配置从头跑两次的差异作为基准，续跑的差异不应明显更大
        C.train(dict(main, name='main_dup', eval_at_milestones=False), device=DEV, eval_limit=lim)
        c = torch.load(C.milestone_path('main_dup', 512 * 40), weights_only=False)['model']
        dup = max(float((a[k].float() - c[k].float()).abs().max()) for k in a)
        tol = max(1e-5, 3 * dup)
    assert diff <= tol, f'从里程碑续跑与主线不一致：{diff:.2e}（同配置重跑的差异 {dup}）'
    cool = base_run('cool', sched={'kind': 'wsd', 'decay_from': 512 * 20, 'shape': '1-sqrt', 'final': 0.0, 'warmup': 5},
                    branch_from={'run': 'main', 'tokens': 512 * 20}, arith=True)
    rc = C.train(cool, device=DEV, eval_limit=lim, arith_limit=2)
    log = [json.loads(l) for l in (C.run_dir('cool') / 'train_log.jsonl').read_text().splitlines() if l.strip()]
    lrs = [x['lr'] for x in log if 'lr' in x]
    assert lrs and lrs[-1] < 3e-4, f'冷却分支最后的学习率应接近 0：{lrs}'
    # 断点续训：第一次在第 12 步后收到 STOP，存 latest；第二次接着跑完；结果与不中断完全一致
    calls = {'n': 0}

    def stop():
        calls['n'] += 1
        return calls['n'] == 13

    s1 = base_run('stopper', sched={'kind': 'wsd', 'decay_fraction': 0.25, 'final': 0.1, 'warmup': 5})
    try:
        C.train(s1, device=DEV, stop=stop, eval_limit=lim)
        raise AssertionError('应该收到 STOP')
    except C.StopJob:
        pass
    r1 = C.train(s1, device=DEV, eval_limit=lim)
    r2 = C.train(dict(s1, name='straight'), device=DEV, eval_limit=lim)
    d = abs(r1['bpb_weighted'] - r2['bpb_weighted'])
    assert d < (1e-4 if MODE == 'cpu' else 2e-2), f'断点续训与不中断结果不同：{r1["bpb_weighted"]} vs {r2["bpb_weighted"]}'
    for o in ('muon', 'soap', 'sf_adamw'):
        r = C.train(base_run(f'e2e_{o}', opt={'name': o, 'lr': 3e-3}, param='mup', mup={'base_width': 32}), device=DEV, eval_limit=lim)
        assert r['bpb_weighted'] > 0 and not r['diverged']
    return {'main_bpb': rm['bpb_weighted'], 'cool_bpb': rc['bpb_weighted'], 'arith': rc.get('arith_mean'), 'branch_diff': diff, 'rerun_diff': dup,
            'resume_diff': d, 'max_attn': rm.get('max_attn_logit')}


# ------------------------------------------------------------------ 调度器与数据计划 ----------
def t_plan():
    from . import lab
    J = lab.plan_jobs()
    names = {j['name']: j for j in J}
    for j in J:
        bf = j.get('branch_from')
        if bf and not isinstance(bf['run'], lab.Ref):
            p = names[bf['run']]
            assert int(bf['tokens']) in [int(t) for t in p.get('milestones', [])], f'{j["name"]} 的起点 {bf} 不是 {p["name"]} 的里程碑'
            assert J.index(p) < J.index(j), f'{j["name"]} 排在起点 {p["name"]} 前面'
            assert p['opt']['name'] == j['opt']['name'] and p['shape'] == j['shape'] and p['seed'] == j['seed'], f'{j["name"]} 与起点配置不一致'
    hours = 0.0
    for j in J:
        if j['type'] != 'train':
            continue
        n = C.param_count(j['shape'])
        start = int(j['branch_from']['tokens']) if j.get('branch_from') else 0
        hours += (int(j['tokens']) - start) / (162_000 * 125.2e6 / n) / 3600 * (1.1 if j['opt']['name'] == 'soap' else 1.0)
    blocks = {}
    for j in J:
        blocks[j['block']] = blocks.get(j['block'], 0) + 1
    state = {'jobs': {}}
    for lr in lab.LR_GRID:
        n = f'mup16_w256_lr{lr:.2e}'
        state['jobs'][n] = {'status': 'done', 'attempts': 1, 'x': {'lr': lr}}
        C.run_dir(n).mkdir(parents=True, exist_ok=True)
        (C.run_dir(n) / 'result.json').write_text(json.dumps({'bpb_weighted': 1.2 + (lr - 2 ** -8) ** 2 * 1e3}))
    R = lab.results(J, state)
    v = lab.resolve(lab.Ref('best_lr', 'mup16', 2.0), J, state, R)
    assert abs(v - 2 * 2 ** -8) < 1e-12, v
    r = lab.readiness(names['mupx_w768_x2'], J, state, R)
    assert r == 'wait', r
    return {'jobs': len(J), 'blocks': blocks, 'gpu_hours_est': round(hours, 1)}


def t_data_plan():
    from . import lab_data as D
    used = D.used_files()
    jobs = D.plan_source('probe', 'zh_fwedu', 50_000_000, used, root='3_4')
    assert jobs and all('3_4' in j['path'] for j in jobs), [j['path'] for j in jobs[:3]]
    jobs45 = D.plan_source('probe', 'zh_fwedu', 50_000_000, used, root='4_5')
    assert jobs45 and all('4_5' in j['path'] for j in jobs45)
    proxy = sum(len(D.plan_source('probe', s, int(D.PROXY_TOKENS * w), used)) for s, w in D.V1_MIX.items())
    D.build_arith('probe_arith', 20_000)
    n = (TMP / 'data' / 'probe_arith' / 'tokens' / 'arith' / '00000.train.bin').stat().st_size // 2
    assert n >= 20_000
    return {'zh_fwedu_3_4_files': len(jobs), 'zh_fwedu_4_5_files': len(jobs45), 'proxy_files': proxy, 'arith_tokens': n}


def t_decon():
    from . import lab_data as D
    from .data_eng import eval_docs
    D.build_eval_index()
    docs = [t for _, t in eval_docs('zh_fwedu')[:20]] + [t for _, t in eval_docs('en_fwedu')[:20]]
    hit = sum(D.is_contaminated(t, str(D.EVAL_INDEX)) for t in docs)
    edited = sum(D.is_contaminated(t[: int(len(t) * 0.97)] + ' 补充一句结尾。', str(D.EVAL_INDEX)) for t in docs if len(t) > 400)
    import random
    rng = random.Random(0)
    chars = '的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经'
    fake = [''.join(rng.choice(chars) for _ in range(600)) for _ in range(20)]
    false = sum(D.is_contaminated(x, str(D.EVAL_INDEX)) for x in fake)
    assert hit == len(docs), f'原文只命中 {hit}/{len(docs)}'
    assert false == 0, f'随机文本误报 {false}'
    return {'index_docs': int(np.load(D.EVAL_INDEX)['sig'].shape[0]), 'exact_hits': hit, 'edited_hits': edited, 'false_pos': false}


def t_ref_model():
    from .lab import V1_MODEL, V1_SHAPE
    ck = torch.load(V1_MODEL, map_location='cpu', weights_only=False, mmap=True)
    sd = ck.get('model', ck)
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    with torch.device('meta'):
        model = C.LabLM(C.shape_config(V1_SHAPE, max_seq_len=4096))
    want = set(model.state_dict())
    missing = [k for k in want if k not in sd and not k.endswith('lm_head.weight')]
    assert not missing, missing[:5]
    return {'keys': len(sd), 'config_in_ckpt': isinstance(ck, dict) and 'config' in ck}


if __name__ == '__main__':
    torch.set_num_threads(int(os.environ.get('LAB_SELFTEST_THREADS', '8')))
    check('schedules', t_schedules)
    for o in ('adamw', 'muon', 'soap', 'sf_adamw'):
        check(f'optimizer_{o}', t_optimizer(o))
    check('muon_rms', t_muon_rms)
    check('soap_basis', t_soap_basis)
    check('schedule_free_modes', t_schedule_free_modes)
    check('mup_init_groups', t_mup)
    check('coord_check', t_coord)
    check('end_to_end', t_e2e)
    check('plan', t_plan)
    if MODE == 'cpu':
        check('data_plan', t_data_plan)
        check('ref_model_keys', t_ref_model)
        check('eval_decontamination', t_decon)
    (TMP / 'selftest.json').write_text(json.dumps(RESULTS, ensure_ascii=False, indent=1))
    ok = all(r['ok'] for r in RESULTS)
    print(('ALL PASS' if ok else 'SOME FAILED') + f' -> {TMP}/selftest.json', flush=True)
    sys.exit(0 if ok else 1)

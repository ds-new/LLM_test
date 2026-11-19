"""路线 A：在 v1（0.48B，25B token）基础上继续预训练 + 退火 + 扩上下文（由 loop.py 在 config.mode == 'cpt' 时调用）。

复用主流程的多智能体（决策 / 数据 / 训练 / 评估 / 结果分析），只改编排：
  阶段：cpt_data1 → cpt_train1 → cpt_data2 → cpt_train2 → final_eval → passk_probe → analysis → done（随后守护脚本关机）
  * 第 1 段（cpt_train1）：从 v1 final.pt 出发（优化器状态取 v1 latest.pt，二者同一步），在 v1 / 对照实验都没用过的
    新文件上训练 10B token；学习率 warmup 500 步升到 2e-4 后恒定；上下文 1024。
  * 第 2 段（cpt_train2）：5B token 高质量配比（数学、代码、教育类、合成教材加权），上下文扩到 4096，学习率从 2e-4
    线性退火到 2e-5。两段共用一条全局步数轴，所以学习率在段间连续（WSD：第 1 段 = stable，第 2 段 = decay）。
  * 评估口径与 v1 完全一致：v1 的 val/test 分片、1024 长度、同样的固定块，所以 eval_history 可以直接接在 v1 曲线后面比；
    第 1 段开始前先评估一次 v1 final.pt 作为基线；最终评估额外报告 4096 长度下的验证 loss（长上下文效果）。
  * 数据盘空间：第 2 段数据构建前删除第 1 段训练分片；第 2 段初始化完成后删除第 1 段检查点。
"""

import json
import multiprocessing
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections import Counter
from pathlib import Path
import shutil
import types

from .common import RUN_DIR, SCRATCH_DIR, FatalError, RestartRequested, StopRequested, deep_merge, free_gb, heartbeat, now, read_json, stamp, write_json
from .decision_agent import DecisionAgent
from .data_agent import load_bbpe, process_file
from .sources import SOURCES

V1 = Path('/root/autodl-tmp/a800_pretrain_v1')
V1_TOK_BBPE = V1 / 'tokenizer' / 'bbpe.json'
V1_TOK_HF = V1 / 'tokenizer' / 'tokenizer_hf.json'
V1_FINAL = V1 / 'model' / 'final.pt'
V1_LATEST = Path('/root/a800_scratch/checkpoints/latest.pt')
OTHER_RECEIPTS = [Path('/root/autodl-tmp/a800_tok_ab/data/receipts')]
EVAL_SEQ = 1024
NAME = 'decision'

CPT_DEFAULT = {
    'stage1': {'tokens': 10_000_000_000, 'seq_len': 1024, 'micro': 32, 'accum': 16, 'peak_lr': 2e-4, 'warmup': 500,
               'weights': {'zh_cci3': 0.28, 'zh_fwedu': 0.14, 'zh_industry': 0.13, 'en_fwedu': 0.18, 'en_cosmo': 0.07,
                           'code': 0.12, 'math': 0.08}},
    'stage2': {'tokens': 5_000_000_000, 'seq_len': 4096, 'micro': 8, 'accum': 16,
               'weights': {'zh_cci3': 0.17, 'zh_fwedu': 0.18, 'zh_industry': 0.06, 'en_fwedu': 0.15, 'en_cosmo': 0.12,
                           'code': 0.15, 'math': 0.17}},
    'workers': 8,
    'load_v1_optimizer': True,
    'delete_v1_latest': False,
    'avg_last_steps': 1600,
    'midcheck': {'enabled': False, 'at_tokens': 2_000_000_000, 'probe_tokens': 500_000_000, 'min_gain': 0.005, 'action': 'route_b'},
}


def eval_manifest():
    m = read_json(V1 / 'data' / 'tokens' / 'manifest.json')
    return {'tokenizer': {'path': str(V1_TOK_BBPE), 'vocab_size': m['tokenizer']['vocab_size']},
            'sources': {n: {split: [{'path': str(V1 / s['path']), 'tokens': s['tokens']} for s in e[split]]
                            for split in ('val', 'test')} for n, e in m['sources'].items()}}


def used_files(extra_dirs):
    used = set()
    for p in (V1 / 'data' / 'receipts').glob('*/*.json'):
        r = read_json(p)
        used.add((r['source'], r['index']))
    for d in list(OTHER_RECEIPTS) + list(extra_dirs):
        for p in Path(d).glob('*/*.json'):
            r = read_json(p)
            used.add((r['source'], r['index']))
    return used


class CPTDecisionAgent(DecisionAgent):
    # 第 1 段从已退火的 v1 出发、学习率重新升高，验证 loss 会先高于基线：发散判定放宽到比基线高 10%
    DIVERGE_RATIO = 1.10

    def __init__(self, config, bus=None, board=None):
        super().__init__(config, bus, board)
        if self.board.get('phase') == 'selftest':
            self.board.update(phase='cpt_data1', mode='cpt')
        self.ccfg = deep_merge(CPT_DEFAULT, config.get('cpt') or {})

    # ---------------- 数据智能体：按阶段构建 ----------------
    def build_stage(self, stage):
        s = self.ccfg[f'stage{stage}']
        root = RUN_DIR / 'data' / f'stage{stage}'
        rec_dir = root / 'receipts'
        # parquet 临时下载：第 1 段放系统盘（数据盘要留给 20GB 的 token），第 2 段放数据盘（系统盘此时有两份检查点）
        dl_root = (SCRATCH_DIR if stage == 1 else RUN_DIR) / 'downloads'
        extra = [RUN_DIR / 'data' / f'stage{k}' / 'receipts' for k in (1, 2) if k != stage]
        used = used_files(extra)
        fl = read_json(V1 / 'data' / 'file_lists.json')
        v1_done = {}
        for p in (V1 / 'data' / 'receipts').glob('*/*.json'):
            r = read_json(p)
            v1_done.setdefault(r['source'], {})[r['index']] = r['stats']
        specs = {x['name']: x for x in SOURCES}
        dcfg = self.config['data']
        dedup = self.stage2_dedup() if stage == 2 else {}
        process_file_excl = None
        if dedup:
            from .global_dedup import process_file_excl
        jobs = []
        for name, w in s['weights'].items():
            files = fl[name]
            size = {f['index']: f['size'] for f in files}
            d = v1_done.get(name, {})
            tpb = sum(x.get('tokens_train', 0) for x in d.values()) / max(1, sum(size[i] for i in d if i in size))
            target = int(s['tokens'] * w)
            if name in dedup:  # 剔除后每个文件产出的 token 变少：按预计剔除比例多选文件，保持本段 token 量
                target = int(target / max(0.5, 1 - dedup[name]['drop_rate']))
            est = 0.0
            for f in files:
                if (name, f['index']) in used:
                    continue
                if est >= target:
                    break
                cap = int((target - est) * 1.02)
                est += f['size'] * tpb
                if (rec_dir / name / f'{f["index"]:05d}.json').exists():
                    continue
                spec = specs[name]
                job = {'source': name, 'kind': spec['kind'], 'format': spec['format'], 'fields': spec['fields'],
                             'url': f['url'], 'path': f['path'], 'size': f['size'], 'index': f['index'], 'mode': 'build',
                             'val_fraction': dcfg['val_fraction'], 'test_fraction': dcfg['test_fraction'],
                             'download_path': str(dl_root / name / f'{f["index"]:05d}.parquet'),
                             'bbpe_path': str(V1_TOK_BBPE), 'hf_path': str(V1_TOK_HF),
                             'out_dir': str(root / 'tokens' / name), 'token_cap': cap}
                if name in dedup:
                    job.update(exclude_path=dedup[name]['exclude_path'], exclude_dup=dedup[name]['exclude_dup'])
                jobs.append(job)
        self.bus.post(NAME, 'data', 'directive', {'task': f'build_stage{stage}', 'jobs': len(jobs), 'tokens': s['tokens'],
                                                   'weights': s['weights']})
        need_gb = s['tokens'] * 2 / 1e9 * 1.05 + 2 if jobs else 0
        if free_gb(RUN_DIR) < need_gb:
            raise FatalError(f'数据盘可用 {free_gb(RUN_DIR):.1f}GB，不够放第 {stage} 段的 token（约需 {need_gb:.1f}GB）')
        workers = self.ccfg['workers']
        pending, futs, attempts, done_n = list(jobs), {}, Counter(), 0
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as ex:
            while pending or futs:
                self.check_control()
                while pending and len(futs) < workers:
                    j = pending.pop(0)
                    futs[ex.submit(process_file_excl if j.get('exclude_path') else process_file, j)] = j
                finished, _ = wait(list(futs), timeout=60, return_when=FIRST_COMPLETED)
                for fut in finished:
                    j = futs.pop(fut)
                    try:
                        res = fut.result()
                        write_json(rec_dir / j['source'] / f'{j["index"]:05d}.json', res)
                        done_n += 1
                    except Exception as exc:  # noqa: BLE001
                        attempts[(j['source'], j['index'])] += 1
                        if attempts[(j['source'], j['index'])] < 3:
                            pending.append(j)
                        else:
                            self.bus.post('data', NAME, 'report', {'action': 'file_skipped', 'source': j['source'],
                                                                   'index': j['index'], 'error': f'{type(exc).__name__}: {str(exc)[:300]}'})
                heartbeat(f'build stage{stage} {done_n}/{len(jobs)}')
        sources, totals = {}, {}
        for name in s['weights']:
            shards = []
            for p in sorted((root / 'tokens' / name).glob('*.train.bin')):
                n = p.stat().st_size // 2
                if n > s['seq_len'] + 1:
                    shards.append({'path': str(p), 'tokens': n})
            sources[name] = {'train': shards}
            totals[name] = sum(x['tokens'] for x in shards)
        manifest = {'tokenizer': {'path': str(V1_TOK_BBPE), 'vocab_size': 32000}, 'bos_id': 2, 'eos_id': 3,
                    'dtype': 'uint16', 'sources': sources}
        write_json(root / 'manifest.json', manifest)
        info = {'tokens': totals, 'total_B': round(sum(totals.values()) / 1e9, 3), 'files': done_n,
                'data_disk_free_gb': round(free_gb(RUN_DIR), 1)}
        self.bus.post('data', NAME, 'report', {'action': f'stage{stage}_data_done', **info})
        if sum(totals.values()) < 0.8 * s['tokens']:
            raise FatalError(f'第 {stage} 段数据只构建了 {info["total_B"]}B token，不足计划的 80%')
        return info

    def stage2_dedup(self):
        """数据工程（CPU 阶段，data_eng.py）的结论：第 2 段哪些来源启用全局去重、剔除评测污染。
        没有结论或读取出错时返回空，第 2 段按原流程构建（附加功能不影响主流程）。"""
        path = Path(self.ccfg.get('global_dedup_path') or '/root/autodl-tmp/data_eng/stage2_dedup.json')
        try:
            dec = read_json(path)
            if not dec or dec.get('status') != 'done' or not Path(dec['exclude_path']).exists():
                self.decide('data', 'global_dedup_skipped', f'没有找到数据工程的去重结论（{path}），第 2 段按原流程构建')
                return {}
            plan = {}
            for name, d in dec['sources'].items():
                if d.get('apply_dup') or d.get('eval_docs', 0) > 0:
                    plan[name] = {'exclude_path': dec['exclude_path'], 'exclude_dup': bool(d.get('apply_dup')),
                                  'drop_rate': float(d.get('drop_rate', 0.0))}
            why = ('第 2 段按数据工程结论构建：' + '；'.join(
                f'{n} 预计剔除 {p["drop_rate"]:.1%}（{"去重 + 评测污染" if p["exclude_dup"] else "只剔除评测污染"}）' for n, p in plan.items())
                if plan else '数据工程结论：各来源重复率都低于阈值、没有评测污染，第 2 段不做额外剔除')
            self.decide('data', 'global_dedup', why, sources=dec['sources'])
            return plan
        except Exception as exc:  # noqa: BLE001
            self.decide('data', 'global_dedup_skipped', f'读取去重结论出错（{type(exc).__name__}: {exc}），第 2 段按原流程构建')
            return {}

    def phase_cpt_data1(self):
        self.board.set('stage1_data', self.build_stage(1))
        return 'cpt_train1'

    def phase_cpt_data2(self):
        if self.trainer is not None:  # 与第 1 段训练同一进程：先重启，彻底释放显存，再构建数据和训练第 2 段
            raise RestartRequested('restart between stage 1 and stage 2')
        freed = 0
        for p in (RUN_DIR / 'data' / 'stage1' / 'tokens').glob('*/*.train.bin'):
            freed += p.stat().st_size
            p.unlink()
        self.decide('data', 'free_stage1_data', f'第 1 段已训练完，删除其训练分片，释放 {freed / 1e9:.1f}GB 给第 2 段数据')
        self.board.set('stage2_data', self.build_stage(2))
        return 'cpt_train2'

    # ---------------- 计划：两段共用全局步数轴 ----------------
    def ensure_plan(self, stage):
        s1, s2 = self.ccfg['stage1'], self.ccfg['stage2']
        tps = s1['micro'] * s1['accum'] * s1['seq_len']
        if s2['micro'] * s2['accum'] * s2['seq_len'] != tps:
            raise FatalError('两段的每步 token 数必须相同，学习率曲线才能在段间连续')
        steps1, steps2 = int(s1['tokens'] // tps), int(s2['tokens'] // tps)
        if self.board.get('stage1_end_step'):  # 第 1 段被探针结论提前结束（见 end_stage1_early）
            steps1 = int(self.board.get('stage1_end_step'))
        plan = self.board.get('plan')
        if not plan or plan.get('stage') != stage:
            base = read_json(V1 / 'state.json')['plan']
            common = {k: base[k] for k in ('name', 'model', 'params', 'precision', 'compile', 'eval_weights')}
            common.update(tokens_per_step=tps, warmup_steps=s1['warmup'], peak_lr=s1['peak_lr'], stage=stage,
                          stage1_steps=steps1, stage2_steps=steps2, init_from=str(V1_FINAL))
            if stage == 1:
                plan = {**common, 'micro_batch': s1['micro'], 'accum': s1['accum'], 'seq_len': s1['seq_len'],
                        'total_steps': steps1, 'decay_fraction': 0.0, 'weights': dict(s1['weights']),
                        'initial_weights': dict(s1['weights'])}
            else:
                total = steps1 + steps2
                plan = {**common, 'micro_batch': s2['micro'], 'accum': s2['accum'], 'seq_len': s2['seq_len'],
                        'total_steps': total, 'decay_fraction': (steps2 + 0.5) / total, 'weights': dict(s2['weights']),
                        'initial_weights': dict(s2['weights'])}
            plan['planned_tokens'] = plan['total_steps'] * tps
            self.board.set('plan', plan)
            self.decide('trainer', f'plan_stage{stage}', f'第 {stage} 段计划：{plan["total_steps"]:,} 步（全局），上下文 {plan["seq_len"]}，'
                        f'峰值学习率 {plan["peak_lr"]}', weights=plan['weights'])
        # 决策智能体的"延长计划"规则以 target_tokens 为上限：设成本段计划量，不让它改动段边界
        self.config['train']['target_tokens'] = plan['planned_tokens']
        return plan

    def phase_cpt_train1(self):
        self.ensure_plan(1)
        self.phase_train()
        self.decide('all', 'stage1_done', '第 1 段（新数据续训）完成，进入第 2 段数据构建（退火 + 4096 上下文）')
        return 'cpt_data2'

    def phase_cpt_train2(self):
        self.ensure_plan(2)
        self.phase_train()
        return 'final_eval'

    # ---------------- 训练 / 评估智能体的装配 ----------------
    def setup_training(self):
        import torch
        from .eval_agent import EvalAgent
        from .trainer_agent import TrainerAgent
        plan = self.board.get('plan')
        stage = plan['stage']
        manifest = read_json(RUN_DIR / 'data' / f'stage{stage}' / 'manifest.json')
        tok_dict = {'type': 'scratch_bbpe', 'model': read_json(V1_TOK_BBPE)}
        self.trainer = TrainerAgent(self.config, self.bus, manifest, plan, self.config['device'], tok_dict,
                                    ckpt_dir=SCRATCH_DIR / f'cpt_stage{stage}')
        base_model_config = self.trainer.model_config

        def model_config():  # RoPE 位置表按 max_seq_len 预先算好：两段都建到 4096，第 1 段也能测长上下文基线
            from ..config import ModelConfig
            cfg = base_model_config().to_dict()
            cfg['max_seq_len'] = max(cfg['max_seq_len'], 4096)
            return ModelConfig(**cfg)
        self.trainer.model_config = model_config
        self.trainer.build()
        self.evaluator = EvalAgent(self.config, self.bus, eval_manifest(), load_bbpe(V1_TOK_BBPE), self.config['device'])
        raw_evaluate = self.evaluator.evaluate

        def evaluate(trainer, final=False):
            # 评估口径固定为 v1 的 1024 长度固定块，第 2 段（4096）也一样，保证曲线可比
            view = types.SimpleNamespace(model=trainer.model, step=trainer.step, tokens_seen=trainer.tokens_seen,
                                         plan={**trainer.plan, 'seq_len': EVAL_SEQ})
            return raw_evaluate(view, final=final)
        self.evaluator.evaluate = evaluate
        if self.trainer.step == 0:
            key = f'stage{stage}_started'
            if self.board.get(key):
                raise FatalError(f'第 {stage} 段已经开始过，但找不到检查点 {self.trainer.ckpt_dir / "latest.pt"}：拒绝重新初始化')
            self.init_stage(stage, torch)
            self.board.set(key, stamp())

    def init_stage(self, stage, torch):
        t = self.trainer
        if stage == 1:
            ck = torch.load(V1_FINAL, map_location='cpu', weights_only=False)
            t.model.load_state_dict(ck['model'])
            opt = False
            if self.ccfg['load_v1_optimizer'] and V1_LATEST.exists():
                ck2 = torch.load(V1_LATEST, map_location='cpu', weights_only=False)
                if ck2.get('step') == ck.get('step'):
                    t.optimizer.load_state_dict(ck2['optimizer'])
                    opt = True
                del ck2
                if self.ccfg.get('delete_v1_latest'):
                    V1_LATEST.unlink(missing_ok=True)  # 用户已同意：路线 A 读完 v1 优化器状态后删除，释放系统盘
            t.save_latest()
            self.decide('trainer', 'init_from_v1', f'从 v1 final.pt（第 {ck.get("step")} 步）初始化；优化器状态'
                        + ('沿用 v1 latest.pt' if opt else '重新初始化'), optimizer_loaded=opt)
            record = self.evaluator.evaluate(t)
            history = self.board.get('eval_history', [])
            history.append({k: record[k] for k in ('time', 'step', 'tokens_seen', 'val_loss', 'val_by_source', 'cloze_acc',
                                                   'sample_rep4', 'greedy_rep4')})
            self.board.update(eval_history=history, best_val={'val_loss': record['val_loss'], 'step': 0},
                              baseline={'val_loss': record['val_loss'], 'val_4096': self.long_context_loss()})
            self.decide('all', 'baseline', f'基线（v1 final.pt）：验证 loss {record["val_loss"]}，4096 长度验证 loss '
                        f'{self.board.get("baseline")["val_4096"]}')
        else:
            src = SCRATCH_DIR / 'cpt_stage1' / 'latest.pt'
            ck = torch.load(src, map_location='cpu', weights_only=False)
            if ck['step'] != self.board.get('plan')['stage1_steps']:
                raise FatalError(f'第 1 段检查点停在第 {ck["step"]} 步，与计划 {self.board.get("plan")["stage1_steps"]} 不一致')
            t.model.load_state_dict(ck['model'])
            t.optimizer.load_state_dict(ck['optimizer'])
            t.step, t.tokens_seen = ck['step'], ck['tokens_seen']
            t.train_seconds, t.lr_scale = ck.get('train_seconds', 0.0), ck.get('lr_scale', 1.0)
            del ck
            (SCRATCH_DIR / 'cpt_stage1' / 'prev.pt').unlink(missing_ok=True)  # 降低系统盘峰值占用
            t.save_latest()
            shutil.rmtree(SCRATCH_DIR / 'cpt_stage1', ignore_errors=True)
            self.decide('trainer', 'init_stage2', f'第 2 段从第 1 段终点（第 {t.step:,} 步）接续：模型与优化器沿用，'
                        '数据换成退火配比、上下文扩到 4096；第 1 段检查点已删除以释放系统盘')

    def long_context_loss(self):
        import torch  # noqa: F401
        plan = self.board.get('plan')
        losses = self.evaluator.loss_by_source(self.trainer.model, 'val', 4096, plan.get('precision', 'bf16'), batch=4)
        self.trainer.model.train()
        w = plan['eval_weights']
        return round(sum(losses[n] * w.get(n, 0) for n in losses) / max(1e-9, sum(w.get(n, 0) for n in losses)), 4)

    def phase_final_eval(self):
        nxt = super().phase_final_eval()
        if self.trainer is None:
            self.setup_training()
        val_4096 = self.long_context_loss()
        base = self.board.get('baseline') or {}
        self.board.set('long_context', {'val_4096': val_4096, 'baseline_val_4096': base.get('val_4096'),
                                        'val_1024': (self.board.get('final') or {}).get('val_loss'),
                                        'baseline_val_1024': base.get('val_loss')})
        self.decide('all', 'long_context', f'4096 长度验证 loss：基线 {base.get("val_4096")} → {val_4096}；'
                    f'1024 长度：基线 {base.get("val_loss")} → {(self.board.get("final") or {}).get("val_loss")}')
        try:
            self.eval_average()
        except Exception as exc:  # noqa: BLE001
            self.decide('all', 'average_failed', f'检查点平均评估出错（{type(exc).__name__}: {str(exc)[:200]}），不影响已保存的模型')
        return nxt


    # ---------------- 中途效果检查（v2 明显更好时启用）：退火探针 ----------------
    def review(self, seg, rec, deadline):
        super().review(seg, rec, deadline)
        for key in ('midcheck', 'midcheck2'):  # midcheck2：第 2 次探针（与第 1 次探针比较，见 end_stage1_early）
            self.maybe_midcheck(key)
        try:
            self.update_average()
        except Exception as exc:  # noqa: BLE001  平均只是附加实验，出错不影响训练
            self.decide('all', 'average_failed', f'检查点平均出错（{type(exc).__name__}: {str(exc)[:200]}），跳过')

    def maybe_midcheck(self, key='midcheck'):
        """第 1 段学习率恒定，验证 loss 天然高于 v1（v1 末尾已退火），不能直接比。
        做法：到 at_tokens 时把现场存成 latest.pt，用 probe_tokens 做一次快速线性退火并测验证 loss，再从 latest.pt 恢复现场。
        探针期间不写检查点：中途崩溃重启会从探针前的现场重来，第 1 段的权重不会被探针污染。
        探针结果低于 v1 基线 min_gain 以上 → 路线 A 有效，继续；否则按 action 处理（route_b：停止 A 并启动路线 B）。"""
        mc = self.ccfg.get(key) or {}
        t = self.trainer
        plan = t.plan
        if not mc.get('enabled') or self.board.get(key) or plan.get('stage') != 1 or t.tokens_seen < mc['at_tokens']:
            return
        t.save_latest()
        start, steps = t.step, max(1, int(mc['probe_tokens'] // plan['tokens_per_step']))
        self.decide('trainer', f'{key}_start', f'中途效果检查（{key}）：在第 {start:,} 步（新数据 {t.tokens_seen / 1e9:.2f}B token）做 {steps} 步快速退火探针，'
                    '结束后恢复现场继续第 1 段', probe_steps=steps)
        log_path, bad, losses = t.log_path, 0, None
        t.plan = {**plan, 'total_steps': start + steps, 'decay_fraction': (steps + 0.5) / (start + steps)}
        t.log_path = RUN_DIR / ('midcheck_train_log.jsonl' if key == 'midcheck' else f'{key}_train_log.jsonl')
        t.save_latest = lambda: None
        try:
            while t.step < start + steps and bad < 3:
                self.check_control()
                rep = t.run_segment(self.config['train']['segment_minutes'] * 60, now() + 10 ** 9)
                bad += int(rep['stop_reason'] == 'nonfinite')
            if t.step >= start + steps:
                losses = self.evaluator.loss_by_source(t.model, 'val', EVAL_SEQ, plan.get('precision', 'bf16'))
        finally:
            del t.save_latest
            t.plan, t.log_path = plan, log_path
            t.load(t.ckpt_dir / 'latest.pt')
            t.model.train()
        v1 = (self.board.get('baseline') or {}).get('val_loss')
        ref = (self.board.get(mc['baseline_from']) or {}) if mc.get('baseline_from') else None  # 例如与第 1 次探针比较
        base = ref.get('probe_val') if ref is not None else v1
        label = f'第 1 次探针（{ref.get("tokens_B")}B token 处）{base}' if ref is not None else f'v1 基线 {base}'
        if losses is None or base is None:
            self.board.set(key, {'step': start, 'probe_val': None if losses is None else self.weighted(losses),
                                 'baseline_val': base, 'passed': None, 'time': stamp()})
            self.decide('all', f'{key}_inconclusive', '中途检查的退火探针出现非有限 loss 或缺少比较基准，没有得到结论；现场已恢复，继续路线 A')
            return
        probe = self.weighted(losses)
        passed = probe <= base - mc['min_gain']
        result = {'step': start, 'tokens_B': round(t.tokens_seen / 1e9, 3), 'probe_steps': steps, 'probe_val': probe,
                  'baseline_val': base, 'baseline_from': mc.get('baseline_from'), 'v1_val': v1, 'by_source': losses,
                  'passed': passed, 'action': mc.get('action'), 'time': stamp()}
        self.board.set(key, result)
        if passed:
            self.decide('all', f'{key}_pass', f'中途检查通过：快速退火后验证 loss {probe}，低于 {label}（要求至少低 {mc["min_gain"]}），继续路线 A')
            return
        self.decide('all', f'{key}_fail', f'中途检查未通过：快速退火后验证 loss {probe}，没有低于 {label} 至少 {mc["min_gain"]}')
        if mc.get('action') == 'route_b':
            (RUN_DIR / 'NEXT.sh').write_text('cd /root/autodl-tmp/verl_megatron_sglang && /root/a800_venv/bin/python -m '
                                             'decoder_from_scratch.a800.cpt_launch route_b\n')
            self.decide('all', 'switch_to_route_b', 'v2 tokenizer 对照明显更好、路线 A 效果不达标：停止路线 A（检查点保留），改走路线 B（v2 从头预训练）')
            raise StopRequested('midcheck failed: switch to route B')
        if mc.get('action') == 'end_stage1':
            self.end_stage1_early(f'第 {start:,} 步（新数据 {t.tokens_seen / 1e9:.2f}B token）的探针 {probe} 没有比 {label} 低 {mc["min_gain"]} 以上')

    def end_stage1_early(self, why):
        """探针说明继续第 1 段已没有收益时（用户 11-18 同意的规则），把第 1 段终点提前：
        终点设为当前步 + 1，下一段只训 1 步就以 plan_complete 结束（避开 0 步的段），随后照常进入第 2 段；
        第 2 段计划的 stage1_steps 读 board['stage1_end_step']（见 ensure_plan），与第 1 段最后的检查点一致。"""
        t = self.trainer
        end = t.step + 1
        tps = t.plan['tokens_per_step']
        t.plan.update(stage1_steps=end, total_steps=end, planned_tokens=end * tps)
        self.config['train']['target_tokens'] = end * tps
        self.board.set('stage1_end_step', end)
        self.board.update(plan=t.plan)
        self.decide('all', 'end_stage1_early', f'{why}：第 1 段提前在第 {end:,} 步（新数据 {end * tps / 1e9:.2f}B token）结束，随后进入第 2 段',
                    stage1_end_step=end)

    # ---------------- 检查点平均（第 2 段末尾；几乎零成本的收尾技巧）----------------
    def weighted(self, losses):
        w = self.board.get('plan')['eval_weights']
        return round(sum(losses[n] * w.get(n, 0) for n in losses) / max(1e-9, sum(w.get(n, 0) for n in losses)), 4)

    def update_average(self):
        """第 2 段最后 avg_last_steps 步内，每次评估后把当前权重并入一个滑动的等权平均（只占一个文件）。"""
        import torch
        t, plan = self.trainer, self.trainer.plan
        info = self.board.get('ckpt_avg') or {'steps': []}
        if plan.get('stage') != 2 or t.step < plan['total_steps'] - int(self.ccfg['avg_last_steps']) or t.step in info['steps'] \
                or 'val_avg' in info:
            return
        path = RUN_DIR / 'model' / 'avg_running.pt'
        if not path.exists():
            info = {'steps': []}
        n = len(info['steps'])
        state = {k: v.detach().float().cpu() for k, v in t.model.state_dict().items()}
        if n:
            avg = torch.load(path, map_location='cpu', weights_only=True)['model']
            state = {k: (avg[k] * n + v) / (n + 1) for k, v in state.items()}
            del avg
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.tmp')
        torch.save({'model': state, 'n': n + 1}, tmp)
        os.replace(tmp, path)
        self.board.set('ckpt_avg', {'steps': info['steps'] + [t.step]})

    def eval_average(self):
        """最终评估后：把平均权重装进模型，按同样口径测验证 loss（1024 与 4096），再换回最终权重。更好则另存 model/avg.pt。"""
        import torch
        info = self.board.get('ckpt_avg') or {}
        path = RUN_DIR / 'model' / 'avg_running.pt'
        if len(info.get('steps', [])) < 2 or not path.exists() or 'val_avg' in info:
            return
        t = self.trainer
        precision = t.plan.get('precision', 'bf16')
        final_state = {k: v.detach().cpu().clone() for k, v in t.model.state_dict().items()}
        final_val = (self.board.get('final') or {}).get('val_loss')
        try:
            t.model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model'])
            val_avg = self.weighted(self.evaluator.loss_by_source(t.model, 'val', EVAL_SEQ, precision))
            val_avg_4096 = self.long_context_loss()
            better = final_val is not None and val_avg < final_val
            if better:
                t.save_weights(RUN_DIR / 'model' / 'avg.pt', val_loss=val_avg, averaged_steps=info['steps'])
        finally:
            t.model.load_state_dict(final_state)
            t.model.train()
        path.unlink(missing_ok=True)
        info.update(val_avg=val_avg, val_final=final_val, val_avg_4096=val_avg_4096,
                    val_final_4096=(self.board.get('long_context') or {}).get('val_4096'), better=better)
        self.board.set('ckpt_avg', info)
        self.decide('all', 'checkpoint_average', f'检查点平均（第 2 段最后 {len(info["steps"])} 个评估点等权平均）：验证 loss {val_avg}，'
                    f'最终权重 {final_val}；4096 长度 {val_avg_4096} 对 {info["val_final_4096"]}。'
                    + ('平均权重更好，已另存 model/avg.pt' if better else '平均没有带来提升，保留最终权重'))

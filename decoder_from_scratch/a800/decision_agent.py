"""决策智能体：编排另外五个智能体（数据清洗 / 结构与算法 / 训练执行与监控 / 测试评估 / 结果分析），审核它们的报告，按明确的规则做决策，并把每条决策的理由写进 events.jsonl。

阶段：selftest → benchmark → sample → tokenizer → fast_tokenizer → build_data → plan → train → final_eval
      → weight_edit → passk_probe → token_probe → profile → analysis → done

主要决策规则（训练阶段每段约 1 小时评估一次后执行）：
  1. 稳定性：连续 3 步非有限 loss，或验证 loss 比最佳值高 5% → 回滚到上一个检查点，学习率 ×0.7；
  2. 计划：用最近 4 段吞吐的中位数预测能否在时间上限内跑完；每次最多调整 8%；进入衰减阶段后不再改总步数；
     总 token 永远不超过用户选择的 25B；
  3. 重复：预测某来源到训练结束会被重复超过 4 遍 → 按比例降低该来源的采样权重；
  4. 遗忘：结果分析智能体判定某来源显著变差（最近 8 次评估回归 t≥3 且拟合上升 >3σ）而总体没有变差
     → 该来源权重 ×1.2（不超过初始值 1.5 倍；同一分析窗口内不重复调整）；
  5. 结束：计划完成或到达时间上限 → 最终评估（含测试集）→ 保存最终模型 → 各项探测 → 结果分析报告 → 守护脚本关机；
  6. 异常：连续 4 次回滚仍不稳定、训练中检查点丢失等无法自动恢复的情况 → FatalError → 退出码 2 → 守护脚本关机。
"""

import math
import shutil
import time
import traceback

from .common import (RUN_DIR, SCRATCH_DIR, Blackboard, Bus, FatalError, RestartRequested, StopRequested,
                     control_signal,
                     free_gb, heartbeat, now, read_json, stamp, write_text)
from .data_agent import DataAgent, load_bbpe
from .sources import SOURCES

NAME = 'decision'


class DecisionAgent:
    DIVERGE_RATIO = 1.05  # 验证 loss 比最佳值高出这个比例判定为发散

    def __init__(self, config, bus=None, board=None):
        self.config = config
        self.bus = bus or Bus()
        self.board = board or Blackboard()
        if not self.board.get('started'):
            self.board.update(started=now(), started_text=stamp(), phase='selftest', decisions=0)
        self.data = DataAgent(config, self.bus)
        self.architect = self.trainer = self.evaluator = None

    # ---------------- 工具 ----------------
    def decide(self, recipient, what, why, **detail):
        self.board.set('decisions', self.board.get('decisions', 0) + 1)
        return self.bus.post(NAME, recipient, 'decision', {'decision': what, 'why': why, **detail})

    def check_control(self):
        signal = control_signal()
        if signal == 'stop':
            raise StopRequested('STOP file present')
        if signal == 'restart':
            raise RestartRequested('RESTART file present')

    def weights(self):
        return {s['name']: s['weight'] for s in SOURCES}

    def get_architect(self):
        if self.architect is None:
            from .architect_agent import ArchitectAgent
            self.architect = ArchitectAgent(self.config, self.bus)
        return self.architect

    def get_analyst(self):
        if getattr(self, 'analyst', None) is None:
            from .analyst_agent import AnalystAgent
            self.analyst = AnalystAgent(self.config, self.bus)
        return self.analyst

    def rollback(self, why):
        """回滚到上一个检查点并把学习率 ×0.7；连续第 4 次回滚判定为无法自动恢复。"""
        n = self.board.get('consecutive_rollbacks', 0) + 1
        self.board.set('consecutive_rollbacks', n)
        if n > 3:
            self.decide('all', 'abort', f'连续 {n} 次回滚后仍不稳定（{why}），判定无法自动恢复，停止训练并关机')
            raise FatalError(f'{n} consecutive rollbacks: {why}')
        self.decide('trainer', 'rollback', why, consecutive=n)
        self.trainer.rollback(0.7)

    # ---------------- 主循环 ----------------
    def run(self):
        while True:
            self.check_control()
            phase = self.board.get('phase')
            if phase == 'done':
                self.write_report()
                return 'done'
            heartbeat(phase)
            self.bus.post(NAME, 'all', 'phase', {'phase': phase})
            started = now()
            next_phase = getattr(self, 'phase_' + phase)()
            timings = self.board.get('phase_hours', {})
            timings[phase] = round(timings.get(phase, 0) + (now() - started) / 3600, 3)
            self.board.update(phase=next_phase, phase_failures=0, phase_hours=timings)
            self.write_report()

    # ---------------- 阶段 ----------------
    def phase_selftest(self):
        import torch
        from .smoke import pipeline_smoke
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA 不可用：实例可能处于无卡模式，请以有卡模式开机')
        arch = self.get_architect()
        self.bus.post(NAME, 'architect', 'directive', {'task': 'self_test'})
        result = {'self_test': arch.self_test(torch.device('cuda'))}
        self.bus.post('architect', NAME, 'report', {'action': 'self_test', **result['self_test']})
        compile_model = bool(self.config['train']['compile'])
        self.bus.post(NAME, 'trainer', 'directive', {'task': 'pipeline_smoke', 'compile': compile_model})
        try:
            result['smoke'] = pipeline_smoke(self.config, torch.device('cuda'), compile_model)
        except Exception as exc:
            if not compile_model:
                raise
            self.decide('trainer', 'disable_compile', f'torch.compile 冒烟测试失败（{type(exc).__name__}: {str(exc)[:160]}），改用 eager 模式重试')
            self.config['train']['compile'] = False
            self.board.set('compile_disabled', True)
            result['smoke'] = pipeline_smoke(self.config, torch.device('cuda'), False)
        self.bus.post('trainer', NAME, 'report', {'action': 'pipeline_smoke', **result['smoke']})
        self.board.set('selftest', result)
        if not (result['self_test']['passed'] and result['smoke']['passed']):
            raise RuntimeError(f'self test failed: {result}')
        self.decide('architect', 'run_benchmark', '结构自检（因果性、KV Cache、bf16 过拟合）与训练/恢复/评估冒烟测试都通过；'
                    '数据准备期间 GPU 空闲，先做吞吐基准')
        return 'benchmark'

    def run_benchmark(self, compile_model):
        tcfg = self.config['train']
        self.bus.post(NAME, 'architect', 'directive', {'task': 'benchmark', 'seq_len': tcfg['seq_len'], 'compile': compile_model})
        rows = self.get_architect().benchmark(tcfg['seq_len'], 32000, compile_model=compile_model)
        self.board.set('benchmark', rows)
        pick = self.get_architect().choose(rows)
        self.bus.post('architect', NAME, 'proposal', {k: v for k, v in pick.items() if k != 'trials'})
        self.board.set('model_choice', {k: v for k, v in pick.items() if k != 'trials'})
        return pick

    def phase_benchmark(self):
        tcfg = self.config['train']
        pick = self.run_benchmark(tcfg['compile'] and not self.board.get('compile_disabled'))
        hours = tcfg['target_tokens'] / pick['tok_s'] / 3600
        self.decide('data', 'start_data', f'{pick["rationale"]} 按此吞吐训练 {tcfg["target_tokens"] / 1e9:.0f}B token 约需 '
                    f'{hours / 24:.1f} 天（不含评估）。开始数据阶段：采样 → 训练 tokenizer → 流式构建数据。')
        return 'sample'

    def phase_compile_check(self):
        """torch.compile 曾因环境问题被禁用时，重新做冒烟测试；通过则用 compile 重新基准并重选模型。

        resume_phase：人工修复环境后插入本阶段时，检查完回到原来的阶段（例如数据阶段中途）。
        """
        resume = self.board.get('resume_phase') or 'plan'
        self.board.set('resume_phase', None)
        if not self.config['train']['compile'] or not self.board.get('compile_disabled'):
            return resume
        import torch
        from .smoke import pipeline_smoke
        try:
            smoke = pipeline_smoke(self.config, torch.device('cuda'), True)
        except Exception as exc:  # noqa: BLE001
            self.decide('trainer', 'keep_eager', f'torch.compile 仍不可用（{type(exc).__name__}: {str(exc)[:200]}），继续使用 eager 模式')
            return resume
        self.board.set('compile_disabled', False)
        self.decide('architect', 'reenable_compile', f'torch {torch.__version__} 下 compile 冒烟测试通过（{smoke["seconds"]} 秒），'
                    '用 compile 重新做吞吐基准并重选模型', smoke=smoke)
        old = self.board.get('model_choice') or {}
        pick = self.run_benchmark(True)
        self.decide('architect', 'rechoose_model', f'{pick["rationale"]}（eager 模式下为 {old.get("name")} {old.get("tok_s")} tok/s）')
        return resume

    def phase_sample(self):
        self.bus.post(NAME, 'data', 'directive', {'task': 'sample_for_tokenizer', 'weights': self.weights()})
        report = self.data.sample(self.weights())
        self.board.set('tokenizer_corpus', report)
        self.decide('data', 'train_tokenizer', f'tokenizer 训练语料 {report["total_chars"] / 1e6:.1f}M 字（按配比从每个来源/子目录抽取），'
                    f'用 tokenizer_training 工程从头训练 {self.config["data"]["tokenizer_vocab"]} 词表')
        return 'tokenizer'

    def phase_tokenizer(self):
        self.bus.post(NAME, 'data', 'directive', {'task': 'train_tokenizer'})
        report = self.data.train_tokenizer(now() + 4 * 3600)
        self.board.set('tokenizer', report)
        return 'fast_tokenizer'

    def phase_fast_tokenizer(self):
        self.bus.post(NAME, 'data', 'directive', {'task': 'convert_and_verify_fast_tokenizer'})
        check = self.data.fast_tokenizer()
        self.board.set('fast_tokenizer', check)
        if check.get('use_fast'):
            self.decide('data', 'use_fast_tokenizer', f'HF 快速编码器与原 BBPE 在 {check["checked"]} 条文本上逐 token 比对，'
                        f'不一致 {check["mismatches"]} 条，判定等价，用于全量编码')
        else:
            self.decide('data', 'use_python_tokenizer', f'快速编码器未通过验证（{check.get("error") or check.get("examples")}），'
                        '改用纯 Python BBPE 编码（更慢但完全一致）')
        return 'build_data'

    def phase_build_data(self):
        dcfg = self.config['data']
        targets = self.board.get('data_targets')
        if not targets:
            free = free_gb(RUN_DIR) - dcfg['disk_margin_gb'] - 5.0  # 5GB 留给 best/final 模型和评估产物
            budget = int(min(dcfg['unique_tokens_target'], free * 1e9 / 2 * 0.95))
            if budget < 2e9:
                raise RuntimeError(f'数据盘可用空间不足（{free:.1f}GB 可用于 token），请扩容或清理 /root/autodl-tmp')
            targets = {name: int(budget * w) for name, w in self.weights().items()}
            self.board.set('data_targets', targets)
            self.decide('data', 'set_data_targets', f'数据盘可用 {free + dcfg["disk_margin_gb"] + 5:.1f}GB，'
                        f'唯一训练 token 目标定为 {budget / 1e9:.2f}B（uint16 每 token 2 字节，留 {dcfg["disk_margin_gb"]}GB 余量），按配比分到各来源',
                        targets_B={k: round(v / 1e9, 2) for k, v in targets.items()})
        self.bus.post(NAME, 'data', 'directive', {'task': 'build', 'workers': self.data.workers})
        manifest, short, disk_full = self.data.build(targets)
        totals = {k: v['train_tokens'] for k, v in manifest['sources'].items()}
        self.board.update(unique_tokens=totals, unique_tokens_total=sum(totals.values()),
                          data_stats={k: v['stats'] for k, v in manifest['sources'].items()})
        if short:
            self.decide('data', 'accept_short_sources', f'以下来源未达目标（文件耗尽或多次失败）：'
                        f'{ {k: f"{a / 1e9:.2f}/{b / 1e9:.2f}B" for k, (a, b) in short.items()} }，按实际数量训练，采样权重不变')
        shutil.rmtree(SCRATCH_DIR / 'downloads', ignore_errors=True)
        self.decide('architect', 'plan_training', f'数据完成：{len(totals)} 个来源共 {sum(totals.values()) / 1e9:.2f}B 唯一训练 token'
                    + ('（数据盘已达安全余量，提前停止）' if disk_full else ''))
        return 'compile_check'

    def phase_plan(self):
        tcfg = self.config['train']
        manifest = read_json(RUN_DIR / 'data' / 'tokens' / 'manifest.json')
        pick = self.board.get('model_choice')
        T = tcfg['seq_len']
        micro = pick['micro_batch']
        accum = max(1, round(tcfg['tokens_per_step'] / (micro * T)))
        tokens_per_step = micro * T * accum
        affordable = pick['tok_s'] * tcfg['max_train_hours'] * 3600 / 1.06  # 约 6% 时间用于评估与检查点
        planned = min(tcfg['target_tokens'], affordable)
        unique = self.board.get('unique_tokens')
        epoch_cap = 4.0 * sum(unique.values())  # 整体重复超过 4 遍收益很低，不如少训
        capped_by_data = planned > epoch_cap
        planned = min(planned, epoch_cap)
        total_steps = int(planned / tokens_per_step)
        weights = {k: w for k, w in self.weights().items() if k in unique}
        s = sum(weights.values())
        weights = {k: w / s for k, w in weights.items()}
        projected = {k: round(total_steps * tokens_per_step * w / unique[k], 2) for k, w in weights.items()}
        plan = {'name': pick['name'], 'model': pick['model'], 'params': pick['params'], 'precision': 'bf16',
                'compile': bool(tcfg['compile'] and not self.board.get('compile_disabled')),
                'micro_batch': micro, 'accum': accum, 'seq_len': T, 'tokens_per_step': tokens_per_step,
                'total_steps': total_steps, 'warmup_steps': tcfg['warmup_steps'], 'decay_fraction': tcfg['decay_fraction'],
                'peak_lr': tcfg['peak_lr'], 'weights': weights, 'eval_weights': dict(weights), 'initial_weights': dict(weights),
                'planned_tokens': total_steps * tokens_per_step, 'expected_tok_s': pick['tok_s'],
                'projected_epochs': projected, 'created': stamp(), 'train_started': None}
        self.board.set('plan', plan)
        why = (f'{pick["name"]}（{pick["params"] / 1e6:.0f}M 参数），每步 {tokens_per_step:,} token（micro {micro} × 序列 {T} × 累积 {accum}），'
               f'共 {total_steps:,} 步 = {plan["planned_tokens"] / 1e9:.1f}B token，按基准吞吐约 {plan["planned_tokens"] / pick["tok_s"] / 86400:.1f} 天；'
               f'WSD 学习率：峰值 {tcfg["peak_lr"]}，warmup {tcfg["warmup_steps"]} 步，最后 {tcfg["decay_fraction"]:.0%} 衰减；'
               f'各来源预计重复 {min(projected.values()):.1f}–{max(projected.values()):.1f} 遍。')
        if planned < tcfg['target_tokens']:
            why += f' 注意：按实测吞吐 {tcfg["max_train_hours"] / 24:.0f} 天上限内只能训 {planned / 1e9:.1f}B，低于目标 25B。'
        if capped_by_data:
            why += f' 注意：唯一数据只有 {sum(unique.values()) / 1e9:.1f}B，按最多 4 遍把计划限制在 {planned / 1e9:.1f}B token。'
        self.decide('trainer', 'approve_plan', why)
        return 'train'

    # ---------------- 训练 ----------------
    def setup_training(self):
        from .eval_agent import EvalAgent
        from .trainer_agent import TrainerAgent
        manifest = read_json(RUN_DIR / 'data' / 'tokens' / 'manifest.json')
        tok_json = read_json(RUN_DIR / manifest['tokenizer']['path'])
        plan = self.board.get('plan')
        self.trainer = TrainerAgent(self.config, self.bus, manifest, plan, self.config['device'],
                                    {'type': 'scratch_bbpe', 'model': tok_json})
        self.trainer.build()
        if self.board.get('phase') == 'train' and self.trainer.step == 0 and self.board.get('eval_history'):
            raise FatalError(f'找不到训练检查点 {self.trainer.ckpt_dir / "latest.pt"}，但已有评估记录：拒绝从第 0 步重新训练，需要人工处理')
        self.evaluator = EvalAgent(self.config, self.bus, manifest, load_bbpe(RUN_DIR / manifest['tokenizer']['path']),
                                   self.config['device'])

    def phase_train(self):
        self.setup_training()
        plan = self.trainer.plan
        if not plan.get('train_started'):
            plan['train_started'] = now()
            self.board.set('plan', plan)
        # 时间上限按实际训练耗时计算（停机期间不计入）：按墙钟算的话，机器停过几天后一恢复就会被判定超时、直接结束训练
        budget = self.config['train']['max_train_hours'] * 3600 - self.trainer.train_seconds * 1.04
        deadline = now() + max(0.0, budget)
        while True:
            self.check_control()
            seconds = min(self.config['train']['segment_minutes'] * 60, max(60, deadline - now()))
            self.bus.post(NAME, 'trainer', 'directive', {'task': 'train_segment', 'minutes': round(seconds / 60)})
            report = self.trainer.run_segment(seconds, deadline)
            self.board.update(last_segment=report, plan=self.trainer.plan)
            if report['stop_reason'] in ('stop', 'restart'):
                self.check_control()
            if report['stop_reason'] == 'nonfinite':
                self.rollback('连续 3 步出现非有限 loss/梯度，回滚到上一个检查点并把学习率降为 0.7 倍')
                continue
            self.bus.post(NAME, 'evaluator', 'directive', {'task': 'evaluate', 'step': report['step']})
            record = self.evaluator.evaluate(self.trainer)
            self.review(report, record, deadline)
            self.board.update(phase_failures=0)
            self.write_report()
            if report['stop_reason'] in ('plan_complete', 'deadline') or now() >= deadline:
                self.decide('all', 'finish_training', f'训练结束（{report["stop_reason"]}），进入最终评估',
                            step=self.trainer.step, tokens_B=round(self.trainer.tokens_seen / 1e9, 2))
                return 'final_eval'

    def review(self, seg, rec, deadline):
        plan = self.trainer.plan
        history = self.board.get('eval_history', [])
        history.append({k: rec[k] for k in ('time', 'step', 'tokens_seen', 'val_loss', 'val_by_source', 'cloze_acc',
                                             'sample_rep4', 'greedy_rep4')})
        self.board.set('eval_history', history[-1000:])
        best = self.board.get('best_val')
        # 1. 稳定性
        if best and seg['step'] > plan['warmup_steps'] * 2 and rec['val_loss'] > best['val_loss'] * self.DIVERGE_RATIO:
            self.rollback(f'验证 loss {rec["val_loss"]} 比最佳 {best["val_loss"]} 高 {self.DIVERGE_RATIO - 1:.0%} 以上，判定为发散，回滚并降低学习率')
            return
        self.board.set('consecutive_rollbacks', 0)
        if best is None or rec['val_loss'] < best['val_loss']:
            self.trainer.save_weights(RUN_DIR / 'model' / 'best.pt', val_loss=rec['val_loss'])
            self.board.set('best_val', {'val_loss': rec['val_loss'], 'step': rec['step']})
        # 2. 计划（只在衰减开始前调整）
        decay_start = int(plan['total_steps'] * (1 - plan['decay_fraction']))
        if seg['stop_reason'] == 'segment_time' and seg['segment_steps'] > 0 and self.trainer.step < decay_start:
            rates = (self.board.get('segment_rates') or []) + [seg['segment_tok_s']]
            self.board.set('segment_rates', rates[-8:])
            recent = sorted(rates[-4:])
            tok_s = (recent[(len(recent) - 1) // 2] + recent[len(recent) // 2]) / 2
            possible = self.trainer.step + int(tok_s / plan['tokens_per_step'] * (deadline - now()) / 1.06)
            cap = int(self.config['train']['target_tokens'] / plan['tokens_per_step'])
            total, new_total = plan['total_steps'], None
            if possible < total * 0.95:
                new_total = max(self.trainer.step + 100, int(max(possible, total * 0.92)))
                why = f'近 4 段吞吐中位数 {tok_s:.0f} tok/s，按当前计划会超出时间上限，缩短计划（单次最多 8%）'
            elif possible > total * 1.08 and total < cap:
                new_total = int(min(possible * 0.97, total * 1.08, cap))
                why = f'近 4 段吞吐中位数 {tok_s:.0f} tok/s，高于计划，延长计划但不超过 25B token（单次最多 8%）'
            if new_total and new_total != total:
                plan['total_steps'] = new_total
                plan['planned_tokens'] = new_total * plan['tokens_per_step']
                self.decide('trainer', 'replan_steps', why, old_total=total, new_total=new_total)
        # 3. 重复  4. 遗忘
        weights = dict(self.trainer.sampler.weights)
        changed = False
        frac = self.trainer.step / max(1, plan['total_steps'])
        if frac > 0.05:
            for name, ep in seg['epochs'].items():
                projected = ep / frac
                if projected > 4.0 and name in weights:
                    new_w = weights[name] * 4.0 / projected
                    self.decide('trainer', 'reduce_weight', f'{name} 预计重复 {projected:.1f} 遍（>4 遍收益明显下降），降低采样权重',
                                source=name, old=round(weights[name], 4), new=round(new_w, 4))
                    weights[name] = new_w
                    changed = True
        analysis = self.get_analyst().analyze(seg, plan)
        bumps = self.board.get('weight_bumps') or {}
        if analysis.get('enough_data') and seg['step'] > plan['warmup_steps'] * 2 and analysis['overall']['status'] != 'regressing':
            for name in analysis['regressing_sources']:
                if name not in weights or bumps.get(name, -1) >= analysis['window_start_step']:
                    continue
                if weights[name] < 1.5 * plan['initial_weights'].get(name, 0):
                    s = analysis['sources'][name]
                    new_w = min(weights[name] * 1.2, 1.5 * plan['initial_weights'][name])
                    self.decide('trainer', 'increase_weight',
                                f'结果分析：{name} 验证 loss 最近 {self.get_analyst().cfg["window"]} 次评估显著上升（拟合 +{s["window_change"]}，'
                                f't={s["t"]}，单次噪声 σ≈{s["sigma"]}）而总体没有变差，可能在被遗忘，提高采样权重',
                                source=name, old=round(weights[name], 4), new=round(new_w, 4))
                    weights[name] = new_w
                    bumps[name] = seg['step']
                    changed = True
            self.board.set('weight_bumps', bumps)
        if changed:
            s = sum(weights.values())
            self.trainer.apply_weights({k: w / s for k, w in weights.items()})
        self.board.update(plan=self.trainer.plan)

    def phase_final_eval(self):
        if self.trainer is None:
            self.setup_training()
        record = self.evaluator.evaluate(self.trainer, final=True)
        self.trainer.save_weights(RUN_DIR / 'model' / 'final.pt', val_loss=record['val_loss'], test_loss=record.get('test_loss'))
        best = self.board.get('best_val')
        if best is None or record['val_loss'] < best['val_loss']:
            self.trainer.save_weights(RUN_DIR / 'model' / 'best.pt', val_loss=record['val_loss'])
            self.board.set('best_val', {'val_loss': record['val_loss'], 'step': record['step']})
        self.board.set('final', {k: v for k, v in record.items()})
        self.decide('all', 'complete', f'训练完成：{self.trainer.step} 步、{self.trainer.tokens_seen / 1e9:.2f}B token；'
                    f'验证 loss {record["val_loss"]}，测试 loss {record.get("test_loss")}，选择题准确率 {record["cloze_acc"]:.0%}。'
                    f'模型保存在 {RUN_DIR / "model"}（数据盘）。'
                    + ('接下来做参数空间编辑实验、pass@k 能力探测、token 频次探测和 profiler 计时，完成后关机' if self.post_phase('final_eval') != 'done' else '守护脚本将在 2 分钟后关机'))
        return self.post_phase('final_eval')

    def post_phase(self, current):
        """训练结束后的附加阶段顺序：final_eval → weight_edit → passk_probe → token_probe → profile → done（未启用的跳过）。"""
        order = ['final_eval', 'weight_edit', 'passk_probe', 'token_probe', 'profile', 'analysis', 'done']
        keys = {'weight_edit': 'weight_edit', 'passk_probe': 'passk', 'token_probe': 'token_probe', 'profile': 'profile',
                'analysis': 'analysis'}
        for name in order[order.index(current) + 1:]:
            if name == 'done' or self.config.get(keys[name], {}).get('enabled'):
                return name
        return 'done'

    def phase_weight_edit(self):
        """训练完成后：按 Weight-Space Programming 的方法只改少量 FFN 输出列，检查各指标是否提升。
        实验失败不影响已保存的模型，记录原因后照常结束（随后关机）。"""
        import torch
        try:
            if self.trainer is None:
                self.setup_training()
            final = RUN_DIR / 'model' / 'final.pt'
            if final.exists():  # 重启后 latest.pt 可能早于最终状态，以 final.pt 为准
                self.trainer.model.load_state_dict(torch.load(final, map_location='cpu', weights_only=True)['model'])
            from .weight_edit import run_weight_edit
            manifest = read_json(RUN_DIR / 'data' / 'tokens' / 'manifest.json')
            self.bus.post(NAME, 'weight_editor', 'directive', {'task': 'weight_space_edit',
                                                               'budget': self.config['weight_edit']['budget']})
            res = run_weight_edit(self.config, self.bus, manifest, self.evaluator, self.trainer,
                                  torch.device(self.config['device']))
            b, a = res['heldout_before'], res['heldout_after']
            summary = {'verdict': res['verdict'], 'weights_changed': res['weights_changed'], 'edits': res['edits'],
                       'before': {k: b[k] for k in ('val_loss', 'test_loss', 'cloze_acc', 'probe_greedy_rep4')}
                       | {'loop_rate': b['heldout_greedy']['loop_rate']},
                       'after': {k: a[k] for k in ('val_loss', 'test_loss', 'cloze_acc', 'probe_greedy_rep4')}
                       | {'loop_rate': a['heldout_greedy']['loop_rate']}}
            self.board.set('weight_edit', summary)
            self.decide('all', 'weight_edit_done',
                        f'参数空间编辑实验完成（{res["verdict"]}）：改动 {res["weights_changed"]:,} 个权重；测试集续写贪心复读率 '
                        f'{summary["before"]["loop_rate"]}→{summary["after"]["loop_rate"]}，测试 loss '
                        f'{b["test_loss"]}→{a["test_loss"]}，选择题 {b["cloze_acc"]:.0%}→{a["cloze_acc"]:.0%}。'
                        f'报告：{RUN_DIR / "weight_edit" / "REPORT.md"}')
        except Exception as exc:  # noqa: BLE001
            self.board.set('weight_edit', {'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()[-3000:]})
            self.decide('all', 'weight_edit_failed', f'参数空间编辑实验出错（{type(exc).__name__}: {str(exc)[:300]}），'
                        '不影响已保存的模型')
        return self.post_phase('weight_edit')

    def phase_passk_probe(self):
        """pass@k 能力探测：原模型与编辑后模型在可验证任务上的 pass@1/4/8/16 与 GRPO 有效分组比例。
        出错只记录原因，不影响已保存的模型，随后照常结束（关机）。"""
        import torch
        try:
            if self.trainer is None:
                self.setup_training()
            from .passk_probe import rl_readiness, run_passk
            self.bus.post(NAME, 'passk_probe', 'directive', {'task': 'passk_probe', 'n': self.config['passk']['n']})
            res = run_passk(self.config, self.bus, self.trainer, self.evaluator.tok, torch.device(self.config['device']))
            summary = {m: [{'task': r['task'], 'greedy': r['greedy_acc'], 'pass@1': r['pass_at_k'].get('1'),
                            'pass@16': r['pass_at_k'].get('16'), 'mixed_groups': r['mixed_groups'], 'verdict': rl_readiness(r)}
                           for r in rows] for m, rows in res['models'].items()}
            self.board.set('passk', {'summary': summary, 'errors': res['errors']})
            base = next(iter(summary.values()))
            ready = [r['task'] for r in base if r['verdict'] == '适合直接 RLVR']
            self.decide('all', 'passk_done', f'pass@k 能力探测完成（{res["minutes"]} 分钟）：适合直接 RLVR 的任务 '
                        f'{ready or "无"}；详见 {RUN_DIR / "passk" / "REPORT.md"}'
                        + ('；守护脚本将在 2 分钟后关机' if self.post_phase('passk_probe') == 'done' else ''))
        except Exception as exc:  # noqa: BLE001
            self.board.set('passk', {'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()[-3000:]})
            self.decide('all', 'passk_failed', f'pass@k 探测出错（{type(exc).__name__}: {str(exc)[:300]}），'
                        '不影响已保存的模型')
        return self.post_phase('passk_probe')

    def phase_token_probe(self):
        """token 频次 / 欠训练 token / GSM8K 13-gram 污染。出错只记录原因，随后照常结束（关机）。"""
        import torch
        try:
            if self.trainer is None:
                self.setup_training()
            from .token_freq_probe import run_token_probe, verdict
            self.bus.post(NAME, 'token_probe', 'directive', {'task': 'token_probe'})
            res = run_token_probe(self.config, self.bus, self.trainer, self.evaluator.tok, torch.device(self.config['device']))
            c = (res.get('contamination') or {}).get('all_test') or {}
            summary = {'verdict': verdict(res), 'freq': {k: res['freq'][k] for k in ('zero', 'lt100', 'lt1000', 'tokens_to_cover')},
                       'auc_norm': res['embedding']['auc_rare_low_norm'], 'auc_indicator': res['embedding']['auc_rare_high_indicator'],
                       'repeat': {g: v['ok_rate'] for g, v in res['repeat'].items()},
                       'gsm8k_any_overlap': c.get('any_field_any_overlap'),
                       'gsm8k_strong_question': (c.get('question') or {}).get('strong_overlap'),
                       'gsm8k_strong_answer': (c.get('answer_clean') or {}).get('strong_overlap')}
            self.board.set('token_probe', summary)
            self.decide('all', 'token_probe_done', f'token 探测完成（{res["minutes"]} 分钟）：{summary["verdict"]}；GSM8K 13-gram 强重叠 '
                        f'题目 {summary["gsm8k_strong_question"]}、答案 {summary["gsm8k_strong_answer"]}；'
                        f'详见 {RUN_DIR / "token_probe" / "REPORT.md"}')
        except Exception as exc:  # noqa: BLE001
            self.board.set('token_probe', {'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()[-3000:]})
            self.decide('all', 'token_probe_failed', f'token 探测出错（{type(exc).__name__}: {str(exc)[:300]}），'
                        '不影响已保存的模型')
        return self.post_phase('token_probe')

    def phase_profile(self):
        """torch.profiler 抓一个真实训练步 + MFU。会做几次优化器更新（模型文件早已保存），出错只记录原因，随后结束（关机）。"""
        import torch
        try:
            if self.trainer is None:
                self.setup_training()
            from .profile_probe import run_profile
            self.bus.post(NAME, 'profiler', 'directive', {'task': 'profile'})
            r = run_profile(self.config, self.bus, self.trainer, torch.device(self.config['device']))
            summary = {k: r[k] for k in ('step_seconds', 'tok_s', 'mfu', 'achieved_tflops', 'gpu_busy', 'compile_speedup',
                                         'eager_note', 'peak_mem_gb')}
            summary['by_category'] = {k: v['share'] for k, v in r['by_category'].items()}
            self.board.set('profile', summary)
            self.decide('all', 'profile_done', f'profiler 完成：每步 {r["step_seconds"]} s，{r["tok_s"]:,} tok/s，MFU {r["mfu"]:.1%}，'
                        f'GPU 忙碌 {r["gpu_busy"]:.0%}；详见 {RUN_DIR / "profile" / "REPORT.md"}')
        except Exception as exc:  # noqa: BLE001
            self.board.set('profile', {'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()[-3000:]})
            self.decide('all', 'profile_failed', f'profiler 出错（{type(exc).__name__}: {str(exc)[:300]}），'
                        '不影响已保存的模型')
        return self.post_phase('profile')

    def phase_analysis(self):
        """结果分析智能体写最终分析报告（曲线、分来源结果、各项探测结论、下一步建议）。出错只记录原因，随后结束（关机）。"""
        try:
            summary = self.get_analyst().final_report(self.board.data)
            self.board.set('analysis_final', summary)
            self.decide('all', 'analysis_done', f'结果分析完成：最终验证 loss {summary["final_val"]}，测试 loss {summary["test_loss"]}；'
                        f'下一步建议 {len(summary["next_steps"])} 条，详见 {RUN_DIR / "analysis" / "FINAL_ANALYSIS.md"}；守护脚本将在 2 分钟后关机')
        except Exception as exc:  # noqa: BLE001
            self.board.set('analysis_final', {'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()[-3000:]})
            self.decide('all', 'analysis_failed', f'结果分析出错（{type(exc).__name__}: {str(exc)[:300]}），不影响已保存的模型；'
                        '守护脚本将在 2 分钟后关机')
        return 'done'

    # ---------------- 报告 ----------------
    def write_report(self):
        try:
            write_text(RUN_DIR / 'REPORT.md', render_report(self.board.data, self.bus))
        except Exception:
            traceback.print_exc()


def render_report(state, bus):
    L = ['# A800 多智能体预训练报告（0.5B 中英代码数学）', '',
         f'- 更新：{state.get("updated", stamp())}　开始：{state.get("started_text")}　当前阶段：**{state.get("phase")}**',
         f'- 各阶段耗时（小时）：{state.get("phase_hours", {})}', '']
    st = state.get('selftest')
    if st:
        s = st['self_test']
        L += [f'- 自检：{s.get("gpu")} {s.get("gpu_mem_gb")}GB，因果性差 {s["causal_max_diff"]:.1e}，KV Cache 差 {s["kv_cache_max_diff"]:.1e}，'
              f'冒烟测试 {"通过" if st["smoke"]["passed"] else "失败"}（compile={st["smoke"]["compile"]}）', '']
    if state.get('benchmark'):
        L += ['## 结构基准（结构与算法智能体）', '', '| 候选 | 参数 | 层数 | micro | tok/s | MFU | 显存 GB |', '|---|---|---|---|---|---|---|']
        for r in state['benchmark']:
            L.append(f'| {r["name"]} | {r["params"] / 1e6:.0f}M | {r["layers"]} | {r.get("micro_batch", "-")} | {r.get("tok_s", "失败")} | '
                     f'{r.get("mfu", "-")} | {r.get("peak_mem_gb", "-")} |')
        L.append('')
    if state.get('tokenizer'):
        L += ['## Tokenizer', '', f'- {state["tokenizer"]}', f'- 快速编码器：{state.get("fast_tokenizer")}', '']
    if state.get('unique_tokens'):
        L += ['## 数据（数据清洗智能体）', '', '| 来源 | 唯一训练 token | 文档 | 接收率 |', '|---|---|---|---|']
        for k, v in state['unique_tokens'].items():
            st2 = state.get('data_stats', {}).get(k, {})
            docs = sum(st2.get('docs_' + s, 0) for s in ('train', 'val', 'test'))
            L.append(f'| {k} | {v / 1e9:.2f}B | {docs:,} | {docs / max(1, st2.get("rows", 1)):.0%} |')
        L.append('')
    plan = state.get('plan')
    if plan:
        seg = state.get('last_segment') or {}
        L += ['## 训练计划与进度', '',
              f'- {plan["name"]}，{plan["params"] / 1e6:.0f}M 参数，{plan["model"]}，compile={plan["compile"]}',
              f'- 每步 {plan["tokens_per_step"]:,} token，计划 {plan["total_steps"]:,} 步（{plan["planned_tokens"] / 1e9:.1f}B token），峰值 lr {plan["peak_lr"]}',
              f'- 进度：{seg.get("step", 0):,}/{plan["total_steps"]:,} 步（{seg.get("step", 0) / max(1, plan["total_steps"]):.1%}），'
              f'{seg.get("tokens_seen", 0) / 1e9:.2f}B token；最近吞吐 {seg.get("segment_tok_s")} tok/s，训练 loss {seg.get("train_loss")}，'
              f'梯度范数中位数 {seg.get("grad_norm_median")}，显存峰值 {seg.get("peak_mem_gb")}GB', '']
    hist = state.get('eval_history') or []
    if hist:
        L += ['## 评估（测试评估智能体）', '', '| 时间 | 步数 | token | 验证 loss | 选择题 | 采样 rep4 | 贪心 rep4 |', '|---|---|---|---|---|---|---|']
        for h in hist[-20:]:
            L.append(f'| {h["time"][5:16]} | {h["step"]:,} | {h["tokens_seen"] / 1e9:.2f}B | {h["val_loss"]} | {h["cloze_acc"]:.0%} | '
                     f'{h["sample_rep4"]} | {h["greedy_rep4"]} |')
        L += ['', '分来源验证 loss（最新）：' + '，'.join(f'{k} {v}' for k, v in sorted(hist[-1]['val_by_source'].items())), '']
    samples = read_json(RUN_DIR / 'samples' / 'latest.json')
    if samples:
        L += ['## 最新生成样例（top-p 0.9，温度 0.8）', '']
        for r in [r for r in samples['generation'] if r['mode'] == 'sample'][:6]:
            L.append(f'- **{r["prompt"].strip()}**{r["completion"][:200].replace(chr(10), " ")}')
        L.append('')
    if state.get('final'):
        f = state['final']
        L += ['## 最终结果', '', f'- 验证 loss {f["val_loss"]}，测试 loss {f.get("test_loss")}，选择题 {f["cloze_acc"]:.0%}（{f.get("cloze_by_cat")}）',
              f'- 模型：{RUN_DIR / "model" / "best.pt"}、{RUN_DIR / "model" / "final.pt"}', '']
    an = read_json(RUN_DIR / 'analysis' / 'latest.json')
    if an and an.get('enough_data'):
        L += ['## 结果分析（结果分析智能体，最新一次）', ''] + [f'- {f}' for f in an.get('findings', [])] + ['']
    af = state.get('analysis_final')
    if af:
        L += ['## 最终分析与下一步建议', '']
        L += [f'- 出错：{af["error"]}'] if 'error' in af else [f'{i}. {r}' for i, r in enumerate(af.get('next_steps', []), 1)] + ['', '详见 analysis/FINAL_ANALYSIS.md']
        L.append('')
    L += ['## 智能体决策记录（最近 30 条）', '']
    for e in bus.tail(30, kinds=('decision',)):
        c = e['content']
        L.append(f'- {e["time"]} **{c.get("decision")}** → {e["to"]}：{c.get("why")}')
    we = state.get('weight_edit')
    if we:
        L += ['', '## 参数空间编辑实验（Weight-Space Programming）', '']
        if 'error' in we:
            L += [f'- 出错：{we["error"]}']
        else:
            L += [f'- 结论：**{we["verdict"]}**，改动 {we["weights_changed"]:,} 个权重，详见 weight_edit/REPORT.md', '',
                  '| 指标 | 编辑前 | 编辑后 |', '|---|---|---|']
            L += [f'| {k} | {we["before"][k]} | {we["after"][k]} |' for k in we['before']]
    pk = state.get('passk')
    if pk:
        L += ['', '## pass@k 能力探测（RLVR 适合度）', '']
        if 'error' in pk:
            L += [f'- 出错：{pk["error"]}']
        else:
            for model_name, rows in pk['summary'].items():
                L += [f'**{model_name}**', '', '| 任务 | 贪心 | pass@1 | pass@16 | 有效分组 | 结论 |', '|---|---|---|---|---|---|']
                L += [f'| {r["task"]} | {r["greedy"]:.0%} | {r["pass@1"]:.0%} | {r["pass@16"]:.0%} | {r["mixed_groups"]:.0%} | {r["verdict"]} |'
                      for r in rows]
                L.append('')
    tp = state.get('token_probe')
    if tp:
        L += ['', '## token 频次 / 欠训练 token / GSM8K 污染', '']
        if 'error' in tp:
            L += [f'- 出错：{tp["error"]}']
        else:
            L += [f'- {tp["verdict"]}', f'- 出现 0 次 / <100 次 / <1000 次的 token：{tp["freq"]["zero"]} / {tp["freq"]["lt100"]} / '
                  f'{tp["freq"]["lt1000"]}；复述成功率 {tp["repeat"]}',
                  f'- GSM8K test 13-gram：任一重叠 {tp["gsm8k_any_overlap"]}，题目强重叠 {tp["gsm8k_strong_question"]}，'
                  f'答案强重叠 {tp["gsm8k_strong_answer"]}；详见 token_probe/REPORT.md']
    pf = state.get('profile')
    if pf:
        L += ['', '## profiler 与 MFU', '']
        if 'error' in pf:
            L += [f'- 出错：{pf["error"]}']
        else:
            L += [f'- 每步 {pf["step_seconds"]} s，{pf["tok_s"]:,} tok/s，MFU {pf["mfu"]:.1%}，GPU 忙碌 {pf["gpu_busy"]:.0%}，'
                  f'torch.compile 加速 ×{pf["compile_speedup"]}；GPU 时间占比 {pf["by_category"]}；详见 profile/REPORT.md']
    return '\n'.join(L) + '\n'

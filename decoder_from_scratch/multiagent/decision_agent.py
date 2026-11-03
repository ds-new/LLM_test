"""决策智能体：编排其他四个智能体，审核它们的报告，按明确策略做出决策并记录理由。

阶段：selftest → data_acquire → data_merge → tokenizer → encode → model_select → train → final_eval → done

训练阶段每段结束后的决策策略（全部写入 events.jsonl，类型 decision）：
  1. 稳定性：连续非有限 loss 或验证 loss 比最佳值高 5% 以上 → 回滚到上一个检查点并把学习率乘 0.7；
  2. 进度：按实测吞吐预测能否在截止前跑完计划步数，必要时缩短（任意时刻）或延长（进度 < 60%）计划，
     余弦学习率随之重新对齐终点；
  3. 数据重复：预测某领域到训练结束的 epoch 数超过上限 → 让数据智能体在独立进程补数据；
     无法补充时下调该领域权重；
  4. 遗忘：某领域验证 loss 连续两次上升而整体下降 → 该领域权重 ×1.2（不超过初始值 1.5 倍）；
  5. 预算：到达截止或 STOP 文件 → 保存检查点、最终评估、写报告。
"""

import math
import multiprocessing
import os
import time
import traceback

from .common import (RUN_DIR, Blackboard, Bus, RestartRequested, StopRequested, control_signal, now,
                     read_json, stamp, write_json, write_text)
from .data_agent import DataAgent, topup_job, load_bbpe

NAME = 'decision'
BENCHMARK_VERSION = 2  # v1 基准受内存抖动影响已作废；旧计划在未开始训练时自动重新选型


class DecisionAgent:
    def __init__(self, config, bus=None, board=None):
        self.config = config
        self.bus = bus or Bus()
        self.board = board or Blackboard()
        if not self.board.get('started'):
            start = now()
            self.board.update(started=start, started_text=stamp(start),
                              deadline=start + config['budget_hours'] * 3600,
                              deadline_text=stamp(start + config['budget_hours'] * 3600),
                              phase='selftest', active_domains=list(config['data']['weights']),
                              decisions=0)
        self.deadline = self.board.get('deadline')
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

    def remaining(self):
        return self.deadline - now()

    def device(self):
        import torch
        name = self.config['device']
        if name == 'mps' and not torch.backends.mps.is_available():
            self.decide('trainer', 'fallback_cpu', 'MPS 不可用，退回 CPU（会非常慢）')
            return 'cpu'
        return name

    def active(self):
        return list(self.board.get('active_domains'))

    # ---------------- 主循环 ----------------
    def run(self):
        while True:
            self.check_control()
            phase = self.board.get('phase')
            if phase == 'done':
                self.write_report()
                return
            plan = self.board.get('plan') or {}
            if (phase == 'train' and plan.get('benchmark_version') != BENCHMARK_VERSION
                    and not (RUN_DIR / 'checkpoints' / 'latest.pt').exists()):
                self.decide('architect', 'redo_model_select', '训练计划基于已作废的 v1 基准（内存抖动导致吞吐失真），训练尚未开始，重新基准测试与选型')
                self.board.update(phase='model_select', plan=None)
                continue
            if self.remaining() <= 0 and phase not in ('train', 'final_eval'):
                self.decide('all', 'budget_exhausted', f'截止时间 {self.board.get("deadline_text")} 已到，阶段 {phase} 未完成')
                self.board.set('phase', 'done')
                continue
            self.bus.post(NAME, 'all', 'phase', {'phase': phase, 'remaining_hours': round(self.remaining() / 3600, 2)})
            next_phase = getattr(self, 'phase_' + phase)()
            self.board.update(phase=next_phase, phase_failures=0)
            self.write_report()

    # ---------------- 阶段实现 ----------------
    def phase_selftest(self):
        from .architect_agent import ArchitectAgent
        import torch
        self.architect = ArchitectAgent(self.config, self.bus)
        results = {}
        for dev in ('cpu', self.device()):
            if dev in results:
                continue
            self.bus.post(NAME, 'architect', 'directive', {'task': 'self_test', 'device': dev})
            results[dev] = self.architect.self_test(torch.device(dev))
            self.bus.post('architect', NAME, 'report', {'action': 'self_test', **results[dev]})
        from .smoke import pipeline_smoke
        self.bus.post(NAME, 'trainer', 'directive', {'task': 'pipeline_smoke', 'device': self.device()})
        results['pipeline_smoke'] = pipeline_smoke(self.config, torch.device(self.device()))
        self.bus.post('trainer', NAME, 'report', {'action': 'pipeline_smoke', **results['pipeline_smoke']})
        self.board.set('selftest', results)
        if not all(r['passed'] for r in results.values()):
            raise RuntimeError(f'model self-test failed: {results}')
        self.decide('data', 'start_data_phase', '模型结构自检与训练/恢复/评估冒烟测试通过，开始准备数据')
        return 'data_acquire'

    def phase_data_acquire(self):
        weights = self.config['data']['weights']
        total = self.config['data']['total_chars']
        attempts = self.board.get('acquire_attempts', {})
        data_deadline = min(self.deadline - 40 * 3600, now() + 10 * 3600)  # 数据下载最多占用约 10 小时
        use_local = {}
        pending = None  # None = 全部领域（已有完整回执的会被复用）
        for round_index in range(self.config['data']['download_attempts'] + 2):
            self.check_control()
            self.bus.post(NAME, 'data', 'directive', {'task': 'acquire', 'round': round_index,
                                                      'domains': pending or 'all', 'use_local': list(use_local)})
            receipts = self.data.acquire(max(data_deadline, now() + 1800), domains=pending, use_local=use_local)
            retry = []
            for domain, r in receipts.items():
                chars = r.get('stats', {}).get('chars', 0)
                target = int(total * weights[domain])
                complete = chars >= 0.98 * target or r.get('finished_stream')
                if r.get('error') and not complete:
                    attempts[domain] = attempts.get(domain, 0) + 1
                    self.board.set('acquire_attempts', attempts)
                    if attempts[domain] < self.config['data']['download_attempts']:
                        retry.append(domain)
                        os.replace(RUN_DIR / 'data' / 'raw_clean' / f'{domain}.receipt.json',
                                   RUN_DIR / 'data' / 'raw_clean' / f'{domain}.receipt.failed{attempts[domain]}.json')
                    elif chars < 0.3 * target and domain not in use_local and self.data.legacy_file(domain):
                        use_local[domain] = self.data.legacy_file(domain)
                        retry.append(domain)
                        self.decide('data', 'use_local_fallback', f'{domain} 下载 {attempts[domain]} 次失败（{r["error"]}），改用本地已有语料',
                                    domain=domain)
                        os.replace(RUN_DIR / 'data' / 'raw_clean' / f'{domain}.receipt.json',
                                   RUN_DIR / 'data' / 'raw_clean' / f'{domain}.receipt.failed_final.json')
            if not retry:
                break
            pending = retry
            self.decide('data', 'retry_download', '部分领域下载中断且未达目标，重试', domains=retry)
            time.sleep(30)
        # 审核：数据过少的领域隔离，其余按实际情况接受
        active, summary = [], {}
        for domain in weights:
            receipt = read_json(RUN_DIR / 'data' / 'raw_clean' / f'{domain}.receipt.json') or {}
            chars = receipt.get('stats', {}).get('chars', 0)
            summary[domain] = {'chars': chars, 'target': int(total * weights[domain]), 'error': receipt.get('error')}
            if chars >= self.config['data'].get('min_domain_chars', 2_000_000):
                active.append(domain)
            else:
                self.decide('data', 'quarantine_domain', f'{domain} 只有 {chars} 字符可用（<200 万），不参与训练', domain=domain)
        if len(active) < self.config['data'].get('min_domains', 4):
            raise RuntimeError(f'too few usable domains: {summary}')
        self.board.update(active_domains=active, acquire_summary=summary)
        self.decide('data', 'accept_raw_data', f'{len(active)} 个领域进入去重与划分',
                    total_chars=sum(v['chars'] for d, v in summary.items() if d in active))
        return 'data_merge'

    def phase_data_merge(self):
        self.bus.post(NAME, 'data', 'directive', {'task': 'merge', 'domains': self.active()})
        report = self.data.merge(self.active())
        active = []
        for domain in self.active():
            s = report['domains'].get(domain, {})
            need = self.config['data'].get('min_val_docs', 30)
            if s.get('docs_val', 0) >= need and s.get('docs_test', 0) >= need and s.get('chars_train', 0) >= 1_000_000:
                active.append(domain)
            else:
                self.decide('data', 'quarantine_domain', f'{domain} 去重后验证/测试文档不足，不参与训练', domain=domain, stats=s)
        self.board.update(active_domains=active, merge_summary={d: report['domains'][d] for d in active})
        self.decide('data', 'train_tokenizer', f'去重划分完成，train 共 {report["total_train_chars"] / 1e6:.0f}M 字符；'
                    f'按要求使用 tokenizer_training 工程从头训练 {self.config["data"]["tokenizer_vocab"]} 词表')
        return 'tokenizer'

    def phase_tokenizer(self):
        self.bus.post(NAME, 'data', 'directive', {'task': 'train_tokenizer'})
        report = self.data.train_tokenizer(self.active(), self.deadline)
        self.board.set('tokenizer', report)
        return 'encode'

    def phase_encode(self):
        self.bus.post(NAME, 'data', 'directive', {'task': 'encode'})
        manifest = self.data.encode(self.active())
        unique = {d: e['train'][0]['tokens'] for d, e in manifest['domains'].items()}
        self.board.update(unique_tokens=unique, unique_tokens_total=sum(unique.values()))
        self.decide('architect', 'select_model', f'编码完成，唯一训练 token {sum(unique.values()) / 1e6:.0f}M；请基准测试候选结构')
        return 'model_select'

    def phase_model_select(self):
        from .architect_agent import ArchitectAgent
        import torch
        self.architect = self.architect or ArchitectAgent(self.config, self.bus)
        manifest = read_json(RUN_DIR / 'data' / 'tokens' / 'manifest.json')
        tcfg = self.config['train']
        device = torch.device(self.device())
        train_seconds = self.remaining() - tcfg['safety_hours'] * 3600
        train_seconds *= (1 - tcfg['eval_reserve_fraction'])
        rows = self.board.get('benchmark') if self.board.get('benchmark_version') == BENCHMARK_VERSION else None
        if not rows:
            self.bus.post(NAME, 'architect', 'directive', {'task': 'benchmark', 'seq_len': tcfg['seq_len'],
                                                           'version': BENCHMARK_VERSION})
            rows = self.architect.benchmark(device, manifest['tokenizer']['vocab_size'], tcfg['seq_len'],
                                            train_seconds=train_seconds, target_ratio=tcfg['target_tokens_per_param'])
            self.board.update(benchmark=rows, benchmark_version=BENCHMARK_VERSION)
        unique_total = self.board.get('unique_tokens_total')
        pick = self.architect.propose(rows, train_seconds, unique_total, tcfg['target_tokens_per_param'])
        self.bus.post('architect', NAME, 'proposal', pick)
        # 审核：数据重复上限
        weights = self.normalized_weights()
        unique = self.board.get('unique_tokens')
        projected = {d: round(pick['planned_tokens'] * w / max(1, unique[d]), 2) for d, w in weights.items()}
        over = {d: e for d, e in projected.items() if e > tcfg['max_epochs_per_domain']}
        tokens_per_micro = pick['micro_batch'] * tcfg['seq_len']
        accum = max(1, round(tcfg['tokens_per_step'] / tokens_per_micro))
        tokens_per_step = tokens_per_micro * accum
        total_steps = max(100, int(pick['planned_tokens'] / tokens_per_step))
        plan = {'name': pick['name'], 'model': pick['model'], 'params': pick['params'], 'precision': pick['precision'],
                'micro_batch': pick['micro_batch'], 'accum': accum, 'seq_len': tcfg['seq_len'],
                'tokens_per_step': tokens_per_step, 'total_steps': total_steps,
                'warmup_steps': max(200, int(total_steps * tcfg['warmup_fraction'])),
                'peak_lr': pick['peak_lr'], 'weights': weights, 'eval_weights': dict(weights),
                'initial_weights': dict(weights), 'planned_tokens': total_steps * tokens_per_step,
                'expected_tok_s': pick['tokens_per_second'], 'projected_epochs': projected,
                'created': stamp(), 'rationale': pick['rationale'], 'extra_shards': [],
                'benchmark_version': BENCHMARK_VERSION}
        self.board.set('plan', plan)
        self.decide('trainer', 'approve_plan', pick['rationale'] + (
            f' 预计以下领域重复超过 {tcfg["max_epochs_per_domain"]} 轮，训练中将补数据或降权：{over}' if over else ' 各领域预计重复次数均在上限内。'),
            plan={k: plan[k] for k in ('name', 'params', 'precision', 'micro_batch', 'accum', 'total_steps',
                                        'warmup_steps', 'peak_lr', 'planned_tokens')})
        return 'train'

    def normalized_weights(self):
        weights = self.config['data']['weights']
        active = self.active()
        s = sum(weights[d] for d in active)
        return {d: weights[d] / s for d in active}

    # ---------------- 训练阶段 ----------------
    def setup_training(self):
        from .trainer_agent import TrainerAgent
        from .eval_agent import EvalAgent
        manifest = read_json(RUN_DIR / 'data' / 'tokens' / 'manifest.json')
        tokenizer_json = read_json(RUN_DIR / manifest['tokenizer']['path'])
        tokenizer_dict = {'type': 'scratch_bbpe', 'model': tokenizer_json}
        plan = self.board.get('plan')
        device = self.device()
        self.trainer = TrainerAgent(self.config, self.bus, manifest, plan, device, tokenizer_dict)
        self.trainer.build()
        self.evaluator = EvalAgent(self.config, self.bus, manifest, load_bbpe(RUN_DIR / manifest['tokenizer']['path']), device)

    def phase_train(self):
        self.setup_training()
        tcfg = self.config['train']
        train_deadline = self.deadline - 20 * 60  # 留 20 分钟做最终评估与报告
        while True:
            self.check_control()
            self.collect_topups()
            seconds = min(tcfg['segment_minutes'] * 60, max(60, train_deadline - now()))
            self.bus.post(NAME, 'trainer', 'directive', {'task': 'train_segment', 'minutes': round(seconds / 60, 1)})
            report = self.trainer.run_segment(seconds, train_deadline)
            self.board.update(last_segment=report, plan=self.trainer.plan)
            if report['stop_reason'] in ('stop', 'restart'):
                self.check_control()
            if report['stop_reason'] == 'nonfinite':
                self.decide('trainer', 'rollback', '连续 3 步出现非有限 loss/梯度，回滚到上一个检查点并降低学习率', lr_factor=0.7)
                self.trainer.rollback(0.7)
                continue
            self.bus.post(NAME, 'evaluator', 'directive', {'task': 'evaluate', 'step': report['step']})
            record, samples = self.evaluator.evaluate(self.trainer)
            self.review(report, record)
            self.board.update(phase_failures=0)
            self.write_report()
            if report['stop_reason'] in ('plan_complete', 'deadline') or now() >= train_deadline:
                self.decide('all', 'finish_training', f'训练结束（{report["stop_reason"]}），进入最终评估',
                            step=self.trainer.step, tokens=self.trainer.tokens_seen)
                return 'final_eval'

    def review(self, seg, rec):
        plan = self.trainer.plan
        history = self.board.get('eval_history', [])
        history.append({k: rec[k] for k in ('step', 'tokens_seen', 'val_loss', 'val_by_domain',
                                             'sample_rep4', 'greedy_rep4', 'time')})
        self.board.set('eval_history', history[-500:])
        best = self.board.get('best_val')
        # 1. 稳定性
        if best is not None and seg['step'] > plan['warmup_steps'] and rec['val_loss'] > best['val_loss'] * 1.05:
            self.decide('trainer', 'rollback', f'验证 loss {rec["val_loss"]} 比最佳 {best["val_loss"]} 高 5% 以上，视为发散',
                        lr_factor=0.7)
            self.trainer.rollback(0.7)
            return
        if best is None or rec['val_loss'] < best['val_loss']:
            self.trainer.save_best(rec['val_loss'])
            self.board.set('best_val', {'val_loss': rec['val_loss'], 'step': rec['step']})
        # 2. 进度与计划（v2：用最近 4 段吞吐的中位数并限制单次调整幅度，避免短时降速导致计划来回振荡）
        if seg['stop_reason'] == 'segment_time' and seg['segment_steps'] > 0:
            rates = (self.board.get('segment_rates') or []) + [seg['segment_tok_s'] / plan['tokens_per_step']]
            self.board.set('segment_rates', rates[-8:])
            recent = sorted(rates[-4:])
            step_rate = recent[len(recent) // 2] if len(recent) % 2 else sum(recent[len(recent) // 2 - 1:len(recent) // 2 + 1]) / 2
            train_deadline = self.deadline - 20 * 60
            eval_overhead = 1 + self.config['train']['eval_reserve_fraction']
            steps_possible = self.trainer.step + int(step_rate * (train_deadline - now()) / eval_overhead)
            total = plan['total_steps']
            progress = self.trainer.step / total
            new_total = None
            if steps_possible < total * 0.95:
                new_total = max(self.trainer.step + 50, int(max(steps_possible, total * 0.92)))
                why = f'近 4 段吞吐中位数 {step_rate * 3600:.0f} 步/小时，按当前计划无法在截止前完成，缩短计划（单次最多 8%）'
            elif steps_possible > total * 1.08 and progress < 0.6:
                new_total = int(min(steps_possible * 0.97, total * 1.08))
                why = f'近 4 段吞吐中位数 {step_rate * 3600:.0f} 步/小时，高于计划，延长以用满预算（进度 {progress:.0%}，单次最多 8%）'
            if new_total and new_total != total:
                plan['total_steps'] = new_total
                plan['planned_tokens'] = new_total * plan['tokens_per_step']
                self.decide('trainer', 'replan_steps', why, old_total=total, new_total=new_total)
        # 3. 数据重复 / 4. 遗忘
        weights = dict(self.trainer.sampler.weights)
        changed = False
        frac = self.trainer.step / max(1, plan['total_steps'])
        max_epochs = self.config['train']['max_epochs_per_domain']
        topups = self.board.get('topups', {})
        for domain, epochs in seg['epochs'].items():
            if frac < 0.05 or domain not in weights:
                continue
            projected = epochs / frac
            if projected > max_epochs:
                state = topups.get(domain, {})
                if state.get('status') in ('running',):
                    continue
                if state.get('status') not in ('exhausted', 'failed_twice'):
                    unique_chars = (self.board.get('merge_summary', {}).get(domain, {}).get('chars_train', 0))
                    extra = int(unique_chars * (projected / max_epochs - 1) * 1.1) + 5_000_000
                    self.launch_topup(domain, min(extra, 200_000_000))
                else:
                    new_w = weights[domain] * max_epochs / projected
                    self.decide('trainer', 'reduce_weight', f'{domain} 预计重复 {projected:.1f} 轮且无法补数据，降权',
                                domain=domain, old=round(weights[domain], 4), new=round(new_w, 4))
                    weights[domain] = new_w
                    changed = True
        if len(history) >= 3 and seg['step'] > plan['warmup_steps']:
            a, b, c = history[-3], history[-2], history[-1]
            if c['val_loss'] < a['val_loss']:
                initial = plan['initial_weights']
                for domain in weights:
                    va, vb, vc = (h['val_by_domain'].get(domain) for h in (a, b, c))
                    if None not in (va, vb, vc) and vc > vb > va and weights[domain] < 1.5 * initial.get(domain, 0):
                        new_w = min(weights[domain] * 1.2, 1.5 * initial[domain])
                        self.decide('trainer', 'increase_weight', f'{domain} 验证 loss 连续两次上升（{va}→{vb}→{vc}）而整体下降，提高采样权重',
                                    domain=domain, old=round(weights[domain], 4), new=round(new_w, 4))
                        weights[domain] = new_w
                        changed = True
        if changed:
            s = sum(weights.values())
            self.trainer.apply_weights({d: w / s for d, w in weights.items()})
        self.board.update(plan=self.trainer.plan)

    # ---------------- 补数据（并行进程） ----------------
    def launch_topup(self, domain, extra_chars):
        topups = self.board.get('topups', {})
        state = topups.get(domain, {})
        ctx = multiprocessing.get_context('spawn')
        proc = ctx.Process(target=topup_job, args=(str(RUN_DIR), domain, extra_chars, self.deadline - 2 * 3600))
        proc.start()
        topups[domain] = {'status': 'running', 'pid': proc.pid, 'started': stamp(), 'extra_chars': extra_chars,
                          'attempts': state.get('attempts', 0) + 1}
        self.board.set('topups', topups)
        self._procs = getattr(self, '_procs', {})
        self._procs[domain] = proc
        self.decide('data', 'topup', f'{domain} 预计重复次数超过上限，在独立进程中补充约 {extra_chars / 1e6:.0f}M 字符',
                    domain=domain, extra_chars=extra_chars)

    def collect_topups(self):
        topups = self.board.get('topups', {})
        for domain, state in list(topups.items()):
            if state.get('status') != 'running':
                continue
            result = read_json(RUN_DIR / 'topup' / f'{domain}.result.json')
            proc = getattr(self, '_procs', {}).get(domain)
            alive = proc.is_alive() if proc else _pid_alive(state.get('pid'))
            if result is None and alive:
                continue
            os.replace(RUN_DIR / 'topup' / f'{domain}.result.json',
                       RUN_DIR / 'topup' / f'{domain}.result.{int(now())}.json') if result else None
            if result and result.get('tokens', 0) > 0 and result.get('shard'):
                self.trainer.add_shard(domain, result['shard'])
                state['status'] = 'idle'
                self.bus.post('data', NAME, 'report', {'action': 'topup_done', **result})
            else:
                exhausted = result and 'exhausted' in str(result.get('error'))
                state['status'] = 'exhausted' if exhausted else ('failed_twice' if state.get('attempts', 0) >= 2 else 'idle')
                self.bus.post('data', NAME, 'report', {'action': 'topup_failed', 'domain': domain,
                                                       'result': result, 'status': state['status']})
            topups[domain] = state
        self.board.set('topups', topups)

    # ---------------- 最终评估 ----------------
    def phase_final_eval(self):
        if self.trainer is None:
            self.setup_training()
        record, samples = self.evaluator.evaluate(self.trainer, final=True)
        best = self.board.get('best_val')
        if best is None or record['val_loss'] < best['val_loss']:
            self.trainer.save_best(record['val_loss'])
            self.board.set('best_val', {'val_loss': record['val_loss'], 'step': record['step']})
        self.trainer.save_latest()
        self.board.set('final', {k: v for k, v in record.items()})
        self.decide('all', 'complete', f'训练完成：{self.trainer.step} 步、{self.trainer.tokens_seen / 1e9:.2f}B token，'
                    f'验证 loss {record["val_loss"]}，测试 loss {record.get("test_loss")}')
        return 'done'

    # ---------------- 报告 ----------------
    def write_report(self):
        try:
            write_text(RUN_DIR / 'REPORT.md', render_report(self.board.data, self.bus))
        except Exception:
            traceback.print_exc()


def _pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def render_report(state, bus):
    lines = ['# 多智能体 Decoder 预训练报告', '',
             f'- 更新时间：{state.get("updated", stamp())}',
             f'- 开始：{state.get("started_text")}　截止：{state.get("deadline_text")}',
             f'- 当前阶段：**{state.get("phase")}**', '']
    plan = state.get('plan')
    if plan:
        seg = state.get('last_segment') or {}
        step = seg.get('step', 0)
        lines += ['## 训练计划（决策智能体批准）', '',
                  f'- 结构：{plan["name"]}，{plan["params"] / 1e6:.1f}M 参数，{plan["model"]}',
                  f'- 精度 {plan["precision"]}，micro batch {plan["micro_batch"]} × seq {plan["seq_len"]} × 累积 {plan["accum"]}'
                  f' = {plan["tokens_per_step"]:,} token/步',
                  f'- 计划 {plan["total_steps"]:,} 步（{plan["planned_tokens"] / 1e9:.2f}B token），峰值学习率 {plan["peak_lr"]}，'
                  f'warmup {plan["warmup_steps"]} 步，余弦衰减到 10%',
                  f'- 选择理由：{plan.get("rationale", "")}', '',
                  '## 进度', '',
                  f'- 已完成 {step:,}/{plan["total_steps"]:,} 步（{step / max(1, plan["total_steps"]):.1%}），'
                  f'已训练 {seg.get("tokens_seen", 0) / 1e9:.3f}B token',
                  f'- 最近一段：训练 loss {seg.get("train_loss")}，{seg.get("segment_tok_s")} tok/s，'
                  f'梯度范数中位数 {seg.get("grad_norm_median")}，非有限步 {seg.get("nonfinite_steps")}', '']
    history = state.get('eval_history') or []
    if history:
        lines += ['## 验证集（固定块、初始配比加权）', '', '| 时间 | 步数 | token | val loss | 采样 rep4 | 贪心 rep4 |',
                  '|---|---|---|---|---|---|']
        for h in history[-15:]:
            lines.append(f'| {h["time"][5:16]} | {h["step"]:,} | {h["tokens_seen"] / 1e9:.3f}B | {h["val_loss"]} | '
                         f'{h.get("sample_rep4")} | {h.get("greedy_rep4")} |')
        last = history[-1]['val_by_domain']
        lines += ['', '分领域验证 loss（最新）：' + '，'.join(f'{d} {v}' for d, v in sorted(last.items())), '']
    samples = read_json(RUN_DIR / 'samples' / 'latest.json')
    if samples:
        lines += ['## 最新生成样例（top-p 0.9, T=0.8）', '']
        for row in [r for r in samples if r['mode'] == 'sample'][:5]:
            text = row['completion'].replace('\n', ' ')[:160]
            lines.append(f'- **{row["prompt"]}**{text}')
        lines.append('')
    if state.get('tokenizer'):
        lines += ['## Tokenizer', '', f'- {state["tokenizer"]}', '']
    if state.get('merge_summary'):
        lines += ['## 数据', '', '| 领域 | train 文档 | train 字符 | val 文档 | test 文档 |', '|---|---|---|---|---|']
        for d, s in state['merge_summary'].items():
            lines.append(f'| {d} | {s.get("docs_train", 0):,} | {s.get("chars_train", 0) / 1e6:.1f}M | '
                         f'{s.get("docs_val", 0)} | {s.get("docs_test", 0)} |')
        lines.append('')
    lines += ['## 智能体决策记录（最近 25 条）', '']
    for e in bus.tail(25, kinds=('decision',)):
        c = e['content']
        lines.append(f'- {e["time"]} **{c.get("decision")}** → {e["to"]}：{c.get("why")}')
    lines.append('')
    return '\n'.join(lines)

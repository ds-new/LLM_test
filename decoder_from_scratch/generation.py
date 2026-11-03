"""Top-k / Top-p 随机采样、贪心解码、Beam Search，以及 KV Cache 的复用和重排。

为清楚展示原理，公共生成函数接收单个无 padding 的 prompt：[1, T]。
Beam Search 内部将多个活动候选作为 batch 一次前向。
"""

import math

import torch


def filter_logits(logits, top_k=0, top_p=1.0):
    """先 Top-k，再对剩余概率做 Top-p；保留首次使累计概率达到/超过 p 的 token。"""
    if type(top_k) is not int or top_k < 0:
        raise ValueError('top_k must be a nonnegative integer (0 disables it)')
    if not 0 < top_p <= 1:
        raise ValueError('top_p must be in (0, 1]')
    filtered = logits.float().clone()
    if top_k > 0:
        values, indices = filtered.topk(min(top_k, filtered.size(-1)), dim=-1)
        filtered.fill_(float('-inf')).scatter_(-1, indices, values)
    if top_p < 1:
        values, indices = filtered.sort(dim=-1, descending=True)
        cumulative = values.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        # 向右移动一位：跨过阈值的 token 本身仍需保留；最高概率 token 总被保留。
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        filtered = filtered.scatter(-1, indices, values.masked_fill(remove, float('-inf')))
    return filtered


def reorder_cache(cache, parent_indices):
    """新 beam i 继承旧 beam parent_indices[i]；同一个父候选允许被复制多次。"""
    if cache is None:
        return None
    return [(k.index_select(0, parent_indices), v.index_select(0, parent_indices)) for k, v in cache]


def _validate_repetition(repetition_penalty, no_repeat_ngram_size):
    if not math.isfinite(repetition_penalty) or repetition_penalty < 1:
        raise ValueError('repetition_penalty must be finite and >= 1 (1 disables it)')
    if type(no_repeat_ngram_size) is not int or no_repeat_ngram_size < 0:
        raise ValueError('no_repeat_ngram_size must be a nonnegative integer (0 disables it)')


def repetition_logits(logits, tokens, repetition_penalty=1.0, no_repeat_ngram_size=0):
    """按完整历史（含 prompt）抑制重复；每个已出现 token 只处罚一次。

    正 logit 除以 penalty，负 logit 乘以 penalty。n-gram 是 token 序列，非字符。
    每个 beam 使用自己的历史。只改变下一 token 分布，不修改模型权重。
    """
    result = logits.float().clone()
    if repetition_penalty != 1:
        seen = result.gather(1, tokens)
        adjusted = torch.where(seen < 0, seen * repetition_penalty, seen / repetition_penalty)
        result.scatter_(1, tokens, adjusted)
    n = no_repeat_ngram_size
    if n and tokens.size(1) >= n:
        for row, history in enumerate(tokens.tolist()):
            prefix = history[-(n - 1):] if n > 1 else []
            banned = {history[i + n - 1] for i in range(len(history) - n + 1)
                      if history[i:i + n - 1] == prefix}
            if banned:
                result[row, list(banned)] = float('-inf')
    return result


def _validate_request(model, input_ids, max_new_tokens, eos_token_id):
    if input_ids.ndim != 2 or input_ids.size(0) != 1 or input_ids.size(1) == 0:
        raise ValueError('generation expects one nonempty prompt with shape [1, T]')
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError('max_new_tokens must be a nonnegative integer')
    if input_ids.size(1) + max_new_tokens > model.config.max_seq_len:
        raise ValueError('prompt length + max_new_tokens exceeds max_seq_len; shorten the request')
    if eos_token_id is not None and not 0 <= eos_token_id < model.config.vocab_size:
        raise ValueError('eos_token_id must be in the vocabulary')
    if model.training:
        raise ValueError('call model.eval() before generation')


@torch.inference_mode()
def generate(model, input_ids, max_new_tokens=100, temperature=0.8, top_k=0, top_p=1.0,
             do_sample=True, eos_token_id=None, use_cache=True,
             repetition_penalty=1.0, no_repeat_ngram_size=0):
    """返回 prompt + 续写 token。温度越小分布越尖锐；do_sample=False 为 argmax。"""
    _validate_request(model, input_ids, max_new_tokens, eos_token_id)
    _validate_repetition(repetition_penalty, no_repeat_ngram_size)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive; use do_sample=False for greedy')
    if type(top_k) is not int or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError('top_k must be nonnegative and top_p must be in (0, 1]')
    tokens, cache = input_ids.clone(), None
    for _ in range(max_new_tokens):
        # 第一次 prefill 全 prompt；以后只计算刚生成的 token。
        current = tokens[:, -1:] if cache is not None else tokens
        out = model(current, past_key_values=cache, use_cache=use_cache, last_logits_only=True)
        cache = out.past_key_values
        logits = out.logits[:, -1].float()
        if repetition_penalty != 1 or no_repeat_ngram_size:
            logits = repetition_logits(logits, tokens, repetition_penalty, no_repeat_ngram_size)
            if not torch.isfinite(logits).any():
                raise ValueError('no finite continuation remains; relax repetition controls')
        if do_sample:
            probabilities = filter_logits(logits / temperature, top_k, top_p).softmax(dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)
        tokens = torch.cat((tokens, next_token), dim=1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
    return tokens


@torch.inference_mode()
def beam_search(model, input_ids, max_new_tokens=100, num_beams=4, length_penalty=1.0,
                eos_token_id=None, use_cache=True,
                repetition_penalty=1.0, no_repeat_ngram_size=0):
    """确定性 Beam Search；按累计 log probability 扩展，最终按 score / length**alpha 排序。

    length 只计新生成 token（包含 EOS），不计 prompt。终止候选单独保存，不再扩展。
    有限 beam width 属于近似搜索，不保证找到整个输出空间的全局最优序列。
    """
    _validate_request(model, input_ids, max_new_tokens, eos_token_id)
    _validate_repetition(repetition_penalty, no_repeat_ngram_size)
    if type(num_beams) is not int or not 1 <= num_beams <= model.config.vocab_size:
        raise ValueError('num_beams must be an integer in [1, vocab_size]')
    if not math.isfinite(length_penalty) or length_penalty < 0:
        raise ValueError('length_penalty must be finite and nonnegative')
    if num_beams == 1:
        return generate(model, input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                        eos_token_id=eos_token_id, use_cache=use_cache,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size)
    if max_new_tokens == 0:
        return input_ids.clone()
    prompt_len = input_ids.size(1)
    sequences = input_ids.clone()
    scores = torch.zeros(1, device=input_ids.device)
    cache, finished = None, []

    def normalized(score, sequence):
        return float(score) / ((sequence.size(0) - prompt_len) ** length_penalty)

    for step in range(max_new_tokens):
        current = sequences[:, -1:] if cache is not None else sequences
        out = model(current, past_key_values=cache, use_cache=use_cache, last_logits_only=True)
        logits = out.logits[:, -1].float()
        if repetition_penalty != 1 or no_repeat_ngram_size:
            logits = repetition_logits(logits, sequences, repetition_penalty, no_repeat_ngram_size)
            # 约束耗尽的 beam 淘汰；避免全 -inf 的 softmax 产生 NaN 污染其他候选。
            dead = ~torch.isfinite(logits).any(dim=-1, keepdim=True)
            log_probs = logits.masked_fill(dead, 0).log_softmax(dim=-1).masked_fill(dead, float('-inf'))
        else:
            log_probs = logits.log_softmax(dim=-1)
        candidates = scores[:, None] + log_probs
        vocab_size = candidates.size(-1)
        # 最多 K 个 EOS 被移到 finished，取 2K 项通常足够留下 K 个活动候选。
        values, flat_indices = candidates.flatten().topk(min(2 * num_beams, candidates.numel()))
        parents, next_tokens, live_scores = [], [], []
        for rank, (value, flat_index) in enumerate(zip(values.tolist(), flat_indices.tolist())):
            if not math.isfinite(value):
                continue
            parent, token = divmod(flat_index, vocab_size)
            if token == eos_token_id:
                if rank < num_beams:
                    sequence = torch.cat((sequences[parent], sequences.new_tensor([token])))
                    finished.append((normalized(value, sequence), sequence))
            elif len(parents) < num_beams:
                parents.append(parent)
                next_tokens.append(token)
                live_scores.append(value)
        # 只需保留最好 K 个已结束候选，避免随生成长度无限累积。
        finished = sorted(finished, key=lambda item: item[0], reverse=True)[:num_beams]
        if not parents:
            sequences = sequences[:0]
            break
        parent_indices = input_ids.new_tensor(parents)
        sequences = torch.cat((sequences.index_select(0, parent_indices),
                               input_ids.new_tensor(next_tokens)[:, None]), dim=1)
        scores = log_probs.new_tensor(live_scores)
        if use_cache and step + 1 < max_new_tokens:
            cache = reorder_cache(out.past_key_values, parent_indices)
    # 到达长度上限的活动候选，也参与最终排名。
    finished.extend((normalized(score.item(), seq), seq) for score, seq in zip(scores, sequences))
    if not finished:
        raise RuntimeError('beam search found no finite-probability continuation')
    return max(finished, key=lambda item: item[0])[1].unsqueeze(0)

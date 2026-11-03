"""数据清洗规则（纯标准库，可在下载子进程中运行）。

规则沿用工程 scale_cleaning.py 的领域感知过滤，并补充页脚/重复行清理与段落级近重复去重。
这是可扩展到上亿字符的自动规则，不是逐篇人工审查；抽样结果写入 data_samples.json 供人工抽查。
"""

from collections import Counter
import hashlib
import re
import unicodedata

SPAM = re.compile(
    r'博彩|百家乐|娱乐城|送彩金|老虎机|投注|代开发票|加微信|加QQ|扫码|亚搏|yabo|fun88|ag8|开云体育|开元体育|必威|买球|'
    r'真人娱乐|优惠券|特价促销|欢迎选购|厂家直销|招商加盟|联系电话|咨询热线|六合彩|马会|特码|开奖|www\.\d{4,}|'
    r'https?://\d{4,}|移民项目|办理移民|色情|成人视频|av女优', re.I)
NAV = re.compile(
    r'^(?:上一篇|下一篇|相关阅读|相关推荐|推荐阅读|责任编辑|本文来源|文章来源|版权声明|免责声明|分享至|点击上方|'
    r'关注公众号|阅读原文|返回首页|收藏本站|相关文章|热门推荐|原标题|图片来源|来源[:：]|编辑[:：]|作者[:：].{0,20}$)'
    r'[:：\s，,]?.*$', re.I)
FOOTER = re.compile(r'^(?:[\[【]?责任编辑[:：].*|[\[【]编辑[:：].*|Powered by .+|©\s*\d{4}.*|版权所有[:：].*|'
                    r'扫码关注公众号.*|扫一扫关注公众号.*|点击.{0,10}关注.*)$', re.I)
PROMO = re.compile(r'品牌推广|市场占有率|全球.{0,12}第一|企业资质|招商|欢迎咨询|销售热线|品牌公关|战略愿景|最具投资价值|'
                   r'最具潜力|携手共赢|创新发展峰会|降本增效')
MARKUP = re.compile(r'<(?:div|script|iframe|span|table|p\s)|&nbsp;|&gt;|&lt;|\[img\]|\{\{', re.I)
VISUAL = re.compile(r'如下图|如图所示|如表所示|见图\s*\d|下图所示')
FORTUNE = re.compile(r'八字.{0,8}(?:喜神|大运|忌神)|批过八字|风水大师|算命')
CODE_DOMAINS = ('programming', 'en_programming', 'zh_ai')
DOMAIN_KEYWORDS = {
    'zh_education': re.compile(r'教学|学习|课程|知识|教育|学生|教师|教材|学校'),
    'zh_math': re.compile(r'数学|方程|函数|数列|矩阵|导数|积分|几何|定理|不等式|概率|证明|解法|公式'),
    'programming': re.compile(r'代码|算法|编程|def |class |function|return |import |#include|javascript|python', re.I),
    'en_programming': re.compile(r'code|algorithm|def |class |function|return |import |#include', re.I),
}


def canonical(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()


def doc_hash(text):
    return hashlib.blake2b(canonical(text).encode('utf-8'), digest_size=16).hexdigest()


def paragraph_hashes(text, min_len=30):
    """段落指纹：长度 >= min_len 的非空行，规范化后取 8 字节哈希。用于发现模板/转载拼接。"""
    out = []
    for line in text.split('\n'):
        line = canonical(line)
        if len(line) >= min_len:
            out.append(int.from_bytes(hashlib.blake2b(line.encode('utf-8'), digest_size=8).digest(), 'big'))
    return out


def _strip_lines(text):
    kept = []
    for line in text.split('\n'):
        stripped = line.strip()
        if len(stripped) <= 200 and (NAV.fullmatch(stripped) or FOOTER.fullmatch(stripped)):
            continue
        if kept and 15 <= len(stripped) <= 200 and stripped == kept[-1].strip():
            continue  # 连续重复行（重复标题等）
        kept.append(line.rstrip())
    text = '\n'.join(kept)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def clean(text, domain, min_chars=300, max_chars=50000):
    """返回 (清洗后文本, None) 或 (None, 拒绝原因)。"""
    if not isinstance(text, str):
        return None, 'not_text'
    if '�' in text or '\x00' in text:
        return None, 'corruption'
    text = unicodedata.normalize('NFC', text).replace('\r\n', '\n').replace('\r', '\n').replace('　', ' ')
    if not min_chars <= len(text) <= max_chars * 2:
        return None, 'length'
    if SPAM.search(text):
        return None, 'spam'
    if len(PROMO.findall(text)) >= 2:
        return None, 'promotion'
    if FORTUNE.search(text):
        return None, 'fortune'
    code = domain in CODE_DOMAINS
    if not code and MARKUP.search(text):
        return None, 'markup'
    if not code and VISUAL.search(text):
        return None, 'missing_visual'
    text = _strip_lines(text)
    if not min_chars <= len(text) <= max_chars:
        return None, 'length_after_clean'
    if domain.startswith('en_'):
        if sum(c.isascii() and c.isalpha() for c in text) / len(text) < .35:
            return None, 'language'
    else:
        cjk = sum('一' <= c <= '鿿' for c in text) / len(text)
        if cjk < (.10 if code else .35):
            return None, 'language'
    lines = [x.strip() for x in text.split('\n') if x.strip()]
    counts = Counter(lines)
    if sum((n - 1) * len(s) for s, n in counts.items() if len(s) >= 15) > len(text) * .15:
        return None, 'repetition'
    # 句内重复：同一 20 字片段出现很多次
    if len(text) >= 400:
        grams = Counter(text[i:i + 20] for i in range(0, len(text) - 20, 10))
        if grams and grams.most_common(1)[0][1] >= 6:
            return None, 'repetition'
    if not code and len(re.findall(r'[。！？.!?]', text)) < 3:
        return None, 'fragment'
    if not code and any(text.count(a) != text.count(b) for a, b in (('“', '”'), ('（', '）'))):
        return None, 'unbalanced'
    keywords = DOMAIN_KEYWORDS.get(domain)
    if keywords is not None:
        need = 1 if domain.endswith('programming') else 3
        if len(keywords.findall(text)) < need:
            return None, 'domain_mismatch'
    return text, None

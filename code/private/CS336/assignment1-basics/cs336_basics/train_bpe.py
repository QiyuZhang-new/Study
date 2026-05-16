import regex as re
from collections import Counter
from typing import Dict, List, Tuple

# GPT-2 风格预分词正则（必须使用 regex 包，标准库 re 不支持 \p{L}）
PAT = r"""'(?:[sdmt]|ll|ve|re)|\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    训练字节级 BPE 分词器
    
    Args:
        input_path: 训练文本路径
        vocab_size: 最大词表大小（含256字节+特殊令牌+合并项）
        special_tokens: 特殊令牌列表（如 ["<|endoftext|>"]）
        
    Returns:
        vocab: dict[int, bytes]  ID -> 字节序列的映射
        merges: list[tuple[bytes, bytes]]  按创建顺序排列的合并规则
    """
    # ==========================================
    # 1️⃣ 词表初始化
    # ==========================================
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    
    # 将特殊令牌预先加入词表
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
        
    merges: List[Tuple[bytes, bytes]] = []
    
    # ==========================================
    # 2️⃣ 读取文本 & 处理特殊令牌硬边界
    # ==========================================
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
        
    # 用特殊令牌切分文本，确保后续统计与合并绝不跨越文档边界
    if special_tokens:
        # 转义特殊令牌中的正则元字符，并用 () 包裹以便 re.split 保留分隔符
        split_pattern = "(" + "|".join(re.escape(st) for st in special_tokens) + ")"
        segments = re.split(split_pattern, text)
        # 过滤掉空字符串和特殊令牌本身（它们不参与合并计数）
        text_segments = [seg for seg in segments if seg and seg not in special_tokens]
    else:
        text_segments = [text]
        
    # ==========================================
    # 3️⃣ 预分词 & 统计初始字节对频率
    # ==========================================
    # 统计每个预分词（转为字节元组）的出现次数
    pretoken_counts: Counter[Tuple[bytes, ...]] = Counter()
    for seg in text_segments:
        for match in re.finditer(PAT, seg):
            token_str = match.group()
            token_bytes = tuple(bytes([b]) for b in token_str.encode("utf-8"))
            if token_bytes:  # 忽略空匹配
                pretoken_counts[token_bytes] += 1
                
    # 辅助函数：从预分词计数中计算相邻字节对的出现频率
    def get_pair_counts(pretoken_counts: Counter) -> Counter[Tuple[bytes, bytes]]:
        pair_counts = Counter()
        for pretoken, count in pretoken_counts.items():
            for i in range(len(pretoken) - 1):
                pair = (pretoken[i], pretoken[i + 1])
                pair_counts[pair] += count
        return pair_counts
        
    pair_counts = get_pair_counts(pretoken_counts)
    
    # ==========================================
    # 4️⃣ 迭代合并循环
    # ==========================================
    while len(vocab) < vocab_size and pair_counts:
        # 找最高频字节对；若频率相同，Python 的 max 会按元组字典序比较，自动选出更大的
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
        pair_a, pair_b = best_pair
        new_token = pair_a + pair_b  # 合并后的字节序列
        
        # 加入词表与合并列表
        vocab[next_id] = new_token
        merges.append(best_pair)
        next_id += 1
        
        # 应用该合并规则：更新所有预分词
        new_pretoken_counts = Counter()
        for pretoken, count in pretoken_counts.items():
            merged_pretoken = []
            i = 0
            while i < len(pretoken):
                # 检查是否命中待合并的字节对
                if i < len(pretoken) - 1 and pretoken[i] == pair_a and pretoken[i + 1] == pair_b:
                    merged_pretoken.append(new_token)
                    i += 2  # 跳过已合并的两个字节
                else:
                    merged_pretoken.append(pretoken[i])
                    i += 1
            if merged_pretoken:
                new_pretoken_counts[tuple(merged_pretoken)] = count
                
        # 更新计数器，为下一轮合并做准备
        pretoken_counts = new_pretoken_counts
        pair_counts = get_pair_counts(pretoken_counts)
        
    # ==========================================
    # 5️⃣ 返回结果
    # ==========================================
    return vocab, merges
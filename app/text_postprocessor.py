#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR 后处理管线：拼音模糊匹配 + 替换词典

处理顺序：
  1. 专有名词匹配（proper_nouns.json）— 滑动窗口 + 拼音匹配 + 英文 Levenshtein
  2. 替换词典（replacements.json）— 简单键值替换

拼音匹配模式（由松到严）：
  - 带调拼音精确：声韵调完全一致
  - 无调拼音匹配：同音不同字（其安信 → 奇安信）
  - 拼音近似匹配：≥3 字词允许 1 个音节不同
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pypinyin

logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────

DEFAULT_PROPER_NOUNS_PATH = os.path.expanduser("~/.config/vocotype/proper_nouns.json")
DEFAULT_REPLACEMENTS_PATH = os.path.expanduser("~/.config/vocotype/replacements.json")

# 英文词正则（连续字母和内部连字符/点/斜杠）
_EN_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z.\-/]*[a-zA-Z]|[a-zA-Z]")

# 非英文字符边界（中文、数字、符号等）
_NON_EN_BOUNDARY_RE = re.compile(r"(?<=[^a-zA-Z.\-/])(?=[a-zA-Z])|(?<=[a-zA-Z])(?=[^a-zA-Z.\-/])")


# ── 数据结构 ──────────────────────────────────────────────────────

@dataclass
class PinyinRepr:
    """一个词/短语的拼音表示。"""
    with_tones: Tuple[str, ...]   # ("qi2", "an1", "xin4")
    without_tones: Tuple[str, ...]  # ("qi", "an", "xin")
    is_chinese: bool               # 是否包含中文字符
    is_english: bool               # 是否纯英文
    length: int                    # 字符数


# ── 拼音工具 ──────────────────────────────────────────────────────

def _to_pinyin(word: str) -> PinyinRepr:
    """计算一个词的拼音表示。"""
    has_chinese = any("\u4e00" <= c <= "\u9fff" for c in word)
    has_english = bool(_EN_WORD_RE.fullmatch(word)) or bool(
        re.search(r"[a-zA-Z]", word)
    )

    if has_chinese:
        tones = tuple(pypinyin.lazy_pinyin(word, style=pypinyin.TONE3))
        plain = tuple(pypinyin.lazy_pinyin(word, style=pypinyin.NORMAL))
    else:
        tones = ()
        plain = ()

    return PinyinRepr(
        with_tones=tones,
        without_tones=plain,
        is_chinese=has_chinese,
        is_english=has_english and not has_chinese,
        length=len(word),
    )


def _levenshtein(a: str, b: str) -> int:
    """编辑距离（O(n*m) 空间优化版）。"""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(
                min(curr[-1] + 1, prev[j + 1] + 1, prev[j] + cost)
            )
        prev = curr
    return prev[-1]


# ── 专有名词匹配器 ──────────────────────────────────────────────

class ProperNounMatcher:
    """滑动窗口拼音匹配 + 英文模糊匹配。

    算法：
      1. 热词按长度降序排列（长词优先）
      2. 对文本逐位置滑动窗口，匹配热词
      3. 中文热词用拼音匹配，英文热词用 Levenshtein
      4. 匹配成功则替换文本中的对应片段
    """

    def __init__(self, config_path: str = DEFAULT_PROPER_NOUNS_PATH):
        self.config_path = config_path
        self._proper_nouns: List[str] = []
        self._pn_pinyins: Dict[str, PinyinRepr] = {}  # 热词 -> 拼音表示
        self._sorted: List[str] = []  # 按长度降序
        self.reload()

    def reload(self) -> None:
        """（重新）加载专有名词列表。"""
        if not os.path.exists(self.config_path):
            logger.info("专有名词配置文件不存在: %s", self.config_path)
            self._proper_nouns = []
            self._pn_pinyins = {}
            self._sorted = []
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            logger.warning("专有名词配置格式错误（应为 JSON 数组），跳过")
            self._proper_nouns = []
            self._pn_pinyins = {}
            self._sorted = []
            return

        # 去重并过滤空字符串
        seen: Set[str] = set()
        self._proper_nouns = []
        for item in raw:
            w = str(item).strip()
            if w and w not in seen:
                seen.add(w)
                self._proper_nouns.append(w)

        # 预计算拼音
        for w in self._proper_nouns:
            self._pn_pinyins[w] = _to_pinyin(w)

        # 长词优先排序
        self._sorted = sorted(self._proper_nouns, key=lambda w: (-len(w), w))

        logger.info(
            "加载了 %d 个专有名词", len(self._proper_nouns)
        )

    def apply(self, text: str) -> str:
        """对文本做专有名词匹配校正。

        算法：
          1. 中文专名：滑动窗口 + 拼音匹配（长词优先）
          2. 英文专名：先提取英文词 token，再逐个用 Levenshtein 匹配
          3. 记录已匹配的位置，避免后续重复替换
        """
        if not text or not self._sorted:
            return text

        chars = list(text)
        n = len(chars)
        # 标记哪些位置已被匹配替换，不再参与后续匹配
        occupied = [False] * n

        # 先处理中文专名（滑动窗口）
        for pn in self._sorted:
            pn_repr = self._pn_pinyins.get(pn)
            if not pn_repr or not pn_repr.is_chinese:
                continue
            chars = self._apply_chinese_single(pn, pn_repr, chars, occupied)
            n = len(chars)

        # 再处理英文专名（基于 token 提取）
        # 需要重新计算 occupied，因为 chars 已经可能变化
        chars = self._apply_english_all(chars)

        return "".join(chars)

    def _apply_chinese_single(
        self, pn: str, pn_repr: PinyinRepr, chars: List[str],
        occupied: List[bool],
    ) -> List[str]:
        """单个中文专名滑动窗口匹配。跳过已占用的位置。"""
        pn_len = len(pn)
        text_str = "".join(chars)
        text_len = len(text_str)

        if text_len < pn_len:
            return chars

        i = 0
        while i <= text_len - pn_len:
            # 如果该位置已被之前的长词占用，跳过
            if occupied[i]:
                i += 1
                continue

            segment = text_str[i : i + pn_len]
            if self._match_chinese(segment, pn, pn_repr):
                logger.debug(
                    "中文专名匹配: '%s' → '%s' (位置 %d)", segment, pn, i
                )
                prefix = list(text_str[:i])
                suffix = list(text_str[i + pn_len :])
                new_text = prefix + list(pn) + suffix

                # 标记新文本中被替换区域为 occupied
                new_occupied = (
                    occupied[:i]
                    + [True] * len(pn)
                    + occupied[i + pn_len :]
                )
                # 调整长度
                while len(new_occupied) < len(new_text):
                    new_occupied.append(False)
                occupied[:] = new_occupied
                return new_text

            i += 1

        return chars

    def _apply_english_all(self, chars: List[str]) -> List[str]:
        """提取文本中的英文词 token，逐个匹配英文专名。"""
        text = "".join(chars)
        if not text:
            return chars

        # 用正则提取英文词 token
        en_tokens = list(_EN_WORD_RE.finditer(text))
        if not en_tokens:
            return chars

        # 按 token 结束位置逆序处理（避免替换后位置偏移）
        en_pn_list = [
            pn for pn in self._sorted
            if self._pn_pinyins.get(pn) and self._pn_pinyins[pn].is_english
        ]

        for token_match in reversed(en_tokens):
            token = token_match.group()
            start = token_match.start()
            end = token_match.end()

            for pn in sorted(en_pn_list, key=len, reverse=True):
                pn_repr = self._pn_pinyins[pn]
                if self._match_english(token, pn, pn_repr):
                    logger.debug(
                        "英文专名匹配: '%s' → '%s' (位置 %d)", token, pn, start
                    )
                    chars = (
                        list(text[:start])
                        + list(pn)
                        + list(text[end:])
                    )
                    text = "".join(chars)
                    break

        return chars

    def _match_chinese(self, segment: str, pn: str, pn_repr: PinyinRepr) -> bool:
        """中文拼音匹配（逐级尝试）。"""
        seg_repr = _to_pinyin(segment)
        if not seg_repr.is_chinese:
            return False

        # 1. 完全一致（无需拼音匹配）
        if segment == pn:
            return True

        # 2. 带调拼音精确匹配
        if (
            seg_repr.with_tones
            and pn_repr.with_tones
            and seg_repr.with_tones == pn_repr.with_tones
        ):
            return True

        # 3. 无调拼音匹配（同音不同字）
        if (
            seg_repr.without_tones
            and pn_repr.without_tones
            and seg_repr.without_tones == pn_repr.without_tones
        ):
            return True

        # 4. 拼音近似匹配（≥3 字允许 1 个异音音节）
        if len(pn) >= 3 and len(segment) >= 3:
            if self._pinyin_approximate_match(seg_repr, pn_repr):
                return True

        return False

    def _match_english(self, segment: str, pn: str, pn_repr: PinyinRepr) -> bool:
        """英文模糊匹配（Levenshtein）。"""
        # 归一化
        seg_norm = segment.lower().replace("-", "").replace(" ", "").replace(".", "")
        pn_norm = pn.lower().replace("-", "").replace(" ", "").replace(".", "")

        if seg_norm == pn_norm:
            return True

        distance = _levenshtein(seg_norm, pn_norm)
        threshold = max(1, len(pn_norm) // 3)
        return distance <= threshold

    @staticmethod
    def _pinyin_approximate_match(a: PinyinRepr, b: PinyinRepr) -> bool:
        """拼音近似匹配：≥3 字允许 1 个音节不同（无调比较）。"""
        if not a.without_tones or not b.without_tones:
            return False

        min_len = min(len(a.without_tones), len(b.without_tones))
        if min_len < 3:
            return False

        diff_count = sum(
            1 for i in range(min_len) if a.without_tones[i] != b.without_tones[i]
        )
        return diff_count <= 1


# ── 替换词典 ──────────────────────────────────────────────────────

class ReplacementDict:
    """简单键→值替换词典（按 key 长度降序优先匹配）。

    支持：
      - 精确匹配
      - 子串匹配（在文本中查找替换）
      - 长 key 优先
    """

    def __init__(self, config_path: str = DEFAULT_REPLACEMENTS_PATH):
        self.config_path = config_path
        self._rules: List[Tuple[str, str]] = []
        self.reload()

    def reload(self) -> None:
        """（重新）加载替换规则。"""
        if not os.path.exists(self.config_path):
            logger.info("替换词典配置文件不存在: %s", self.config_path)
            self._rules = []
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            logger.warning("替换词典格式错误（应为 JSON 字典），跳过")
            self._rules = []
            return

        # 按 key 长度降序优先
        self._rules = sorted(
            [(str(k).strip(), str(v).strip()) for k, v in raw.items() if k and v],
            key=lambda r: (-len(r[0]), r[0]),
        )

        logger.info("加载了 %d 条替换规则", len(self._rules))

    def apply(self, text: str) -> str:
        """应用替换规则（单遍非重叠，长 key 优先，外部调用用）。

        注：主管线 (TextPostProcessor.process) 使用内部 _safe_apply_replacement，
        此方法保留供独立调用使用。
        """
        if not text or not self._rules:
            return text

        chars = list(text)
        n = len(chars)
        replaced = [False] * n

        for key, value in self._rules:
            key_len = len(key)
            i = 0
            while i <= len(chars) - key_len:
                # 跳过已被替换占用的位置
                if any(replaced[i + j] for j in range(key_len)):
                    i += 1
                    continue

                segment = "".join(chars[i : i + key_len])
                if segment == key:
                    logger.debug("替换词典: '%s' → '%s' (位置 %d)", key, value, i)
                    # 替换
                    chars = (
                        chars[:i]
                        + list(value)
                        + chars[i + key_len :]
                    )
                    # 更新 replaced 标记
                    new_replaced = (
                        replaced[:i]
                        + [True] * len(value)
                        + replaced[i + key_len :]
                    )
                    replaced = new_replaced
                    i += len(value)
                else:
                    i += 1

        return "".join(chars)


# ── 管线 ──────────────────────────────────────────────────────────

class TextPostProcessor:
    """ASR 后处理管线。

    管线顺序（单遍共享 occupied 标记，避免重复替换）：
      1. 替换词典（ReplacementDict）— 精确子串替换
      2. 专有名词拼音匹配（ProperNounMatcher）— 中文拼音 + 英文模糊
      3. 替换词典补扫 — 对未被专名覆盖的部分
    """

    def __init__(
        self,
        proper_nouns_path: str = DEFAULT_PROPER_NOUNS_PATH,
        replacements_path: str = DEFAULT_REPLACEMENTS_PATH,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.matcher = ProperNounMatcher(proper_nouns_path)
        self.replacer = ReplacementDict(replacements_path)

    def reload(self) -> None:
        """重新加载所有配置。"""
        self.matcher.reload()
        self.replacer.reload()

    def process(self, text: str) -> str:
        """对文本执行完整后处理管线。

        先做替换词典（精确匹配），再做专有名词匹配（拼音/模糊）。
        专名匹配会标记已覆盖位置，替换词典不会重复替换被专名覆盖的区域。
        """
        if not self.enabled or not text:
            return text

        chars = list(text)
        occupied = [False] * len(chars)

        # 1. 替换词典 — 扩展短形式（如 "AI" → "人工智能"）
        #    用 occupied 追踪，但允许替换，长 key 优先
        chars, occupied = self._safe_apply_replacement(chars, occupied)

        # 2. 专有名词拼音匹配（滑动窗口 + 拼音）
        chars, occupied = self._apply_proper_nouns_chinese(chars, occupied)

        # 3. 英文专名匹配（token 边界）
        chars = self._apply_proper_nouns_english(chars)

        return "".join(chars)

    def _safe_apply_replacement(
        self, chars: List[str], occupied: List[bool],
    ) -> Tuple[List[str], List[bool]]:
        """替换词典：精确子串替换，避免破坏已完整的专名。

        智能跳过：
          - 已被 occupied 标记的位置
          - key 后面紧跟着 value 的后缀字符（说明文本已完整，无需再替换）
        """
        text = "".join(chars)
        for key, value in self.replacer._rules:
            key_len = len(key)
            i = 0
            while i <= len(chars) - key_len:
                if any(occupied[i + j] for j in range(key_len)):
                    i += 1
                    continue
                segment = "".join(chars[i : i + key_len])
                if segment == key:
                    # 智能处理后缀重叠：value_suffix 可能与原文已有部分重叠
                    # 例如 "达梦"→"达梦数据库" + 原文"库" → 只加"数据"避免"达梦数据库库"
                    value_suffix = value[len(key):]
                    if value_suffix and i + key_len < len(chars):
                        following = "".join(
                            chars[i + key_len : i + key_len + len(value_suffix)]
                        )
                        if following == value_suffix:
                            # 已包含完整形式，标记为 occupied 避免后续匹配
                            for j in range(key_len + len(value_suffix)):
                                if i + j < len(chars):
                                    occupied[i + j] = True
                            i += len(key)
                            continue
                        # 存在部分重叠：只添加非重叠部分
                        # 从 suffix 末尾往前匹配（"库" 匹配 "数据库" 的末尾）
                        overlap = 0
                        max_check = min(len(value_suffix), len(chars) - i - key_len)
                        for ol in range(max_check, 0, -1):
                            following = "".join(chars[i + key_len : i + key_len + ol])
                            if following == value_suffix[-ol:]:
                                overlap = ol
                                break
                        if overlap > 0:
                            # 部分重叠：重叠在 suffix 末尾，"数据库"+"库"→只加"数据"，原文"库"保留
                            addition = list(value_suffix[:-overlap]) if overlap < len(value_suffix) else []
                            if addition:
                                logger.debug(
                                    "替换词典: '%s'→'%s' (位置 %d, 重叠 %d 字符)",
                                    key, value, i, overlap,
                                )
                                # 只在原文前加缺少部分，重叠字符保留在原文中
                                chars = (
                                    chars[:i]
                                    + list(key)
                                    + addition
                                    + chars[i + key_len:]
                                )
                                new_len = len(key) + len(addition)
                                new_occ = (
                                    occupied[:i]
                                    + [True] * (len(key) + len(value_suffix))
                                    + occupied[i + key_len + overlap:]
                                )
                                occupied = new_occ
                                i += new_len
                            else:
                                # 完全重叠，标记 occupied
                                for j in range(key_len + overlap):
                                    if i + j < len(chars):
                                        occupied[i + j] = True
                                i += len(key)
                            continue

                    logger.debug("替换词典: '%s' → '%s' (位置 %d)", key, value, i)
                    chars = chars[:i] + list(value) + chars[i + key_len :]
                    new_occ = (
                        occupied[:i]
                        + [True] * len(value)
                        + occupied[i + key_len :]
                    )
                    occupied = new_occ
                    i += len(value)
                else:
                    i += 1
        return chars, occupied

    def _apply_proper_nouns_chinese(
        self, chars: List[str], occupied: List[bool],
    ) -> Tuple[List[str], List[bool]]:
        """中文专名匹配：滑动窗口 + 拼音，跳过已占用位置。"""
        for pn in self.matcher._sorted:
            pn_repr = self.matcher._pn_pinyins.get(pn)
            if not pn_repr or not pn_repr.is_chinese:
                continue
            chars = self.matcher._apply_chinese_single(
                pn, pn_repr, chars, occupied
            )
        return chars, occupied

    def _apply_proper_nouns_english(self, chars: List[str]) -> List[str]:
        """英文专名匹配：基于 token 边界。"""
        return self.matcher._apply_english_all(chars)

    def process_debug(self, text: str) -> Tuple[str, dict]:
        """带调试信息的后处理。"""
        if not self.enabled or not text:
            return text, {"steps": []}

        original = text
        processed = self.process(text)
        steps = []
        if processed != original:
            steps.append({"step": "correction", "before": original, "after": processed})
        return processed, {"steps": steps}

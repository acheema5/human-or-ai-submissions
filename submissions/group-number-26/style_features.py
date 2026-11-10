"""Passage-level measurements used by the Group 26 style models."""

import re
from collections import Counter
from statistics import mean, pstdev


WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
CONTRACTION = re.compile(r"\b(?:\w+n['’]t|\w+['’](?:m|re|ve|ll|d))\b", re.I)
PUNCTUATION = ",;:!?()\"“”—–-"
FEATURES_WITHOUT_LAYOUT_PUNCTUATION = (0, 1, 2, 3, 4, 5, 8, 9)


def style_features(text):
    words = WORD.findall(text.lower())
    count = max(1, len(words))
    chars = max(1, len(text))
    sentence_lengths = [len(WORD.findall(part)) for part in SENTENCE_BREAK.split(text)]
    sentence_lengths = [length for length in sentence_lengths if length] or [0]
    word_counts = Counter(words)
    punctuation = Counter(text)
    return [
        mean(sentence_lengths),
        pstdev(sentence_lengths),
        mean(map(len, words)) if words else 0,
        len(word_counts) / count**0.5,
        sum(n == 1 for n in word_counts.values()) / count,
        len(CONTRACTION.findall(text)) / count,
        text.count("\n") / count,
        len(re.findall(r"\n\s*\n", text)) / count,
        sum(c.isupper() for c in text) / chars,
        sum(c.isdigit() for c in text) / chars,
        *(punctuation[mark] / chars for mark in PUNCTUATION),
    ]

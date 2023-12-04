#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, cast

import numpy
from numpy.core.multiarray import dot
from numpy.linalg import norm
from spacy.lang.en import English, EnglishDefaults
from spacy.tokenizer import Tokenizer

from dsl.input import DSLColumn


@dataclass
class DSLTokenizer:
    tokenizer: Tokenizer

    def tokenize(
        self, sentences: Iterable[str]
    ) -> Generator[Iterable[str], None, None]:
        for doc in self.tokenizer.pipe(sentences, batch_size=50):
            yield (str(w) for w in doc)


def jaccard_sim_test(col1: DSLColumn, col2: DSLColumn):
    col1data = set(col1.get_textual_data())
    col2data = set(col2.get_textual_data())

    if len(col1data) == 0 or len(col2data) == 0:
        return 0

    return len(col1data.intersection(col2data)) / len(col1data.union(col2data))


def cosine_similarity(vec1: numpy.ndarray, vec2: numpy.ndarray) -> float:
    norm1 = norm(vec1)
    norm2 = norm(vec2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot(vec1, vec2) / (norm1 * norm2)


class CustomizedEnglishDefaults(EnglishDefaults):
    infixes = cast(list[str], EnglishDefaults.infixes) + [
        "(?<=[0-9A-Za-z])[\\.](?=[0-9])",
        "(?<=[0-9])[\\.](?=[0-9A-Za-z])",
    ]


class CustomizedEnglish(English):
    Defaults = CustomizedEnglishDefaults


def get_tokenizer() -> DSLTokenizer:
    return DSLTokenizer(CustomizedEnglish().tokenizer)

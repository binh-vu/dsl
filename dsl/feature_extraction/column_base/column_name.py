#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import annotations

import re


def jaccard_sim_test(col1_name: str, col2_name: str, lower: bool = False) -> float:
    if lower:
        lbl1 = {x.lower() for x in tokenize_label(col1_name)}
        lbl2 = {x.lower() for x in tokenize_label(col2_name)}
    else:
        lbl1 = set(tokenize_label(col1_name))
        lbl2 = set(tokenize_label(col2_name))

    return len(lbl1.intersection(lbl2)) / len(lbl1.union(lbl2))


camel_reg = re.compile(r".+?(?:(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z0-9])|$)")
split_reg = re.compile(r"_| |:|\.")


def tokenize_label(lbl: str) -> list[str]:
    result = []
    for name in split_reg.split(lbl):
        for match in camel_reg.finditer(name):
            result.append(match.group(0))

    return result

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import fastnumbers
import ftfy
import orjson
from loguru import logger
from sm.dataset import FullTable
from sm.inputs.table import ColumnBasedTable
from sm.outputs.semantic_model import SemanticType


@dataclass
class DSLTable:
    table: ColumnBasedTable
    columns: list[DSLColumn]

    @staticmethod
    def from_full_table(table: FullTable) -> DSLTable:
        return DSLTable.from_column_based_table(table.table)

    @staticmethod
    def from_column_based_table(table: ColumnBasedTable) -> DSLTable:
        return DSLTable(
            table=table,
            columns=[
                DSLColumn.from_table_column(table, col.index) for col in table.columns
            ],
        )

    def nrows(self) -> int:
        return self.table.nrows()

    def select_rows(self, indices: list[int]) -> DSLTable:
        """Select a subset of rows based on a boolean mask"""
        return DSLTable(
            table=self.table.select_rows(indices),
            columns=self.columns,
        )


@dataclass
class DSLColumn:
    id: str
    table_id: str
    col_index: int
    col_name: str
    type: ColumnType
    type_stats: dict[ColumnType, float]
    size: int
    num_array: list[int | float]
    num_idx_array: list[int]
    str_array: list[str]
    str_idx_array: list[int]

    @staticmethod
    def from_table_column(
        table: ColumnBasedTable, col_index: int, original_preprocessing: bool = False
    ):
        col = table.columns[col_index]
        col_values = [
            norm_val(
                val,
                empty_as_null=True,
                replace_not_allowed_chars=original_preprocessing,
            )
            for val in col.values
        ]
        id = f"{table.table_id}:{col_index}:{col.clean_multiline_name}"
        size = len(col.values)

        type_stats = {type: 0.0 for type in list(ColumnType)}
        for val in col_values:
            type_stats[get_type(val)] += 1
        for key, val in type_stats.items():
            if size == 0:
                type_stats[key] = 0.0
            else:
                type_stats[key] = val / size

        type = guess_col_type(id, type_stats)

        num_array = []
        num_idx_array = []
        str_array = []
        str_idx_array = []

        for idx, val in enumerate(col_values):
            if val is not None:
                if isinstance(val, (int, float)):
                    num_array.append(val)
                    num_idx_array.append(idx)
                elif original_preprocessing:
                    nums, val = split_number_text_fn(val)
                    for num in nums:
                        num_array.append(float(num))
                        num_idx_array.append(idx)
                    str_array.append(val)
                    str_idx_array.append(idx)
                else:
                    str_array.append(val)
                    str_idx_array.append(idx)

        return DSLColumn(
            id=id,
            table_id=table.table_id,
            col_index=col_index,
            col_name=col.clean_multiline_name or "",
            type=type,
            type_stats=type_stats,
            size=size,
            num_array=num_array,
            num_idx_array=num_idx_array,
            str_array=str_array,
            str_idx_array=str_idx_array,
        )

    def get_n_null(self) -> int:
        return round(self.type_stats[ColumnType.NULL] * self.size)

    def get_textual_data(self) -> list[str]:
        return self.str_array

    def get_numeric_data(self) -> list[int | float]:
        return self.num_array

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "table_id": self.table_id,
            "col_index": self.col_index,
            "col_name": self.col_name,
            "type": self.type.value,
            "type_stats": {t.value: v for t, v in self.type_stats.items()},
            "size": self.size,
            "num_array": self.num_array,
            "num_idx_array": self.num_idx_array,
            "str_array": self.str_array,
            "str_idx_array": self.str_idx_array,
        }

    @staticmethod
    def from_dict(obj: dict) -> DSLColumn:
        return DSLColumn(
            id=obj["id"],
            table_id=obj["table_id"],
            col_index=obj["col_index"],
            col_name=obj["col_name"],
            type=ColumnType(obj["type"]),
            type_stats={ColumnType(t): v for t, v in obj["type_stats"].items()},
            size=obj["size"],
            num_array=obj["num_array"],
            num_idx_array=obj["num_idx_array"],
            str_array=obj["str_array"],
            str_idx_array=obj["str_idx_array"],
        )


@dataclass
class DSLSemanticType:
    type: SemanticType
    label: str


class ColumnType(str, Enum):
    NUMBER = "number"
    STRING = "string"
    DATETIME = "datetime"
    NULL = "null"

    def is_comparable(self):
        return self == ColumnType.NUMBER or self == ColumnType.DATETIME


def norm_val(
    val: Any, empty_as_null: bool, replace_not_allowed_chars: bool
) -> Optional[str | int | float]:
    """Normalize a value"""
    if val is None:
        return None

    if fastnumbers.isfloat(val) or fastnumbers.isint(val):
        val = fastnumbers.float(val)
        if math.isnan(val):
            return None
        return val

    if isinstance(val, str):
        if replace_not_allowed_chars:
            val = re.sub(not_allowed_chars, " ", val)
        val = ftfy.fix_text(val).replace("\xa0", " ").strip()
        if len(val) == 0 and empty_as_null:
            return None

    assert isinstance(val, (str, int, float))
    return val


def get_type(val: Optional[str | int | float]) -> ColumnType:
    if val is None:
        return ColumnType.NULL

    if isinstance(val, (int, float)):
        return ColumnType.NUMBER

    return ColumnType.STRING


def guess_col_type(col_id: str, type_stats: dict[ColumnType, float]) -> ColumnType:
    if type_stats[ColumnType.STRING] > type_stats[ColumnType.NUMBER]:
        return ColumnType.STRING

    if (
        type_stats[ColumnType.NULL] < 0.7
        and (type_stats[ColumnType.NUMBER] + type_stats[ColumnType.NULL]) < 0.9
    ):
        return ColumnType.STRING

    if (
        type_stats[ColumnType.NUMBER] > type_stats[ColumnType.STRING]
        and (type_stats[ColumnType.NUMBER] + type_stats[ColumnType.NULL]) >= 0.9
    ):
        return ColumnType.NUMBER

    if type_stats[ColumnType.NULL] == 1.0:
        return ColumnType.NULL

    logger.error(
        "Cannot decide type with the stats: {}",
        orjson.dumps(type_stats, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS),
    )
    raise Exception(f"Cannot decide type of column: {col_id}")


not_allowed_chars = '[\/*?"<>|\s\t]'


def split_number_text_fn(example):
    numbers = re.findall(r"(\d+(\.\d+([Ee]\d+)?)?)", example)
    text = re.sub(r"(\d+(\.\d+([Ee]\d+)?)?)", "", example)
    return numbers, text

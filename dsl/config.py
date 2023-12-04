from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import serde.yaml

DSL_CONFIG = None


@dataclass
class DSLConfig:
    max_num_rows: int
    seed: int

    @staticmethod
    def from_yaml(infile: Path | str) -> DSLConfig:
        obj = serde.yaml.deser(infile)

        return DSLConfig(
            max_num_rows=obj["max_num_rows"],
            seed=obj["seed"],
        )

    @staticmethod
    def default() -> DSLConfig:
        return DSLConfig.from_yaml(Path(__file__).parent / "config.default.yml")

    @staticmethod
    def get_instance():
        global DSL_CONFIG
        if DSL_CONFIG is None:
            DSL_CONFIG = DSLConfig.default()
        return DSL_CONFIG

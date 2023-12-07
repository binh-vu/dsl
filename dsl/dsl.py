#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Protocol, Sequence

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

import serde.pickle
from dsl.config import DSLConfig
from dsl.generate_train_data import (
    DefaultSemanticTypeComparator,
    ISemanticTypeComparator,
    generate_training_data,
)
from dsl.input import DSLTable
from dsl.sm_type_db import SemanticTypeDB
from kgdata.models.ont_class import OntologyClass
from kgdata.models.ont_property import OntologyProperty
from sm.dataset import Example, sample_table_data
from sm.outputs.semantic_model import SemanticType


class DSLModel(Protocol):
    def predict_proba(self, X: list[list[float]] | np.ndarray) -> np.ndarray:
        """Predict probability of examples stored in a 2D array X (n_examples, n_features).
        The output is a 2D array of (n_examples, n_classes)"""
        ...


@dataclass
class DSLPrediction:
    semantic_type: SemanticType
    score: float


@dataclass
class DSLPredictionHierarchy(DSLPrediction):
    parent_stypes: list[DSLPrediction]


class DSL(object):
    VERSION = 103

    instance = None

    def __init__(
        self,
        train_examples: Sequence[Example[DSLTable]],
        exec_dir: Path,
        classes: Mapping[str, OntologyClass],
        props: Mapping[str, OntologyProperty],
    ) -> None:
        self.train_source_ids = {ex.id for ex in train_examples}
        self.exec_dir = exec_dir
        self.exec_dir.mkdir(exist_ok=True, parents=True)

        self.model: Optional[DSLModel] = None
        self.classes = classes
        self.props = props

        if (exec_dir / "stype_db.pkl").exists():
            self.stype_db = serde.pickle.deser(exec_dir / "stype_db.pkl")
        else:
            self.stype_db = SemanticTypeDB(train_examples, classes, props)

    def get_model(
        self,
        train_if_not_exist: bool = True,
        stype_cmp: Optional[ISemanticTypeComparator] = None,
        save_train_data: bool = False,
    ) -> DSLModel:
        """Try to load previous model if possible"""
        if self.model is not None:
            return self.model

        model_file = self.exec_dir / "model.pkl"
        if model_file.exists():
            logger.debug("Load previous trained model...")
            model: DSLModel = serde.pickle.deser(model_file)
            self.model = model
            return model

        if train_if_not_exist:
            self.train_model(stype_cmp, save_train_data)
            assert self.model is not None
            return self.model

        logger.error("Cannot load model...")
        raise Exception("Model doesn't exist..")

    def train_model(
        self,
        stype_cmp: Optional[ISemanticTypeComparator] = None,
        save_train_data: bool = False,
    ):
        """Train a model and save it to disk"""
        if stype_cmp is None:
            stype_cmp = DefaultSemanticTypeComparator(self.classes, self.props)
        trainset, testsets = generate_training_data(
            self.stype_db, stype_cmp, {}, include_traceback=save_train_data
        )

        # clf = LogisticRegression(class_weight="balanced")
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced", random_state=120
        )

        clf = clf.fit(trainset["x"], trainset["y"])

        y_pred = clf.predict_proba(trainset["x"])[:, 1]
        logger.debug("Performance:")
        print(classification_report(trainset["y"], y_pred >= 0.5))

        if save_train_data:
            data = {
                "refcol": trainset["refcol"],
                "col": trainset["col"],
                "y": trainset["y"],
                "y_pred": y_pred,
            }
            for i in range(len(trainset["x"][0])):
                data[f"feat{i}"] = [x[i] for x in trainset["x"]]
            pd.DataFrame(data).to_csv(self.exec_dir / "trainset.csv", index=False)

        logger.debug("Save model...")
        serde.pickle.ser(clf, self.exec_dir / "model.pkl")
        self.model = clf

    @staticmethod
    def init_instance(
        train_examples: Sequence[Example[DSLTable]],
        exec_dir: Path,
        classes: Mapping[str, OntologyClass],
        props: Mapping[str, OntologyProperty],
    ):
        assert DSL.instance is None
        cfg = DSLConfig.get_instance()
        train_examples = sample_table_data(train_examples, cfg.max_num_rows, cfg.seed)
        DSL.instance = DSL(train_examples, exec_dir, classes, props)
        return DSL.instance

    @staticmethod
    def get_instance() -> "DSL":
        if DSL.instance is None:
            raise Exception("The DSL singleton object must be initialized before use")
        return DSL.instance

    def __call__(
        self, examples: Sequence[Example[DSLTable]], top_n: int, verbose: bool = False
    ) -> list[list[list[DSLPrediction]]]:
        cfg = DSLConfig.get_instance()
        examples = sample_table_data(examples, cfg.max_num_rows, cfg.seed)
        columns = [col for ex in examples for col in ex.table.columns]
        simmatrix = self.stype_db.get_similarity_matrix(columns, verbose=verbose)

        colpreds: dict[str, list[DSLPrediction]] = {}
        for ci, col in tqdm(
            enumerate(columns), desc="Predicting semantic types", disable=not verbose
        ):
            pred = self.pred_type(ci, col.id, top_n, simmatrix)
            colpreds[col.id] = pred

        return [[colpreds[col.id] for col in ex.table.columns] for ex in examples]

    def pred_type(
        self,
        target_col_index: int,
        target_col_id: str,
        top_n: int,
        similarity_matrix: np.ndarray,
    ) -> list[DSLPrediction]:
        X = []
        refcols = [
            refcol
            for refcol in self.stype_db.train_columns
            if refcol.id != target_col_id
        ]
        for refcol in refcols:
            iref = self.stype_db.col2idx[refcol.id]
            X.append(similarity_matrix[target_col_index, iref])

        result = self.get_model().predict_proba(X)[:, 1]
        result = sorted(
            zip(result, (self.stype_db.col2types[rc.id] for rc in refcols)),
            key=lambda x: x[0],
            reverse=True,
        )
        top_k_st = {}
        for score, stype in result:
            if stype not in top_k_st:
                top_k_st[stype] = score
                if len(top_k_st) == top_n:
                    break

        return sorted(
            [DSLPrediction(stype, score) for stype, score in top_k_st.items()],
            reverse=True,
            key=lambda x: x.score,
        )

    # def pred_full_stype(
    #     self,
    #     target_col_index: int,
    #     target_col_id: str,
    #     top_n: int,
    #     similarity_matrix: np.ndarray,
    # ) -> list[tuple[tuple[bytes, bytes], float, dict[tuple[bytes, bytes], float]]]:
    #     X = []
    #     refcols = [
    #         refcol
    #         for refcol in self.stype_db.train_columns
    #         if refcol.id != target_col_id
    #     ]
    #     for refcol in refcols:
    #         iref = self.stype_db.col2idx[refcol.id]
    #         X.append(similarity_matrix[target_col_index, iref])

    #     result = self.get_model().predict_proba(X)[:, 1]
    #     result = _(
    #         zip(result, (self.stype_db.col2dnodes[rc.id] for rc in refcols))
    #     ).sort(key=lambda x: x[0], reverse=True)

    #     # each top_k_st is map between stype, its score, and list of parent stypes with score
    #     top_k_st: dict[
    #         tuple[bytes, bytes], tuple[float, dict[tuple[tuple[bytes, bytes], float]]]
    #     ] = {}
    #     for score, dnode in result:
    #         link = dnode.get_first_incoming_link()
    #         parent = link.get_source_node()
    #         parent_link = parent.get_first_incoming_link()
    #         if parent_link is None:
    #             parent_stype = None
    #         else:
    #             parent_stype = (parent_link.get_source_node().label, parent_link.label)

    #         stype = (parent.label, link.label)
    #         if stype not in top_k_st:
    #             if len(top_k_st) == top_n:
    #                 # ignore stype which doesn't make itself into top k
    #                 continue

    #             top_k_st[stype] = (score, {parent_stype: score})
    #         else:
    #             # keep looping until we collect enough parent_link, default is top 3
    #             if parent_stype not in top_k_st[stype][1]:
    #                 # if we have seen the parent_stype, we don't need to update score because it's already the greatest
    #                 top_k_st[stype][1][parent_stype] = score

    #     return sorted(
    #         [
    #             (stype, score, parent_stypes)
    #             for stype, (score, parent_stypes) in top_k_st.items()
    #         ],
    #         reverse=True,
    #         key=lambda x: x[1],
    #     )

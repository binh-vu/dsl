#!/usr/bin/python
# -*- coding: utf-8 -*-


from abc import ABC, abstractmethod
from typing import Mapping

from kgdata.models.ont_class import OntologyClass
from kgdata.models.ont_property import OntologyProperty
from sm.outputs.semantic_model import SemanticType

from dsl.input import DSLColumn
from dsl.sm_type_db import SemanticTypeDB
from dsl.sm_type_db_v2 import SemanticTypeDBV2


class ISemanticTypeComparator(ABC):
    @abstractmethod
    def __call__(self, stype1: SemanticType, stype2: SemanticType) -> float:
        """Compare two semantic types and return a score between 0 and 1. 1 means the two semantic types are identical."""
        raise NotImplementedError()


class DefaultSemanticTypeComparator(ISemanticTypeComparator):
    def __init__(
        self,
        classes: Mapping[str, OntologyClass],
        props: Mapping[str, OntologyProperty],
    ):
        self.classes = classes
        self.props = props
        self.cached_equiv_classes = {}
        self.cached_equiv_props = {}

    def __call__(self, stype1: SemanticType, stype2: SemanticType):
        if stype1.class_abs_uri not in self.cached_equiv_classes:
            equiv_cls = {stype1.class_abs_uri}
            equiv_cls.update(self.classes[stype1.class_abs_uri].equivalent_classes)
            self.cached_equiv_classes[stype1.class_abs_uri] = equiv_cls

        if stype1.predicate_abs_uri not in self.cached_equiv_props:
            equiv_props = {stype1.predicate_abs_uri}
            equiv_props.update(
                self.props[stype1.predicate_abs_uri].equivalent_properties
            )
            self.cached_equiv_props[stype1.predicate_abs_uri] = equiv_props

        return int(
            (
                stype2.class_abs_uri in self.cached_equiv_classes[stype1.class_abs_uri]
                and stype2.predicate_abs_uri
                in self.cached_equiv_props[stype1.predicate_abs_uri]
            )
        )


def generate_training_data(
    stype_db: SemanticTypeDB | SemanticTypeDBV2,
    stype_cmp: ISemanticTypeComparator,
    testsets: dict[str, list[tuple[DSLColumn, SemanticType]]],
    include_traceback: bool,
):
    if isinstance(stype_db, SemanticTypeDB):
        return generate_training_data_v1(
            stype_db, stype_cmp, testsets, include_traceback
        )
    elif isinstance(stype_db, SemanticTypeDBV2):
        return generate_training_data_v2(
            stype_db, stype_cmp, testsets, include_traceback
        )
    else:
        raise Exception("Unknown SemanticTypeDB type")


def generate_training_data_v1(
    stype_db: SemanticTypeDB,
    stype_cmp: ISemanticTypeComparator,
    testsets: dict[str, list[tuple[DSLColumn, SemanticType]]],
    include_traceback: bool,
) -> tuple[dict, dict[str, dict]]:
    trainset = {"x": [], "y": [], "refcol": [], "col": []}

    train_sim_matrix = stype_db.get_similarity_matrix(
        stype_db.train_columns, verbose=True
    )

    testset_output = {
        test_name: {"x": [], "y": [], "relcol": [], "col": []}
        for test_name in testsets.keys()
    }
    testset_matrix = {
        test_name: stype_db.get_similarity_matrix(
            [xy[0] for xy in test_columns], verbose=True
        )
        for test_name, test_columns in testsets.items()
    }

    for i, ref_col in enumerate(stype_db.train_columns):
        for j, col in enumerate(stype_db.train_columns):
            if i == j:
                continue
            trainset["x"].append(train_sim_matrix[j, i])
            trainset["y"].append(
                stype_cmp(stype_db.col2types[ref_col.id], stype_db.col2types[col.id])
            )

            if include_traceback:
                trainset["refcol"].append(ref_col.id)
                trainset["col"].append(col.id)

        for test_name, test_columns in testsets.items():
            xy_test = testset_output[test_name]
            test_sim_matrix = testset_matrix[test_name]

            for j, (col, col_stype) in enumerate(test_columns):
                xy_test["x"].append(test_sim_matrix[j, i])
                xy_test["y"].append(
                    stype_cmp(stype_db.col2types[ref_col.id], col_stype)
                )

                if include_traceback:
                    xy_test["refcol"].append(ref_col.id)
                    xy_test["col"].append(col.id)

    if len(trainset) == 0:
        raise Exception("No training data")

    return trainset, testset_output


def generate_training_data_v2(
    stype_db: SemanticTypeDBV2,
    stype_cmp: ISemanticTypeComparator,
    testsets: dict[str, list[tuple[DSLColumn, SemanticType]]],
    include_traceback: bool,
) -> tuple[dict, dict[str, dict]]:
    trainset = {"x": [], "y": [], "refcol": [], "col": []}

    traincols = [col for groupcol in stype_db.train_columns for col in groupcol.cols]

    train_sim_matrix = stype_db.get_similarity_matrix(
        traincols,
        verbose=True,
    )

    testset_output = {
        test_name: {"x": [], "y": [], "relcol": [], "col": []}
        for test_name in testsets.keys()
    }
    testset_matrix = {
        test_name: stype_db.get_similarity_matrix(
            [xy[0] for xy in test_columns], verbose=True
        )
        for test_name, test_columns in testsets.items()
    }

    for i, refgroup in enumerate(stype_db.train_columns):
        for j, col in enumerate(traincols):
            if i == j:
                continue
            trainset["x"].append(train_sim_matrix[j, i])
            trainset["y"].append(
                stype_cmp(
                    stype_db.col2types[refgroup.cols[0].id], stype_db.col2types[col.id]
                )
            )

            if include_traceback:
                trainset["refcol"].append([col.id for col in refgroup.cols])
                trainset["col"].append(col.id)

        for test_name, test_columns in testsets.items():
            xy_test = testset_output[test_name]
            test_sim_matrix = testset_matrix[test_name]

            for j, (col, col_stype) in enumerate(test_columns):
                xy_test["x"].append(test_sim_matrix[j, i])
                xy_test["y"].append(
                    stype_cmp(stype_db.col2types[refgroup.cols[0].id], col_stype)
                )

                if include_traceback:
                    xy_test["refcol"].append(refgroup.cols[0].id)
                    xy_test["col"].append(col.id)

    if len(trainset) == 0:
        raise Exception("No training data")

    return trainset, testset_output

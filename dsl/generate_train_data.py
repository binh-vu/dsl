#!/usr/bin/python
# -*- coding: utf-8 -*-


from abc import ABC, abstractmethod
from typing import Mapping

from dsl.input import DSLColumn
from dsl.sm_type_db import SemanticTypeDB
from kgdata.models.ont_class import OntologyClass
from kgdata.models.ont_property import OntologyProperty
from sm.outputs.semantic_model import SemanticType


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
    stype_db: SemanticTypeDB,
    stype_cmp: ISemanticTypeComparator,
    testsets: dict[str, list[tuple[DSLColumn, SemanticType]]],
) -> tuple[list, list, dict[str, tuple[list, list]]]:
    x_train, y_train = [], []

    train_sim_matrix = stype_db.get_similarity_matrix(
        stype_db.train_columns, verbose=True
    )

    testset_output = {test_name: ([], []) for test_name in testsets.keys()}
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
            x_train.append(train_sim_matrix[j, i])
            y_train.append(
                stype_cmp(stype_db.col2types[ref_col.id], stype_db.col2types[col.id])
            )

        for test_name, test_columns in testsets.items():
            x_test, y_test = testset_output[test_name]
            test_sim_matrix = testset_matrix[test_name]

            for j, (col, col_stype) in enumerate(test_columns):
                x_test.append(test_sim_matrix[j, i])
                y_test.append(stype_cmp(stype_db.col2types[ref_col.id], col_stype))

    if len(x_train) == 0:
        raise Exception("No training data")

    return x_train, y_train, testset_output

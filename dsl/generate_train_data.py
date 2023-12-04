#!/usr/bin/python
# -*- coding: utf-8 -*-


from dsl.input import DSLColumn
from dsl.sm_type_db import SemanticTypeDB
from sm.outputs.semantic_model import SemanticType


def generate_training_data(
    stype_db: SemanticTypeDB,
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
            y_train.append(stype_db.col2types[col.id] == stype_db.col2types[ref_col.id])

        for test_name, test_columns in testsets.items():
            x_test, y_test = testset_output[test_name]
            test_sim_matrix = testset_matrix[test_name]

            for j, (col, col_stype) in enumerate(test_columns):
                x_test.append(test_sim_matrix[j, i])
                y_test.append(col_stype == stype_db.col2types[ref_col.id])

    if len(x_train) == 0:
        raise Exception("No training data")

    return x_train, y_train, testset_output

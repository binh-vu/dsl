#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from typing import Mapping, Optional, Sequence

import numpy
from kgdata.models.ont_class import OntologyClass
from kgdata.models.ont_property import OntologyProperty
from sm.dataset import Example, FullTable, sample_table_data
from sm.misc.ray_helper import ray_map, ray_put
from sm.outputs.semantic_model import SemanticType
from tqdm.auto import tqdm

from dsl.config import DSLConfig
from dsl.feature_extraction.column_base import column_name, numeric, textual
from dsl.input import DSLColumn, DSLSemanticType, DSLTable


class SemanticTypeDB(object):
    SIMILARITY_METRICS = [
        "label_jaccard",
        "stype_jaccard",
        "num_ks_test",
        "num_mann_whitney_u_test",
        "num_jaccard",
        "text_jaccard",
        "text_tf-idf",
    ]
    instance = None

    def __init__(
        self,
        train_examples: Sequence[Example[DSLTable]],
        classes: Mapping[str, OntologyClass],
        props: Mapping[str, OntologyProperty],
    ):
        self.train_examples: Sequence[Example[DSLTable]] = train_examples
        self.train_columns: list[DSLColumn] = []
        self.col2types: dict[str, SemanticType] = {}

        for ex in self.train_examples:
            for ci, col in enumerate(ex.table.table.columns):
                col_id = ex.table.columns[col.index].id
                stypes = set()
                for sm in ex.sms:
                    if sm.has_data_node(col.index):
                        stypes.update(sm.get_semantic_types_of_column(col.index))

                if len(stypes) == 0:
                    continue
                elif len(stypes) > 1:
                    raise Exception(
                        f"column {col_id} should have only one semantic type"
                    )

                assert col_id not in self.col2types, "column id must be unique"
                self.col2types[col_id] = list(stypes)[0]
                self.train_columns.append(ex.table.columns[ci])

        self.col2idx = {col.id: idx for idx, col in enumerate(self.train_columns)}
        self.train_column_stypes = [
            DSLSemanticType(
                stype,
                classes[stype.class_abs_uri].label
                + " | "
                + props[stype.predicate_abs_uri].label,
            )
            for stype in (self.col2types[col.id] for col in self.train_columns)
        ]

        self.tfidf_db = TfidfDatabase.create(
            textual.get_tokenizer(),
            self.train_columns,
        )

    def get_similarity_matrix(
        self, columns: list[DSLColumn], verbose: bool = False
    ) -> numpy.ndarray:
        n_train_columns = len(self.train_columns)
        similarity_matrix = numpy.zeros(
            (
                len(columns),
                n_train_columns,
                len(self.SIMILARITY_METRICS),
            ),
            dtype=float,
        )

        # loop through train source ids and compute similarity between columns
        for idx, col in tqdm(
            enumerate(columns),
            desc="Compute similarity matrix",
            total=len(columns),
            disable=not verbose,
        ):
            sim_features = self._compute_feature_vectors(
                col, self.train_columns, self.train_column_stypes
            )
            similarity_matrix[idx, :, :] = numpy.asarray(sim_features).reshape(
                (n_train_columns, -1)
            )

        return similarity_matrix

    def _compute_feature_vectors(
        self,
        col: DSLColumn,
        refcols: list[DSLColumn],
        refcol_stypes: list[DSLSemanticType],
    ):
        ref_tfidfs = self.tfidf_db.compute_tfidf(refcols)
        col_tfidf = self.tfidf_db.compute_tfidf([col])[0]

        features = []
        for i, refcol in enumerate(refcols):
            features.append(
                [
                    # name features
                    column_name.jaccard_sim_test(
                        refcol.col_name, col.col_name, lower=True
                    ),
                    column_name.jaccard_sim_test(
                        refcol_stypes[i].label, col.col_name, lower=True
                    ),
                    # numeric features
                    numeric.ks_test(refcol, col),
                    numeric.mann_whitney_u_test(refcol, col),
                    numeric.jaccard_sim_test(refcol, col),
                    # text features
                    textual.jaccard_sim_test(refcol, col),
                    textual.cosine_similarity(
                        ref_tfidfs[i],
                        col_tfidf,
                    ),
                ]
            )

        return features


class TfidfDatabase(object):
    def __init__(
        self,
        tokenizer: textual.DSLTokenizer,
        vocab: dict[str, int],
        inverse_doc_freq: dict[str, int],
        col2tfidf: dict[str, numpy.ndarray],
    ) -> None:
        self.vocab = vocab
        self.inverse_doc_freq = inverse_doc_freq
        self.tokenizer = tokenizer
        self.n_docs = len(col2tfidf)
        self.cache_col2tfidf = col2tfidf

    @staticmethod
    def create(
        tokenizer: textual.DSLTokenizer, train_columns: list[DSLColumn]
    ) -> TfidfDatabase:
        tfidf_cols, vocab, inverse_doc_freq = TfidfDatabase._compute_tfidf(
            tokenizer,
            train_columns,
            len(train_columns),
            vocab=None,
            inverse_doc_freq=None,
            vocab_min_word_doc_count=0,
            vocab_ignore_number_min_doc_count=2,
        )
        return TfidfDatabase(
            tokenizer,
            vocab,
            inverse_doc_freq,
            {col.id: tfidf for col, tfidf in zip(train_columns, tfidf_cols)},
        )

    def compute_tfidf(
        self, cols: list[DSLColumn], cache: bool = False
    ) -> list[numpy.ndarray]:
        unk_cols = [col for col in cols if col.id not in self.cache_col2tfidf]
        unk_tfidf_cols, _, _ = TfidfDatabase._compute_tfidf(
            self.tokenizer, unk_cols, self.n_docs, self.vocab, self.inverse_doc_freq
        )
        unk_out = {col.id: tfidf for col, tfidf in zip(unk_cols, unk_tfidf_cols)}

        if cache:
            self.cache_col2tfidf.update(unk_out)

        return [
            self.cache_col2tfidf[col.id]
            if col.id in self.cache_col2tfidf
            else unk_out[col.id]
            for col in cols
        ]

    @staticmethod
    def _compute_tfidf(
        tokenizer: textual.DSLTokenizer,
        columns: list[DSLColumn],
        n_docs: int,
        vocab: Optional[dict[str, int]] = None,
        inverse_doc_freq: Optional[dict[str, int]] = None,
        vocab_min_word_doc_count: int = 0,
        vocab_ignore_number_min_doc_count: int = 2,
    ) -> tuple[list[numpy.ndarray], dict[str, int], dict[str, int]]:
        """Compute TF-IDF for a list of columns. If the vocab is not provided,
        the columns will be treated as the training data and the vocab and invert_doc_freq
        will be computed.
        """
        using_ray = len(columns) > 1

        # compute tf first
        tokenizer_ref = ray_put(tokenizer, using_ray=using_ray)
        tf_cols = ray_map(
            TfidfDatabase._compute_tf,
            [(tokenizer_ref, col) for col in columns],
            is_func_remote=False,
            using_ray=using_ray,
        )

        # if the vocab isn't provided, columns will be treated as the training data
        if vocab is None:
            # compute inverse_doc_freq
            inverse_doc_freq = defaultdict(lambda: 0)
            for tf_col in tf_cols:
                for w in tf_col:
                    inverse_doc_freq[w] += 1

            # build vocab
            vocab = {}
            for w in list(inverse_doc_freq.keys()):
                wc = inverse_doc_freq[w]
                should_ignore = wc < vocab_min_word_doc_count or (
                    wc < vocab_ignore_number_min_doc_count and w.isdigit()
                )
                if should_ignore:
                    # delete this word
                    del inverse_doc_freq[w]
                else:
                    vocab[w] = len(vocab)
        else:
            assert inverse_doc_freq is not None

        out = []
        for tf_col in tf_cols:
            tfidf = numpy.zeros((len(vocab)))
            for w, tf in tf_col.items():
                if w in vocab:
                    tfidf[vocab[w]] = tf * numpy.log(n_docs / (1 + inverse_doc_freq[w]))
            out.append(tfidf)
        return out, vocab, inverse_doc_freq

    @staticmethod
    def _compute_tf(
        tokenizer: textual.DSLTokenizer, col: DSLColumn
    ) -> dict[str, float]:
        counter = Counter()
        sents = (
            subsent for sent in col.get_textual_data() for subsent in sent.split("/")
        )
        for doc in tokenizer.tokenize(sents):
            counter.update((str(w) for w in doc))

        number_of_token = sum(counter.values())
        new_counter = {}
        for token, val in counter.items():
            new_counter[token] = val / number_of_token
        return new_counter

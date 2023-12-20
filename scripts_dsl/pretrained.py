from pathlib import Path
from typing import Mapping, TypeVar

from kgdata.models.ont_class import OntologyClass
from kgdata.models.ont_property import OntologyProperty
from sm.dataset import Dataset

from dsl.dsl import DSL
from dsl.input import DSLTable

T = TypeVar("T", OntologyClass, OntologyProperty)


class DynItem(Mapping[str, T]):
    def __init__(self, isclass: bool):
        self.isclass = isclass

    def __len__(self):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def __getitem__(self, uri):
        assert uri.startswith("http")
        if self.isclass:
            item = OntologyClass.empty(uri)
        else:
            item = OntologyProperty.empty(uri)
        item.label = uri.split("/")[-1]
        return item


dataset_dir = Path(__file__).parent.parent / "data"

examples = []
for dpath in dataset_dir.iterdir():
    if dpath.is_dir():
        examples += Dataset(dpath).load()

dsl = DSL(
    [ex.replace_table(DSLTable.from_full_table(ex.table)) for ex in examples],
    dataset_dir,
    DynItem(True),
    DynItem(False),
)
dsl.get_model()
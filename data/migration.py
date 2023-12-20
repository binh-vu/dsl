import csv
import json
import locale
import re
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import sm.outputs.semantic_model as O
from loguru import logger

from dsl.input import not_allowed_chars, split_number_text_fn


class Source:
    def __init__(self, name):
        self.name = name
        self.column_map: dict[str, Column] = {}
        self.empty_val_columns = {}

    @staticmethod
    def read_data(file_path: Path, model_folder_path: Path):
        source = Source(file_path.stem)
        if "full" in str(file_path.parent):
            source.read_data_from_wc_csv(str(file_path))
        elif file_path.suffix == ".csv":
            source.read_data_from_csv(str(file_path))
        elif file_path.suffix == ".json":
            source.read_data_from_json(str(file_path))
        elif file_path.suffix == ".xml":
            source.read_data_from_xml(str(file_path))
        else:
            assert file_path.suffix == ".txt"
            source.read_data_from_text_file(str(file_path))

        if model_folder_path.exists():
            (model_file_path,) = list(model_folder_path.glob(f"{source.name}.*"))
            if model_file_path.suffix == ".json":
                source.read_semantic_type_json(model_file_path)
            else:
                source.read_semantic_type_from_gold(model_file_path)
        return source

    def read_data_from_wc_csv(self, file_path):
        with open(file_path) as csv_file:
            reader = csv.DictReader(csv_file)
            headers = reader.fieldnames
            assert headers is not None
            for header in headers:
                header = header.replace(" ", "")
                self.column_map[header] = Column(header, file_path)

            idx = 0
            for row in reader:
                if idx == 0:
                    for header in self.column_map.keys():
                        if "ontology" not in row[header]:
                            del self.column_map[header]
                        else:
                            self.column_map[header].semantic_type = row[header]
                    idx = 1
                    continue
                else:
                    for header in self.column_map.keys():
                        # if "http://" in row[header]:
                        #     self.column_map[header].add_value(row[header].split("/")[-1].replace("_", " "))
                        # else:
                        self.column_map[header].add_value(row[header])

    def read_data_from_csv(self, file_path):
        with open(file_path, "r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            headers = reader.fieldnames
            assert headers is not None
            for idx, header in enumerate(headers):
                idx = str(idx)
                if header:
                    header = header.replace(" ", "")
                    self.column_map[header] = Column(header, file_path)
                    # for weather 2 data
                    self.column_map[header].semantic_type = header
            for row in reader:
                for header in row.keys():
                    if header:
                        self.column_map[header.replace(" ", "")].add_value(row[header])

    def read_data_from_json(self, file_path):
        with open(file_path, "r") as f:
            json_array = json.load(f)
            for node in json_array:
                for field in node.keys():
                    if field not in self.column_map:
                        column = Column(field, file_path)
                        self.column_map[field] = column
                    if isinstance(node[field], list):
                        for value in node[field]:
                            self.column_map[field].add_value(str(value))
                    elif isinstance(node[field], dict):
                        for field1 in node[field].keys():
                            if field1 not in self.column_map:
                                column = Column(field1, file_path)
                                self.column_map[field1] = column
                            self.column_map[field1].add_value(str(node[field][field1]))
                    else:
                        self.column_map[field].add_value(str(node[field]))

    def read_data_from_xml(self, file_path):
        xml_tree = ElementTree.parse(file_path)
        root = xml_tree.getroot()
        for child in root:
            for attrib_name in child.attrib.keys():
                if attrib_name not in self.column_map:
                    column = Column(attrib_name, file_path)
                    self.column_map[attrib_name] = column
                self.column_map[attrib_name].add_value(child.attrib[attrib_name])
            for attrib in child:
                if attrib.tag not in self.column_map:
                    column = Column(attrib.tag, file_path)
                    self.column_map[attrib.tag] = column
                self.column_map[attrib.tag].add_value(attrib.text)

    def read_data_from_text_file(self, file_path):
        with open(file_path, "r") as f:
            num_types = int(f.readline().strip())
            f.readline()
            for num_type in range(num_types):
                semantic_type = f.readline().strip()
                column = Column(str(num_type), file_path)
                column.semantic_type = "---".join(
                    [
                        part.split("/")[-1]
                        for part in semantic_type.replace("#", "").split("|")
                    ]
                )
                num_values = int(f.readline())
                for num_val in range(num_values):
                    column.add_value(f.readline().split(" ", 1)[1])
                f.readline()
                self.column_map[column.name] = column

    def read_semantic_type_json(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
            node_array = data["graph"]["nodes"]

            for node in node_array:
                if "userSemanticTypes" in node:
                    semantic_object = node["userSemanticTypes"]
                    name = node["columnName"]
                    self.set_semantic_type(semantic_object[0], name)
                    # domain = semantic_object[0]["domain"]["uri"].split("/")[-1]
                    # type = semantic_object[0]["type"]["uri"].split("/")[-1]
                    # self.column_map[name].semantic_type = domain + "---" + type

    def read_semantic_type_from_gold(self, file_path):
        with open(file_path, "r") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if len(row) > 1 and row[1].strip() in self.column_map:
                    self.column_map[row[1].strip()].semantic_type = row[0].strip()

    def set_semantic_type(self, semantic_object, name):
        # domain = semantic_object["domain"]["uri"].split("/")[-1]
        # _type = semantic_object["type"]["uri"].split("/")[-1]
        domain = semantic_object["domain"]["uri"]
        _type = semantic_object["type"]["uri"]
        try:
            if name.replace(" ", "") in self.column_map:
                self.column_map[name.replace(" ", "")].semantic_type = (
                    domain + "---" + _type
                )
            else:
                if name.replace(" ", "") not in self.empty_val_columns:
                    assert name.lower().endswith("_uri") or name.lower().endswith(
                        " uri"
                    ), "This is transformed column so it's not in the original table"
        except Exception:
            logger.exception("Hit exception when set_semantic_type")
            return


class Column:
    def __init__(self, name: str, source_name: str):
        self.source_name = source_name
        self.name = name.replace("#", "")
        self.raw_value = []
        self.value_list = []
        self.textual_list = []
        self.textual_set = set()
        self.word_set = set()
        self.semantic_type = ""
        self.numeric_list = []
        self.sample_list = []
        self.value_text = ""
        self.is_prepared = False
        self.word2vec = []
        self.word_lengths = []
        self.char_lengths = []
        self.histogram_list = []

    def add_value(self, value):
        self.raw_value.append(value)
        if not value:
            return

        value = value.strip()

        if not value or value == "NULL":
            return

        # try:
        # value = value.decode('utf-8').encode('ascii', 'ignore')
        # except:

        # try:
        #     value = value.encode("ascii", "ignore")
        # except:
        #     value = value.decode("unicode_escape").encode("ascii", "ignore")

        value = re.sub(not_allowed_chars, " ", value)

        self.word_set = self.word_set.union(set(value.split(" ")))

        if "full" in self.source_name and len(self.value_list) > 500:
            return
        self.value_list.append(value)

        self.word_lengths.append(len(value.split(" ")))
        self.char_lengths.append(len(value))

        numbers, text = split_number_text_fn(value)

        if text:
            self.value_text += " " + text

            self.textual_set.add(text)
            self.textual_list.append(text)

        if numbers:
            self.numeric_list.extend([locale.atof(num[0]) for num in numbers])


def save_source(source: Source, outdir: Path):
    (outdir / "tables").mkdir(exist_ok=True)
    data = {
        "version": 2,
        "table": {
            "version": 2,
            "table_id": outdir.name + "__" + source.name,
            "columns": [
                {"index": ci, "name": col.name, "values": col.value_list}
                for ci, col in enumerate(source.column_map.values())
            ],
        },
        "context": {
            "version": 3,
            "page_title": None,
            "page_url": None,
            "entities": [],
            "literals": [],
            "content_hierarchy": [],
        },
    }
    with open(outdir / "tables" / (source.name + ".json"), "w") as f:
        f.write(json.dumps(data))


def save_description(source: Source, outdir: Path):
    sm = O.SemanticModel()
    has_sem_type = any(
        col.semantic_type.find("---") != -1 for col in source.column_map.values()
    )
    for ci, col in enumerate(source.column_map.values()):
        if col.semantic_type.find("---") == -1:
            if has_sem_type:
                continue

            cls = "https://example.com/Column"
            prop = f"https://example.com/{col.semantic_type}"
        else:
            cls, prop = col.semantic_type.split("---")

        vid = sm.add_node(
            O.DataNode(
                col_index=ci,
                label=col.name,
            )
        )

        try:
            uid = next(
                sm.iter_filter_nodes(
                    lambda u: isinstance(u, O.ClassNode) and u.abs_uri == cls
                )
            ).id
        except StopIteration:
            # we do not have this class yet
            uid = sm.add_node(
                O.ClassNode(
                    abs_uri=cls,
                    rel_uri=cls,
                )
            )

        sm.add_edge(
            O.Edge(
                source=uid,
                target=vid,
                abs_uri=prop,
                rel_uri=prop,
            )
        )
    (outdir / "descriptions").mkdir(exist_ok=True)
    (outdir / "descriptions" / f"{source.name}.json").write_text(
        json.dumps([sm.to_dict()], indent=2)
    )


if __name__ == "__main__":
    basedir = Path(__file__).parent
    sources = []

    for dsname in ["soccer", "museum", "weather", "new_data"]:
        print(">>> dsname", dsname)
        for filepath in sorted((basedir / dsname / "data").iterdir()):
            source = Source.read_data(filepath, basedir / dsname / "model")
            sources.append(source)

            save_source(source, basedir / dsname)
            save_description(source, basedir / dsname)

    print(len(sources))

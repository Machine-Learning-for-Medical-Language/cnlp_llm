import csv
import json
import os
from pathlib import Path
from typing import Any

from inspect_ai.dataset import Dataset, MemoryDataset, Sample

CNLP_DATASET_EXTENSIONS = [".json", ".csv", ".tsv"]


def cnlp_json_dataset(
    json_file: str,
    task: str | None,
    *,
    shuffle: bool = False,
    seed: int | None = None,
    limit: int | None = None,
    encoding: str = "utf-8",
    name: str | None = None,
) -> Dataset:
    with open(json_file, encoding=encoding) as f:
        json_data: dict[str, dict[str, Any]] = json.load(f)

    rows = json_data.get("data", None)
    if rows is None:
        raise KeyError("missing data field")
    if not isinstance(rows, list):
        raise ValueError("data must be a list of instances")

    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f'all instances must be json objects. Found "{row}".')
        records.append(row)

    fields = records[0].keys()
    input_fields = [f for f in fields if f == "text" or f.startswith("text_")]
    if len(input_fields) == 0:
        raise ValueError('cnlp json datasets must contain a "text" field for input.')

    if task is not None and task not in fields:
        raise ValueError(f'"{task}" field not found in {json_file}')

    def record_to_sample(record: dict[str, Any]):
        return Sample(
            # if there are multiple input columns, we'll just take the first one
            # (everything is going into metadata anyway)
            input=record[input_fields[0]],
            target=record[task] if task is not None else "",
            metadata=record,
        )

    if name is None:
        if metadata := json_data.get("metadata", None):
            if task_name := metadata.get("task", None):
                name = task_name
                if task_version := metadata.get("version", None):
                    name += " " + task_version

    dataset = MemoryDataset(
        samples=[record_to_sample(record) for record in records],
        name=name,
        location=json_file,
    )

    if shuffle:
        dataset.shuffle(seed=seed)

    if limit:
        return dataset[0:limit]

    return dataset


def cnlp_csv_dataset(
    csv_file: str,
    task: str | None,
    *,
    delimiter: str = ",",
    shuffle: bool = False,
    seed: int | None = None,
    limit: int | None = None,
    dialect: str = "unix",
    encoding: str = "utf-8",
    name: str | None = None,
) -> Dataset:
    # read and convert samples
    with open(csv_file, encoding=encoding) as f:
        # filter out rows with empty values
        valid_data = [
            data
            for data in csv.DictReader(f, dialect=dialect, delimiter=delimiter)
            if data and any(value.strip() for value in data.values())
        ]

    name = name if name else Path(csv_file).stem

    columns = valid_data[0].keys()
    input_columns = [col for col in columns if col == "text" or col.startswith("text_")]
    if len(input_columns) == 0:
        raise ValueError('cnlp csv datasets must contain a "text" field for input.')

    if task is not None and task not in columns:
        raise ValueError(f'"{task}" column not found in {csv_file}')

    def row_to_sample(row: dict[str | Any, str | Any]):
        return Sample(
            # if there are multiple input columns, we'll just take the first one
            # (everything is going into metadata anyway)
            input=row[input_columns[0]],
            target=row[task] if task is not None else "",
            metadata=row,
        )

    dataset = MemoryDataset(
        samples=[row_to_sample(row) for row in valid_data],
        name=name,
        location=csv_file,
    )

    if shuffle:
        dataset.shuffle(seed=seed)

    if limit:
        return dataset[0:limit]

    return dataset


def cnlp_file_dataset(
    file_path: str,
    task: str | None,
    *,
    shuffle: bool = False,
    seed: int | None = None,
    limit: int | None = None,
    dialect: str = "unix",
    encoding: str = "utf-8",
    name: str | None = None,
) -> Dataset:
    _root, ext = os.path.splitext(file_path)

    dataset_kwargs = dict(
        task=task,
        shuffle=shuffle,
        seed=seed,
        limit=limit,
        encoding=encoding,
        name=name,
    )

    if ext == ".json":
        return cnlp_json_dataset(file_path, **dataset_kwargs)  # type: ignore
    elif ext == ".csv":
        return cnlp_csv_dataset(
            file_path,
            dialect=dialect,
            delimiter=",",
            **dataset_kwargs,  # type: ignore
        )
    elif ext == ".tsv":
        return cnlp_csv_dataset(
            file_path,
            dialect=dialect,
            delimiter="\t",
            **dataset_kwargs,  # type: ignore
        )
    raise ValueError(
        f"invalid file format: {file_path}. Data format must be one of {CNLP_DATASET_EXTENSIONS}"
    )


def cnlp_dataset(
    file_or_dir: str,
    task: str | None,
    *,
    split: str | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    limit: int | None = None,
    dialect: str = "unix",
    encoding: str = "utf-8",
    name: str | None = None,
) -> Dataset:
    def file_dataset_wrapper(file_path):
        return cnlp_file_dataset(
            file_path,
            task,
            shuffle=shuffle,
            seed=seed,
            limit=limit,
            dialect=dialect,
            encoding=encoding,
            name=name,
        )

    if os.path.isdir(file_or_dir):
        # if given a directory, we'll look for the split as any file in the directory with a valid extension

        splits: dict[str, str] = {}  # map split names to file paths
        for file_name in os.listdir(file_or_dir):
            file_path = os.path.join(file_or_dir, file_name)
            if os.path.isfile(file_path):
                root, ext = os.path.splitext(file_name)
                if ext in CNLP_DATASET_EXTENSIONS:
                    splits[root] = file_path

        if len(splits) == 0:
            raise ValueError(
                f"no valid data found in {file_or_dir}. Data format must be one of {CNLP_DATASET_EXTENSIONS}"
            )

        if split is None:
            raise ValueError(
                f"when loading a cnlp dataset from a directory, you must specify a split (found {list(splits.keys())})"
            )

        if split not in splits:
            raise ValueError(
                f'split "{split}" not found in {file_or_dir} (found {list(splits.keys())})'
            )

        return file_dataset_wrapper(os.path.join(splits[split]))
    else:
        if split is not None:
            raise ValueError("specifying a split is invalid when loading a single file")
        return file_dataset_wrapper(file_or_dir)

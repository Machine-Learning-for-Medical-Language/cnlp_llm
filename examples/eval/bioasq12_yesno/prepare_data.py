"""
Data Download Source:
https://participants-area.bioasq.org/datasets/
DUA needed

Data Source:
BioASQ

Important Notes:
When using this dataset, you agree that you
1) don't distribute the data to anyone else
"""

import os
import sys
import json
import csv

from pathlib import Path
import random
import pandas as pd

VERSION = 12


def bioasq_datareader(originalFilePath="resources", version=VERSION):
    """
    Finds files and read them
    """

    train_raw_data = list()  # raw_input_data
    test_data = list()

    with open(
        os.path.join(originalFilePath, f"training{version}b_new.json"), "r"
    ) as fp:
        train_raw_data = json.load(fp)["questions"]

    file_list = os.listdir(originalFilePath)
    test_file_list = [
        filename
        for filename in file_list
        if filename.startswith(str(VERSION)) and filename.find("golden") != -1
    ]

    for filename in test_file_list:
        with open(os.path.join(originalFilePath, filename), "r") as fp:
            test_data.extend(json.load(fp)["questions"])

    return train_raw_data, test_data


def bioasq_select_qtype(
    train_raw_data,
    test_data,
    q_type="yesno",
):
    train_raw_data = [qas for qas in train_raw_data if qas["type"] == q_type]
    test_data = [qas for qas in test_data if qas["type"] == q_type]

    return train_raw_data, test_data


def bioasq_split_dev(
    train_raw_data,
    q_type="yesno",
    dev_ratio=0.2,
):
    random.seed(0)

    sample_size = int(len(train_raw_data) * dev_ratio)
    dev_data = random.sample(train_raw_data, sample_size)

    train_data = [item for item in train_raw_data if item not in dev_data]

    assert len(train_raw_data) == len(train_data) + len(dev_data)

    return train_data, dev_data


def bioasq_sanity_check(
    train_data,
    dev_data,
    test_data,
):
    train_ids = {item["id"] for item in train_data}
    dev_ids = {item["id"] for item in dev_data}
    test_ids = {item["id"] for item in test_data}

    # Check for overlapping IDs
    train_dev_overlap = train_ids & dev_ids
    train_test_overlap = train_ids & test_ids
    dev_test_overlap = dev_ids & test_ids

    total_unique_ids = len(train_ids | dev_ids | test_ids)
    total_expected = len(train_data) + len(dev_data) + len(test_data)

    return {
        "train_dev_overlap": train_dev_overlap,
        "train_test_overlap": train_test_overlap,
        "dev_test_overlap": dev_test_overlap,
        "total_unique_ids": total_unique_ids,
        "total_expected": total_expected,
        "is_valid": not (train_dev_overlap or train_test_overlap or dev_test_overlap)
        and total_unique_ids == total_expected,
    }


def to_boolean(label):
    if label == "yes":
        return "Yes"
    elif label == "no":
        return "No"
    else:
        raise Exception(f"Unexpected label in dataset! {label}")


def bioasq_clean_data(data):
    """
    remove unnecessary data in dictionary
    """

    output = []
    for question in data:
        output.append(
            {
                "id": question["id"],
                "text": question["body"].replace("\t", " ").splitlines()[0],  # question
                "exact_answer": question["exact_answer"],
                "label": to_boolean(question["exact_answer"]),
            }
        )
    assert len(data) == len(output)

    return output


def main(args):
    if len(args) < 2:
        sys.stderr.write("2 required arguments: <input directory> <output directory>\n")
        sys.exit(-1)

    output_path = Path(sys.argv[-1])

    train_raw_data, test_data = bioasq_datareader(
        originalFilePath=args[0], version=VERSION
    )
    train_raw_data, test_data = bioasq_select_qtype(
        train_raw_data, test_data, q_type="yesno"
    )
    train_data, dev_data = bioasq_split_dev(train_raw_data)

    sanity_report = bioasq_sanity_check(train_data, dev_data, test_data)
    print(f"sanity_report: {sanity_report}")

    output_path.mkdir(parents=True, exist_ok=True)
    for data_name in ["train_data", "dev_data", "test_data"]:
        cleaned_data = bioasq_clean_data(eval(data_name))
        cleaned_data_df = pd.DataFrame(cleaned_data)
        cleaned_data_df.to_csv(
            os.path.join(output_path, f"{data_name}.tsv"),
            sep="\t",
            encoding="utf-8",
            index=False,
            header=True,
            quoting=csv.QUOTE_NONE,
            escapechar=None,
        )
        print(f"{data_name} written. # of instances: {len(cleaned_data_df)}")


if __name__ == "__main__":
    main(sys.argv[1:])

"""
Data Download Source:
https://github.com/ds4dh/CliniFact

Data Source:
Jin, Q., Wang, Z., Floudas, C.S. et al. Matching patients to clinical trials with large language models. Nat Commun 15, 9074 (2024). https://doi.org/10.1038/s41467-024-53081-z
https://www.nature.com/articles/s41467-024-53081-z#data-availability


"""

import csv
import os
import sys
from pathlib import Path

import pandas as pd

VERSION = 12


def datareader(original_file_path=".", version=VERSION):
    """
    Finds files and read them
    """

    train_path = os.path.join(original_file_path, "train_set.csv")
    train_data = pd.read_csv(
        train_path, sep=",", quoting=1, quotechar='"', escapechar="\\", engine="python"
    )

    dev_path = os.path.join(original_file_path, "validation_set.csv")
    dev_data = pd.read_csv(
        dev_path, sep=",", quoting=1, quotechar='"', escapechar="\\", engine="python"
    )

    test_path = os.path.join(original_file_path, "test_set.csv")
    test_data = pd.read_csv(
        test_path, sep=",", quoting=1, quotechar='"', escapechar="\\", engine="python"
    )

    return train_data, dev_data, test_data


def sanity_check(
    train_data,
    dev_data,
    test_data,
):
    train_ids = set(train_data["index"].to_list())
    dev_ids = set(dev_data["index"].to_list())
    test_ids = set(test_data["index"].to_list())

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
    if int(label) == 1:
        return "TRUE"
    elif int(label) == 0 or int(label) == 2:
        return "FALSE"
    # elif int(label) == 2:
    #    return "NONE"
    else:
        raise Exception(f"Unexpected label in dataset! {label}")


def clean_text(text):
    return (
        " ".join(text.splitlines())
        .replace("\t", " ")
        .replace("  ", " ")
        .replace("  ", " ")
    )


def form_pair(claim, abstract) -> str:
    # Note that title is not used

    return_str = " ".join(["## Input: Claim:", claim, "Abstract:", abstract])
    return clean_text(return_str)


def clean_data(data):
    """
    remove unnecessary data in dictionary
    """

    output = []
    for idx, instance in data.iterrows():
        output.append(
            {
                "id": instance["index"],
                "text": form_pair(
                    claim=clean_text(instance["claim"]),
                    abstract=clean_text(instance["article_abstract"]),
                ),
                "label": to_boolean(instance["label"]),
            }
        )
    assert len(data) == len(output)

    return output


def main(args):
    if len(args) < 2:
        sys.stderr.write("2 required arguments: <input directory> <output directory>\n")
        sys.exit(-1)

    output_path = Path(sys.argv[-1])

    train_data, dev_data, test_data = datareader(
        original_file_path=args[0], version=VERSION
    )

    sanity_report = sanity_check(train_data, dev_data, test_data)
    print(f"sanity_report: {sanity_report}")

    output_path.mkdir(parents=True, exist_ok=True)
    for data_name in ["train_data", "dev_data", "test_data"]:
        cleaned_data = clean_data(eval(data_name))
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

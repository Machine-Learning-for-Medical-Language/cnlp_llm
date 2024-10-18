import os

from cnlp_llm.eval import cnlp_dataset


def test_cnlp_dataset():
    dataset_dir = os.path.join(os.path.split(__file__)[0], "datasets")
    json_dataset_file = os.path.join(dataset_dir, "json_dataset.json")
    csv_dataset_file = os.path.join(dataset_dir, "csv_dataset.csv")
    tsv_dataset_file = os.path.join(dataset_dir, "tsv_dataset.tsv")

    # load task-a on all three datasets
    json_dataset = cnlp_dataset(json_dataset_file, task="task-a")
    csv_dataset = cnlp_dataset(csv_dataset_file, task="task-a")
    tsv_dataset = cnlp_dataset(tsv_dataset_file, task="task-a")

    assert json_dataset.name == "json-dataset-task 0.1.0"

    for dataset in [json_dataset, csv_dataset, tsv_dataset]:
        assert len(dataset) == 2
        for i, sample in enumerate(dataset, 1):
            assert sample.input == f"input-{i}"
            assert sample.target == f"output-a-{i}"

    # try loading task-b instead
    json_dataset = cnlp_dataset(json_dataset_file, task="task-b")
    csv_dataset = cnlp_dataset(csv_dataset_file, task="task-b")
    tsv_dataset = cnlp_dataset(tsv_dataset_file, task="task-b")

    for dataset in [json_dataset, csv_dataset, tsv_dataset]:
        for i, sample in enumerate(dataset, 1):
            assert sample.target == f"output-b-{i}"

    # load from directory and use split argument
    json_dataset = cnlp_dataset(dataset_dir, task="task-a", split="json_dataset")

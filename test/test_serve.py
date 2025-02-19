import pytest
from fastapi.testclient import TestClient
from inspect_ai import Task, task
from inspect_ai.solver import prompt_template
from pytest import TempPathFactory

from cnlp_llm.cli.serve import create_task_server


@task
def template_task(template: str):
    return Task(
        dataset=None,
        solver=[prompt_template(template)],
    )


@pytest.fixture()
def template_task_client(tmp_path_factory: TempPathFactory):
    server_log_dir = tmp_path_factory.mktemp("test_server_logs")
    app = create_task_server(
        task_spec=f"{__file__}@template_task",
        log_dir=str(server_log_dir),
        model_name="mockllm/model",
        t=('template="{prompt}{prompt}"',),
    )
    with TestClient(app) as client:
        yield client


def test_serve_task(template_task_client: TestClient):
    # test posting to the task
    response = template_task_client.post("/evaluate", json=["a", "b", "c"])
    response.raise_for_status()
    samples = response.json()["samples"]
    for sample, target in zip(samples, ["aa", "bb", "cc"]):
        assert sample["messages"][0]["content"] == target

    # test getting the webpage
    response = template_task_client.get("/")
    response.raise_for_status()
    assert response.content.decode().startswith("<!DOCTYPE html>")

    # test getting the task name
    response = template_task_client.get("/name")
    response.raise_for_status()
    assert response.json()["name"] == "template_task"

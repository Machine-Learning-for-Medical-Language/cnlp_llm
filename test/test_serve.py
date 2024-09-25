import subprocess

import pytest
import requests
from inspect_ai import Task, task
from inspect_ai.solver import prompt_template
from pytest import TempPathFactory
from requests.adapters import HTTPAdapter, Retry


@task
def template_task(template: str):
    return Task(
        dataset=[],
        solver=[prompt_template(template)],
    )


def start_task_server(log_dir):
    echo_task = template_task("{prompt}{prompt}")
    echo_task.serve(host="localhost", port=8000, log_dir=log_dir)


@pytest.fixture(scope="session")
def server(tmp_path_factory: TempPathFactory):
    server_log_dir = tmp_path_factory.mktemp("server_logs")
    host = "localhost"
    port = "8000"
    proc = subprocess.Popen(
        [
            "cnlp_llm",
            "serve",
            f"{__file__}@template_task",
            "--log-dir",
            server_log_dir,
            "--host",
            host,
            "--port",
            port,
            "-T",
            'template="{prompt}{prompt}"',
        ]
    )
    yield f"http://{host}:{port}/"
    proc.kill()


def test_serve_pipeline(server: str):
    s = requests.Session()
    s.mount("http://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=0.1)))

    # test posting to the pipeline
    response = s.post(f"{server}/evaluate", json=["a", "b", "c"])
    response.raise_for_status()
    samples = response.json()["samples"]
    for sample, target in zip(samples, ["aa", "bb", "cc"]):
        assert sample["messages"][0]["content"] == target

    # test getting the webpage
    response = s.get(f"{server}/")
    response.raise_for_status()
    assert response.content.decode().startswith("<!DOCTYPE html>")

    # test getting the pipeline name
    response = s.get(f"{server}/name")
    response.raise_for_status()
    assert response.json()["name"] == "template_task"

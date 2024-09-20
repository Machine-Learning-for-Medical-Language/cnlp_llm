from pathlib import Path
from multiprocessing import Process
import pytest
from pytest import TempPathFactory

import requests
from requests.adapters import Retry, HTTPAdapter
from inspect_ai.log import EvalSample
from inspect_ai.solver import prompt_template

from cnlp_llm.pipeline import Pipeline, servable


def extract_msg_text(sample: EvalSample):
    return str(sample.messages[0].content)


@servable
def get_template_pipeline(template: str):
    return Pipeline(
        name="template",
        plan=[prompt_template(template)],
        postprocess=extract_msg_text,
    )


def test_pipeline(tmp_path: Path):
    double_pipeline = get_template_pipeline("{prompt}{prompt}")
    result = double_pipeline(["a", "b", "c"], log_dir=str(tmp_path))
    assert result is not None
    assert result[0] == "aa"
    assert result[1] == "bb"
    assert result[2] == "cc"


def start_pipeline_server(log_dir):
    double_pipeline = get_template_pipeline("{prompt}{prompt}")
    double_pipeline.serve(host="localhost", port=8000, log_dir=log_dir)


@pytest.fixture(scope="session")
def server(tmp_path_factory: TempPathFactory):
    server_log_dir = tmp_path_factory.mktemp("server_logs")
    proc = Process(
        target=start_pipeline_server, args=(str(server_log_dir),), daemon=True
    )
    proc.start()
    yield "http://localhost:8000/"
    proc.kill()


def test_serve_pipeline(server: str):
    s = requests.Session()
    s.mount("http://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=0.1)))

    # test posting to the pipeline
    response = s.post(server, json=["a", "b", "c"])
    response.raise_for_status()
    result = response.json()["response"]
    assert result[0] == "aa"
    assert result[1] == "bb"
    assert result[2] == "cc"

    # test getting the webpage
    response = s.get(f"{server}/")
    response.raise_for_status()
    assert response.content.decode().startswith("<!DOCTYPE html>")

    # test getting the pipeline name
    response = s.get(f"{server}/name")
    response.raise_for_status()
    assert response.json()["name"] == "template"

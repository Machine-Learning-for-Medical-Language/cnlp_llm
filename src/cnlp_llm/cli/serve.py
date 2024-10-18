from pathlib import Path

import click
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from inspect_ai import eval as task_eval
from inspect_ai._cli.util import parse_cli_args
from inspect_ai._eval.loader import load_task_spec
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelName, get_model


@click.command()
@click.argument("task_spec")
@click.option(
    "-h",
    "--host",
    help="host address",
    type=str,
    default="localhost",
    show_default=True,
)
@click.option(
    "-p", "--port", help="port number", type=int, default=8000, show_default=True
)
@click.option(
    "--root",
    help="application root path",
    type=str,
    default="",
    show_default=True,
)
@click.option("--log-dir", type=str, envvar=["CNLP_SERVE_LOG_DIR"])
@click.option(
    "-m",
    "--model",
    "model_name",
    help="model for evaluation",
    type=str,
    envvar=["CNLP_SERVE_MODEL"],
)
@click.option(
    "-M",
    multiple=True,
    type=str,
    envvar=["CNLP_SERVE_MODEL_ARGS"],
    help="One or more native model arguments (e.g. -M arg=value)",
)
@click.option(
    "-T",
    multiple=True,
    type=str,
    envvar=["CNLP_SERVE_TASK_ARGS"],
    help="One or more task arguments (e.g. -T arg=value)",
)
def serve(
    task_spec: str,
    host: str,
    port: int,
    root: str,
    log_dir: str,
    model_name: str | None = None,
    m: tuple[str] | None = None,
    t: tuple[str] | None = None,
):
    "Start a FastAPI server to serve a task. TASK_SPEC is a path to a function that returns a Task and is decorated with @task."

    model_args = parse_cli_args(m)
    task_args = parse_cli_args(t)

    model = get_model(model_name)

    task = load_task_spec(
        task_spec,
        ModelName(model),
        task_args=task_args,
    )[0]

    app = FastAPI(title=task.name, root_path=root)

    static_path = Path(__file__).parent.resolve().joinpath("static")
    app.mount("/static", StaticFiles(directory=static_path, html=True), name="static")

    @app.post("/evaluate")
    def evaluate(input: list[str]):
        task.dataset = MemoryDataset(samples=[Sample(input=x) for x in input])
        # TODO: we can maybe use eval_async here instead? Not sure if that's necessary
        eval_result = task_eval(
            task,
            model=model_name,
            model_args=model_args,
            task_args=task_args,
            log_dir=log_dir,
        )[0]
        return eval_result.model_dump(mode="json")

    @app.get("/name")
    def get_name():
        return {"name": task.name}

    @app.get("/")
    def get_html():
        return RedirectResponse("/static")

    uvicorn.run(app, host=host, port=port)

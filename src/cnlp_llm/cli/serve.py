import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from typing import Callable, cast

import click
from inspect_ai._cli.util import parse_cli_args
from inspect_ai._util.path import chdir_python

from ..pipeline.servable import ServablePipeline, __servable_pipelines__


@click.command()
@click.argument("pipeline_path")
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
@click.option("--log-dir", type=str, envvar=["CNLP_PIPELINE_LOG_DIR"])
@click.option(
    "-m",
    "--model",
    "model_name",
    help="model for evaluation",
    type=str,
    envvar=["CNLP_PIPELINE_MODEL"],
)
@click.option(
    "-M",
    multiple=True,
    type=str,
    envvar=["CNLP_PIPELINE_MODEL_ARGS"],
    help="One or more native model arguments (e.g. -M arg=value)",
)
@click.option(
    "-P",
    multiple=True,
    type=str,
    envvar=["CNLP_PIPELINE_ARGS"],
    help="One or more pipeline arguments (e.g. -P arg=value)",
)
def serve(
    pipeline_path: str,
    host: str,
    port: int,
    root: str,
    log_dir: str,
    model_name: str | None = None,
    m: tuple[str] | None = None,
    p: tuple[str] | None = None,
):
    "Start a FastAPI server to serve a Pipeline. PIPELINE_PATH is a path to a function that returns a Pipeline and is decorated with @servable. e.g., examples/infer/pirate.py@get_pipeline"

    # parse the input path
    if "@" in pipeline_path:
        path, fn_name = pipeline_path.split("@")
    else:
        path = pipeline_path
        fn_name = None

    path = os.path.abspath(path)
    dirname, filename = os.path.split(path)
    with chdir_python(dirname):

        # load the module spec
        spec = spec_from_file_location(filename, path)
        if spec is None:
            raise ValueError(f"Failed to load spec for {path}")
        if spec.loader is None:
            raise ValueError(f"Loader not available for spec at {path}")

        # load the module
        mod = module_from_spec(spec)
        sys.modules[filename] = mod

        # execute the module
        spec.loader.exec_module(mod)

        if fn_name is None:
            if len(__servable_pipelines__) == 0:
                return ValueError(f"Servable pipeline not found at {pipeline_path}.")
            pipeline_fn = __servable_pipelines__[filename][0]
        else:
            pipeline_fn = cast(Callable[..., ServablePipeline], getattr(mod, fn_name))

    model_args = parse_cli_args(m)
    pipeline_args = parse_cli_args(p)

    pipeline_fn(**pipeline_args).serve(
        host=host,
        port=port,
        root_path=root,
        model=model_name,
        model_args=model_args,
        log_dir=log_dir,
    )

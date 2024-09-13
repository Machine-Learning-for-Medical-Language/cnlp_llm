import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from typing import cast

import click

from ..pipeline.servable import ServablePipelineFactory, __servable_pipelines__


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
def serve(pipeline_path: str, host: str, port: int):
    "Start a FastAPI server to serve a Pipeline. PIPELINE_PATH is a path to a function that returns a Pipeline and is decorated with @servable. e.g., examples/infer/pirate.py@get_pipeline"

    # parse the input path
    if "@" in pipeline_path:
        path, fn_name = pipeline_path.split("@")
    else:
        path = pipeline_path
        fn_name = None

    # load the module spec
    _, filename = os.path.split(path)
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
        pipeline_fn = cast(ServablePipelineFactory, getattr(mod, fn_name))

    pipeline_fn().serve(host=host, port=port)

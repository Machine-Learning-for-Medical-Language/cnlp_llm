from functools import partial, wraps
from pathlib import Path
from typing import Callable, cast, overload

import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from cnlp_llm.pipeline.pipeline import Pipeline


class ServablePipeline(Pipeline[str]):
    def serve(self, host: str = "localhost", port: int = 8000) -> None: ...


StrPipelineFactory = Callable[[], Pipeline[str]]
ServablePipelineFactory = Callable[[], ServablePipeline]
ServableDecorator = Callable[[StrPipelineFactory], ServablePipelineFactory]

__servable_pipelines__: dict[str, list[ServablePipelineFactory]] = {}

static_path = Path(__file__).parent.resolve().joinpath("static")


@overload
def servable(func: StrPipelineFactory) -> ServablePipelineFactory: ...


@overload
def servable(
    *, root_path: str = "", log_dir: str | None = None
) -> ServableDecorator: ...


def servable(
    func: StrPipelineFactory | None = None,
    *,
    root_path: str = "",
    log_dir: str | None = None,
) -> ServablePipelineFactory | ServableDecorator:
    if func is None:
        return cast(
            ServableDecorator,
            partial(servable, root_path=root_path, log_dir=log_dir),
        )

    @wraps(func)
    def wrapper() -> ServablePipeline:
        pipeline = cast(ServablePipeline, func())

        def serve(host: str = "localhost", port: int = 8000):
            app = FastAPI(title=pipeline.name, root_path=root_path)

            app.mount(
                "/static", StaticFiles(directory=static_path, html=True), name="static"
            )

            @app.get("/")
            def get_html():
                return RedirectResponse("/static")

            @app.post("/")
            def process(input: list[str]):
                # TODO: we can probably use pipeline.call_async here instead?
                return {"response": pipeline(input, log_dir=log_dir)}

            @app.get("/name")
            def get_name():
                return {"name": pipeline.name}

            uvicorn.run(app, host=host, port=port)

        pipeline.serve = serve
        return pipeline

    result = wrapper

    if func.__module__ not in __servable_pipelines__:
        __servable_pipelines__[func.__module__] = []
    __servable_pipelines__[func.__module__].append(result)

    return cast(ServablePipelineFactory, result)

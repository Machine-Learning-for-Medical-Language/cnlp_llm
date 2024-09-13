from functools import wraps
from pathlib import Path
from typing import ParamSpec, Callable, cast

import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from cnlp_llm.pipeline.pipeline import Pipeline


class ServablePipeline(Pipeline[str]):
    def serve(
        self,
        host: str = ...,
        port: int = ...,
        root_path: str = ...,
        **kwargs,
    ) -> None: ...


# registry of servable pipeline factories for `cnlp_llm serve`
__servable_pipelines__: dict[str, list[Callable[..., ServablePipeline]]] = {}

static_path = Path(__file__).parent.resolve().joinpath("static")

P = ParamSpec("P")


def servable(func: Callable[P, Pipeline[str]]) -> Callable[P, ServablePipeline]:

    @wraps(func)
    def wrapper(*args, **kwargs) -> ServablePipeline:
        pipeline = cast(ServablePipeline, func(*args, **kwargs))

        def serve(
            host: str = "localhost",
            port: int = 8000,
            root_path: str = "",
            **kwargs,
        ):
            app = FastAPI(title=pipeline.name, root_path=root_path)

            app.mount(
                "/static", StaticFiles(directory=static_path, html=True), name="static"
            )

            @app.post("/")
            def process(input: list[str]):
                # TODO: we can probably use pipeline.call_async here instead?
                return {"response": pipeline(input, **kwargs)}

            @app.get("/")
            def get_html():
                return RedirectResponse("/static")

            @app.get("/name")
            def get_name():
                return {"name": pipeline.name}

            uvicorn.run(app, host=host, port=port)

        pipeline.serve = serve
        return pipeline

    result = wrapper

    # register this pipeline so it can be found by `cnlp_llm serve`
    if func.__module__ not in __servable_pipelines__:
        __servable_pipelines__[func.__module__] = []
    __servable_pipelines__[func.__module__].append(result)

    return cast(Callable[P, ServablePipeline], result)

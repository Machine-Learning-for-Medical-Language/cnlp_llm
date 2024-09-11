import asyncio
from typing import Awaitable, Callable, Literal, TypeVar, cast, overload

from inspect_ai._eval.task.generate import task_generate as _task_generate
from inspect_ai._eval.task.run import resolve_dataset
from inspect_ai._util.dotenv import init_dotenv
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.model import (
    CachePolicy,
    GenerateConfig,
    GenerateConfigArgs,
    Model,
    ModelName,
    get_model,
)
from inspect_ai.solver import Solver, TaskState
from typing_extensions import Unpack

T = TypeVar("T")


@overload
def execute_plan(
    data: Dataset | list[Sample] | list[str],
    plan: list[Solver],
    *,
    model: str | Model | None = ...,
    initialize_dotenv: bool = ...,
) -> None: ...


@overload
def execute_plan(
    data: Dataset | list[Sample] | list[str],
    plan: list[Solver],
    *,
    postprocess: Literal[None],
    model: str | Model | None = ...,
    initialize_dotenv: bool = ...,
) -> None: ...


@overload
def execute_plan(
    data: Dataset | list[Sample] | list[str],
    plan: list[Solver],
    *,
    postprocess: Callable[[TaskState], Awaitable[T]],
    model: str | Model | None = ...,
    initialize_dotenv: bool = ...,
) -> list[T]: ...


def execute_plan(
    data: Dataset | list[Sample] | list[str],
    plan: list[Solver],
    *,
    postprocess: Callable[[TaskState], Awaitable[T]] | None = None,
    model: str | Model | None = None,
    initialize_dotenv: bool = True,
) -> list[T] | None:
    if len(data) == 0:
        raise ValueError("Data must not be empty.")

    if initialize_dotenv:
        init_dotenv()
    model = get_model()

    match data:
        case Dataset():
            dataset = data
        case [str(), *_]:
            dataset = MemoryDataset(samples=[Sample(input=x) for x in data])
        case [Sample(), *_]:
            dataset = MemoryDataset(samples=data)
        case _:
            raise ValueError(f"Invalid data type: {type(data)}")

    dataset, samples, task_states = asyncio.run(
        resolve_dataset(
            dataset,
            ModelName(model),
            limit=None,
            epochs=1,
            log_images=False,
            max_messages=None,
        )
    )

    async def task_generate(
        state: TaskState,
        tool_calls: Literal["loop", "single", "none"] = "loop",
        cache: bool | CachePolicy = False,
        **kwargs: Unpack[GenerateConfigArgs],
    ) -> TaskState:
        return await _task_generate(
            model=model,
            state=state,
            tool_calls=tool_calls,
            cache=cache,
            config=GenerateConfig(**kwargs),
        )

    async def execute_sample_async(task_state):
        for step in plan:
            task_state = await step(task_state, task_generate)

        if postprocess is not None:
            return await postprocess(task_state)
        return None

    async def execute_plan_async():
        return await asyncio.gather(*map(execute_sample_async, task_states))

    result = asyncio.run(execute_plan_async())
    if postprocess is not None:
        return cast(list[T], result)
    else:
        return None

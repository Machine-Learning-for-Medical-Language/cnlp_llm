import asyncio
from typing import Callable, TypeVar, Generic
from inspect_ai import Task, eval_async
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.solver import Plan, Solver, generate
from inspect_ai.log import EvalLog, EvalSample

T = TypeVar("T")


class Pipeline(Generic[T]):
    def __init__(
        self,
        name: str = "pipeline",
        plan: Plan | Solver | list[Solver] = generate(),
        postprocess: Callable[[EvalSample], T] = lambda x: x,
    ) -> None:
        self.name = name
        self.plan = plan
        self.postprocess = postprocess
        self.log: EvalLog | None = None

    async def call_async(
        self, data: Dataset | list[Sample] | list[str], **kwargs
    ) -> list[EvalLog]:
        match data:
            case Dataset():
                dataset = data
            case str():
                dataset = MemoryDataset(samples=[Sample(input=data)])
            case [str(), *_]:
                dataset = MemoryDataset(samples=[Sample(input=x) for x in data])
            case [Sample(), *_]:
                dataset = MemoryDataset(samples=data)
            case _:
                raise ValueError(f"Invalid data type: {type(data)}")

        return await eval_async(
            Task(dataset=dataset, plan=self.plan, name=self.name), **kwargs
        )

    def __call__(
        self, data: Dataset | list[Sample] | list[str], **kwargs
    ) -> list[T] | None:
        result = asyncio.run(self.call_async(data, **kwargs))
        self.log = result[0]
        samples = self.log.samples
        if samples is None:
            return None
        return [self.postprocess(sample) for sample in samples]

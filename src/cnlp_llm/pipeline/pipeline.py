import asyncio
from typing import Callable, TypeVar, Generic
from inspect_ai import Task, eval_async
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.solver import Plan, Solver, generate
from inspect_ai.log import EvalLog, EvalSample

T = TypeVar("T")


class Pipeline(Generic[T]):
    """
    An inference pipeline. This class bundles a `Plan` or list of `Solver`s
    with an optional postprocessing step into a pipeline that can be used
    later for inference.
    """

    def __init__(
        self,
        name: str = "pipeline",
        plan: Plan | Solver | list[Solver] = generate(),
        postprocess: Callable[[EvalSample], T] = lambda x: x,
    ) -> None:
        """Create a new pipeline.

        Args:
            name (str): The pipeline's name as it will show up in the progress indicator and eval logs. Defaults to "pipeline".
            plan (Plan | Solver | list[Solver]): The plan for the pipeline to execute when called. Defaults to generate().
            postprocess (Callable[[EvalSample], Any]): An optional postprocessing step that acts on resulting `EvalSample`s.
        """
        self.name = name
        self.plan = plan
        self.postprocess = postprocess
        self._last_eval_log: EvalLog | None = None

    def get_last_eval_log(self) -> EvalLog | None:
        """Get the most recent eval log. An eval log is generated whenever the pipeline is called.

        Returns:
            output (EvalLog | None): The most recent eval log, or `None` if this pipeline has never been called.
        """
        return self._last_eval_log

    async def call_async(
        self, data: Dataset | list[Sample] | list[str], **kwargs
    ) -> list[T] | None:
        """Asynchronously call this Pipeline on some data.

        Args:
            data (Dataset | list[Sample] | list[str]): The input data.
            **kwargs (dict[str, Any]): kwargs for Inspect's `eval_async()`.

        Returns:
            output (list[T] | None): If the pipeline completes successfully, this function will return a list of postprocessed results. If the pipeline is unsuccessful, this will return `None`. You can access the log via the `get_last_eval_log()` method.
        """
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

        result = await eval_async(
            Task(dataset=dataset, plan=self.plan, name=self.name), **kwargs
        )

        self._last_eval_log = result[0]
        samples = self._last_eval_log.samples
        if samples is None:
            return None

        return [self.postprocess(sample) for sample in samples]

    def __call__(
        self, data: Dataset | list[Sample] | list[str], **kwargs
    ) -> list[T] | None:
        """Call this Pipeline on some data. This method is blocking. For asynchronous execution, use `call_async()`

        Args:
            data (Dataset | list[Sample] | list[str]): The input data.
            **kwargs (dict[str, Any]): kwargs for Inspect's `eval_async()`.

        Returns:
            output (list[T] | None): If the pipeline completes successfully, this function will return a list of postprocessed results. If the pipeline is unsuccessful, this will return `None`. You can access the log via the `get_last_eval_log()` method.
        """
        return asyncio.run(self.call_async(data, **kwargs))

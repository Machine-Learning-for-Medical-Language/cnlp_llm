import logging
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING, TypeVar, cast, final

import anyio
import torch
from inspect_ai.model import ChatMessage, Model
from inspect_ai.model._providers.hf import HuggingFaceAPI
from inspect_ai.util import trace_action
from transformers import Cache, PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

if TYPE_CHECKING:
    from transformers.models.auto.modeling_auto import _BaseModelWithGenerate

logger = logging.getLogger(__name__)


@final
@dataclass
class _PrefixJob:
    encoding: BatchEncoding
    future: Future[tuple[torch.Tensor, Cache, torch.Tensor]]


@final
@dataclass
class _ChoicesJob:
    encoding: BatchEncoding  # does not include the prefix!
    prefix_cache: Cache
    first_token_logits: torch.Tensor
    future: Future[torch.Tensor]


_T = TypeVar("_T")


def _calculate_logprobs(
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    first_token_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    if first_token_logits is not None:
        logits = torch.concat((first_token_logits.unsqueeze(1), logits), 1)
        input_ids = input_ids[..., None]
    else:
        input_ids = input_ids[:, 1:, None]
        attention_mask = attention_mask[:, 1:]

    logprobs = torch.log_softmax(logits, dim=-1)  # (B, t1..tn, V)
    logprobs = logprobs[:, :-1, :]  # (B, t1..tn-1, V)
    logprobs = logprobs.gather(-1, input_ids)
    logprobs = logprobs.squeeze(-1)  # (B, t1..tn-1)
    return logprobs * attention_mask


@final
class BatchedLogprobsGenerator:
    def __init__(self, model: Model):
        if not isinstance(model.api, HuggingFaceAPI):
            raise ValueError(
                "Only HuggingFace models are supported for logprobs generation"
            )

        self.inspect_model = model
        self.hf_api: HuggingFaceAPI = model.api
        self.model: _BaseModelWithGenerate = self.hf_api.model
        self.tokenizer: PreTrainedTokenizer = self.hf_api.tokenizer  # type: ignore
        self.device: torch.device = self.model.device

        self.job_queue: Queue[_PrefixJob | _ChoicesJob] = Queue()
        self.batch_thread = Thread(target=self._process, daemon=True)
        self.batch_thread.start()

    def _tokenize_inputs(
        self,
        prefix_messages: list[ChatMessage],
        choices_messages: list[list[ChatMessage]],
    ) -> tuple[BatchEncoding, BatchEncoding]:
        def _tokenize(text: str | list[str]) -> BatchEncoding:
            return self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                padding_side="right",
                add_special_tokens=False,
            ).to(self.device)

        prefix_str: str = self.tokenizer.apply_chat_template(
            prefix_messages,  # type: ignore
            add_generation_prompt=True,
            tokenize=False,
        )  # type: ignore

        choices_strs: list[str] = []
        for choice_msg in choices_messages:
            choice_str = self.tokenizer.apply_chat_template(
                prefix_messages + choice_msg,  # type: ignore
                tokenize=False,
            ).removeprefix(prefix_str)  # type: ignore
            choices_strs.append(choice_str)

        return _tokenize(prefix_str), _tokenize(choices_strs)

    def _run_prefix(
        self, prefix_encoding: BatchEncoding
    ) -> tuple[torch.Tensor, Cache, torch.Tensor]:
        with torch.inference_mode():
            out = self.model(
                **prefix_encoding,
                use_cache=True,
                return_dict=True,
            )
        logprobs = _calculate_logprobs(
            prefix_encoding.input_ids,
            out.logits,
            prefix_encoding.attention_mask,
        )
        final_logits = out.logits[:, -1, :]
        return logprobs, out.past_key_values, final_logits

    def _run_choices(
        self,
        choices_encoding: BatchEncoding,
        prefix_cache: Cache,
        first_token_logits: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = choices_encoding.input_ids
        attention_mask = choices_encoding.attention_mask

        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids,
                past_key_values=prefix_cache,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

        logprobs = _calculate_logprobs(
            input_ids,
            out.logits,
            attention_mask,
            first_token_logits=first_token_logits,
        )

        return logprobs

    def _process(self):
        while True:
            # block until there's a job
            job = self.job_queue.get()
            if isinstance(job, _PrefixJob):
                logprobs, cache, final_logits = self._run_prefix(job.encoding)
                job.future.set_result((logprobs, cache, final_logits))
            else:  # _ChoicesJob
                # TODO smarter batching
                logprobs = self._run_choices(
                    job.encoding, job.prefix_cache, job.first_token_logits
                )
                job.future.set_result(logprobs)

    async def _resolve_future(self, future: Future[_T]) -> _T:
        with trace_action(logger, "HF Logprobs Generator", "HF Logprobs Generator"):
            while True:
                try:
                    return future.result(timeout=0.01)
                except TimeoutError:
                    pass
                await anyio.sleep(1)

    async def get_choice_token_logprobs(
        self,
        prefix: list[ChatMessage],
        choices: list[ChatMessage] | list[list[ChatMessage]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(choices[0], list):
            choices = [[c] for c in choices]  # type: ignore
        choices = cast(list[list[ChatMessage]], choices)

        prefix_encoding, choices_encoding = self._tokenize_inputs(prefix, choices)

        prefix_job = _PrefixJob(prefix_encoding, Future())
        self.job_queue.put(prefix_job)
        prefix_logprobs, prefix_cache, prefix_final_logits = await self._resolve_future(
            prefix_job.future
        )

        # expand cache and first token logits to size of batch
        prefix_cache.batch_repeat_interleave(len(choices))
        first_token_logits = prefix_final_logits.repeat_interleave(len(choices), 0)

        choices_job = _ChoicesJob(
            choices_encoding, prefix_cache, first_token_logits, Future()
        )
        self.job_queue.put(choices_job)
        choices_logprobs = await self._resolve_future(choices_job.future)

        return prefix_logprobs, choices_logprobs

    async def get_choice_logprobs(
        self,
        prefix: list[ChatMessage],
        choices: list[ChatMessage] | list[list[ChatMessage]],
    ) -> list[float]:
        _, choice_logprobs = await self.get_choice_token_logprobs(prefix, choices)
        return choice_logprobs.sum(dim=-1).tolist()

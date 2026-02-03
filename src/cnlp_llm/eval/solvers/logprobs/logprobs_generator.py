import logging
import random
from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING, TypeVar, cast, final

import anyio
import torch
from inspect_ai.model import ChatMessage, Model
from inspect_ai.model._providers.hf import HuggingFaceAPI
from inspect_ai.model._providers.mockllm import MockLLM
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
    prefix_attention_mask: torch.Tensor
    future: Future[torch.Tensor]


_T = TypeVar("_T")


def _calculate_logprobs(
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    first_token_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate logprobs of inputs based on logits.
    The attention mask is necessary to ignore padding.
    Optionally include logits for generating the first token,
    otherwise the first token will be ignored.
    """

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


class BatchedLogprobsGenerator(ABC):
    @staticmethod
    def from_model(model: Model) -> "BatchedLogprobsGenerator":
        if isinstance(model.api, MockLLM):
            return MockBatchedLogprobsGenerator()
        elif isinstance(model.api, HuggingFaceAPI):
            return HfBatchedLogprobsGenerator(model)
        else:
            raise ValueError(
                "Only HuggingFace models and mockllm/model are supported for BatchedLogprobsGenerator"
            )

    @abstractmethod
    async def get_choice_logprobs(
        self,
        prefix: list[ChatMessage],
        choices: list[ChatMessage] | list[list[ChatMessage]],
    ) -> list[float]: ...


@final
class MockBatchedLogprobsGenerator(BatchedLogprobsGenerator):
    async def get_choice_logprobs(
        self,
        prefix: list[ChatMessage],
        choices: list[ChatMessage] | list[list[ChatMessage]],
    ) -> list[float]:
        return [-random.lognormvariate(0, 0.5) for _ in choices]


@final
class HfBatchedLogprobsGenerator(BatchedLogprobsGenerator):
    def __init__(self, model: Model):
        if not isinstance(model.api, HuggingFaceAPI):
            raise ValueError(
                "Only HuggingFace models are supported for HfBatchedLogprobsGenerator"
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
        """
        Apply the chat template, tokenize the prefix, and
        tokenize the choices as a single batch with right-padding.
        """

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
        """
        Get logprobs, kv cache, and final logits for the prefix.
        Final logits are necessary to get logprobs for the first token
        when processing choices.
        """

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
        ).squeeze(0)
        final_logits = out.logits[:, -1, :]
        return logprobs, out.past_key_values, final_logits

    def _run_choices(
        self,
        choices_encoding: BatchEncoding,
        prefix_cache: Cache,
        first_token_logits: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get logprobs for choices after a prefix."""

        input_ids = choices_encoding.input_ids
        joined_attention_mask = torch.cat(
            (prefix_attention_mask, choices_encoding.attention_mask),
            dim=1,
        )

        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids,
                past_key_values=prefix_cache,
                attention_mask=joined_attention_mask,
                return_dict=True,
            )

        logprobs = _calculate_logprobs(
            input_ids,
            out.logits,
            choices_encoding.attention_mask,
            first_token_logits=first_token_logits,
        )

        return logprobs

    def _process(self):
        """Continuously process jobs from the queue."""

        while True:
            # block until there's a job
            job = self.job_queue.get()
            if isinstance(job, _PrefixJob):
                logprobs, cache, final_logits = self._run_prefix(job.encoding)
                job.future.set_result((logprobs, cache, final_logits))
            else:  # _ChoicesJob
                # TODO smarter batching
                logprobs = self._run_choices(
                    job.encoding,
                    job.prefix_cache,
                    job.first_token_logits,
                    job.prefix_attention_mask,
                )
                job.future.set_result(logprobs)

    async def _resolve_future(self, future: Future[_T]) -> _T:
        """Asynchronously wait for a future to complete."""

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
        """Get token-level logprobs for a prefix and each continuation of that prefix.

        Args:
            prefix: A list of chat messages to prefix each choice.
            choices: A list of chat messages or lists of chat messages, one for each choice.

        Returns:
            A tuple of `(prefix_logprobs, choices_logprobs)`, where `prefix_logprobs` is a
            1-D Tensor of logprobs for each token in the prefix, and `choices_logprobs` is
            a 2-D Tensor of logprobs with shape `(num_choices, max_choice_len)`, right-padded
            with zeros for choices with shorter sequence lengths.
        """

        if not isinstance(choices[0], list):
            choices = [[c] for c in choices]  # type: ignore
        choices = cast(list[list[ChatMessage]], choices)

        prefix_encoding, choices_encoding = self._tokenize_inputs(prefix, choices)

        prefix_job = _PrefixJob(prefix_encoding, Future())
        self.job_queue.put(prefix_job)
        prefix_logprobs, prefix_cache, prefix_final_logits = await self._resolve_future(
            prefix_job.future
        )

        # expand cache, first token logits, and prefix attention mask to size of batch
        prefix_cache.batch_repeat_interleave(len(choices))
        first_token_logits = prefix_final_logits.repeat_interleave(len(choices), 0)
        prefix_attention_mask: torch.Tensor = (
            prefix_encoding.attention_mask.repeat_interleave(len(choices), 0)
        )

        choices_job = _ChoicesJob(
            choices_encoding,
            prefix_cache,
            first_token_logits,
            prefix_attention_mask,
            Future(),
        )
        self.job_queue.put(choices_job)
        choices_logprobs = await self._resolve_future(choices_job.future)

        return prefix_logprobs, choices_logprobs

    async def get_choice_logprobs(
        self,
        prefix: list[ChatMessage],
        choices: list[ChatMessage] | list[list[ChatMessage]],
    ) -> list[float]:
        """Get sequence-level logprobs for each continuation of a prefix.

        Args:
            prefix: A list of chat messages to prefix each choice.
            choices: A list of chat messages or lists of chat messages, one for each choice.

        Returns:
            A list of sequence-level logprobs, one for each choice.
        """

        _, choice_logprobs = await self.get_choice_token_logprobs(prefix, choices)
        return choice_logprobs.sum(dim=-1).tolist()

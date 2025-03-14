import asyncio
import functools
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Thread
from typing import cast

import torch
from inspect_ai.model import (
    ChatMessage,
    Model,
)
from inspect_ai.model._providers.hf import HuggingFaceAPI
from inspect_ai.tool import ToolInfo
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class _QueueItem:
    input: str
    future: asyncio.Future[float]


class BatchedLogprobsGenerator:
    def __init__(self, model: Model):
        self.inspect_model = model
        self.model_config = model.config
        if not isinstance(model.api, HuggingFaceAPI):
            raise ValueError(
                "Only the Huggingface API is supported for evaluating sequence probabilities."
            )
        self.hf_api: HuggingFaceAPI = model.api

        self.model: PreTrainedModel = cast(PreTrainedModel, self.hf_api.model)
        self.tokenizer: PreTrainedTokenizer = cast(
            PreTrainedTokenizer, self.hf_api.tokenizer
        )

        self.batch_size = (
            self.model_config.max_connections or self.hf_api.max_connections()
        )

        self.batch_queue: Queue[_QueueItem] = Queue()
        self.loop = asyncio.get_event_loop()

        self.batch_thread = Thread(target=self.get_batch_processor(), daemon=True)
        self.batch_thread.start()

    def get_batch_processor(self):
        batch_size = self.batch_size
        batch_queue = self.batch_queue
        loop = self.loop
        model = self.model
        tokenize = functools.partial(
            self.tokenizer,
            return_tensors="pt",
            padding=True,
        )
        pad_token_id = self.tokenizer(self.tokenizer.pad_token).input_ids[-1]

        def process():
            while True:
                inputs: list[_QueueItem] = []
                while True:
                    try:
                        inputs.append(batch_queue.get(timeout=2))
                        if len(inputs) == batch_size:
                            break
                    except Empty:
                        break

                if len(inputs) == 0:
                    continue

                tokenized_inputs = tokenize([input.input for input in inputs])
                input_ids = cast(torch.Tensor, tokenized_inputs["input_ids"])
                attention_mask = cast(torch.Tensor, tokenized_inputs["attention_mask"])
                input_ids = input_ids.to(model.device)
                pad_mask = input_ids != pad_token_id
                attention_mask = attention_mask.to(model.device)

                with torch.inference_mode():
                    logits = model(input_ids, attention_mask=attention_mask).logits
                    logprobs = torch.nn.functional.log_softmax(logits, -1)[:, :-1, :]
                    logprobs = logprobs.gather(-1, input_ids[:, 1:, None]).squeeze()
                    logprobs = logprobs * pad_mask[:, :-1]
                    seq_probs = logprobs.sum(dim=-1).tolist()

                for input, prob in zip(inputs, seq_probs):
                    loop.call_soon_threadsafe(input.future.set_result, prob)

        return process

    async def get_logprob(self, input: list[ChatMessage], tools: list[ToolInfo] | None):
        input_str = self.hf_api.hf_chat(input, tools or [])
        future: asyncio.Future[float] = asyncio.Future(loop=self.loop)

        self.batch_queue.put(_QueueItem(input_str, future))

        return await future

    async def compare_logprobs(
        self,
        inputs: list[ChatMessage] | list[list[ChatMessage]],
        prepend: list[ChatMessage] | None = None,
        tools: list[ToolInfo] | None = None,
    ):
        if prepend is None:
            prepend = []

        if not isinstance(inputs[0], list):
            inputs = [[input] for input in cast(list[ChatMessage], inputs)]

        inputs = cast(list[list[ChatMessage]], inputs)
        return await asyncio.gather(
            *[self.get_logprob(prepend + input, tools) for input in inputs]
        )

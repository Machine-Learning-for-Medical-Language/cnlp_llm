import click


@click.command("completions")
@click.option(
    "-m",
    "--model",
    "model_name",
    type=str,
    envvar=["CNLP_CHAT_MODEL"],
    required=True,
    help="Model used to evaluate tasks. Can also be set via INSPECT_EVAL_MODEL environment variable.",
)
@click.option(
    "--model-base-url",
    type=str,
    help="Base URL for for model API.",
)
@click.option(
    "--api-key",
    type=str,
    help="API key for model provider.",
)
@click.option(
    "--max-tokens",
    type=int,
    help="The maximum number of tokens that can be generated in the completion (default is model specific)",
)
@click.option(
    "--temperature",
    type=float,
    help="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.",
)
@click.option(
    "-M",
    multiple=True,
    type=str,
    envvar=["CNLP_CHAT_MODEL_ARGS"],
    help="One or more native model arguments (e.g. -M arg=value)",
)
def completions(
    model_name: str,
    model_base_url: str | None,
    api_key: str | None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    m: tuple[str] | None = None,
):
    from typing import cast

    import torch
    from inspect_ai._cli.util import parse_cli_args
    from inspect_ai.model import GenerateConfig, get_model
    from inspect_ai.model._providers.hf import HuggingFaceAPI
    from transformers import PreTrainedModel, PreTrainedTokenizer, TextIteratorStreamer

    config = GenerateConfig(max_tokens=max_tokens, temperature=temperature)

    model_args = parse_cli_args(m)

    inspect_model = get_model(
        model=model_name,
        base_url=model_base_url,
        api_key=api_key,
        config=config,
        **model_args,
    )

    if not isinstance(inspect_model.api, HuggingFaceAPI):
        raise ValueError(
            "Only the HuggingFace API is supported for generating completions."
        )

    hf_api: HuggingFaceAPI = inspect_model.api

    hf_model: PreTrainedModel = cast(PreTrainedModel, hf_api.model)
    tokenizer: PreTrainedTokenizer = cast(PreTrainedTokenizer, hf_api.tokenizer)
    streamer = TextIteratorStreamer(tokenizer)  # type: ignore

    def generate_completion(prompt: str):
        with torch.inference_mode():
            inputs = tokenizer(prompt, return_tensors="pt")
            hf_model.generate(**inputs, streamer=streamer)  # type: ignore
        yield from streamer

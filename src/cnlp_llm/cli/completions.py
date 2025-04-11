from threading import Thread
from typing import Any

import click
from rich import markup


@click.command("completions")
@click.option(
    "-m",
    "--model",
    "model_name",
    type=str,
    envvar=["CNLP_CHAT_MODEL"],
    required=True,
    help="Model used to evaluate tasks. Can also be set via CNLP_CHAT_MODEL environment variable.",
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

    from inspect_ai._cli.util import parse_cli_args
    from inspect_ai.model import GenerateConfig, get_model
    from inspect_ai.model._providers.hf import HuggingFaceAPI
    from transformers import PreTrainedModel, PreTrainedTokenizer, TextIteratorStreamer

    from ..console import console

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

    generation_kwargs: dict[str, Any] = {
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": True,
    }
    if config.max_tokens is not None:
        generation_kwargs["max_new_tokens"] = config.max_tokens
    if config.temperature is not None:
        generation_kwargs["temperature"] = config.temperature
    if config.top_p is not None:
        generation_kwargs["top_p"] = config.top_p
    if config.top_k is not None:
        generation_kwargs["top_k"] = config.top_k

    def generate_completion(prompt: str):
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)  # type: ignore
        inputs = tokenizer(prompt, return_tensors="pt").to(hf_model.device)
        thread = Thread(
            target=hf_model.generate,
            kwargs=dict(inputs, streamer=streamer, **generation_kwargs),
        )
        thread.start()
        yield from streamer

    commands = {
        "/help": "Show this help message.",
        "/quit": "Quit the completions interface.",
        "/max-tokens <INT>": "Set the maximum completion length.",
        "/temperature <FLOAT>": "Set the generation temperature.",
    }

    console.print(
        f"\nEntering text completion mode with {markup.escape(model_name)}.",
        style="bold",
        highlight=False,
    )
    console.print("Type '/quit' to quit, or '/help' for more options.\n")

    while True:
        prompt = console.input("> ")
        if (cmd := prompt.strip().split(" ", 1)[0].lower()) in [
            c.split(" ", 1)[0] for c in commands.keys()
        ]:
            match cmd:
                case "/help":
                    for c, d in commands.items():
                        console.print(f"'{c}': {d}")
                case "/quit":
                    return
                case "/max-tokens":
                    try:
                        new_max_toks = int(prompt.strip().split(" ", 1)[1])
                        generation_kwargs["max_new_tokens"] = new_max_toks
                        console.print(
                            f"[green italic]Maximum completion length set to {new_max_toks}."
                        )
                    except Exception:
                        console.print_exception()
                case "/temperature":
                    try:
                        new_temperature = float(prompt.strip().split(" ", 1)[1])
                        generation_kwargs["temperature"] = new_temperature
                        console.print(
                            f"[green italic]Temperature set to {new_temperature}."
                        )
                    except Exception:
                        console.print_exception()
        else:
            console.print(f"[b]{markup.escape(prompt)}[/b]", end="")
            for chunk in generate_completion(prompt):
                console.print(f"[white italic]{markup.escape(chunk)}[/]", end="")

        console.print()

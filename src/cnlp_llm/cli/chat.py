import asyncio

import click
from inspect_ai.model import ChatMessage, ChatMessageUser, get_model


@click.command("chat")
@click.option(
    "-m",
    "--model",
    "model_name",
    type=str,
    envvar="INSPECT_EVAL_MODEL",
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
    "-v",
    "--verbose",
    is_flag=True,
    help="Use verbose output.",
)
# TODO: add more arguments for GenerateConfig
def chat(
    model_name: str,
    model_base_url: str | None,
    api_key: str | None,
    verbose: bool = False,
):
    """Start an interactive chat session with a model."""

    model = get_model(
        model=model_name,
        base_url=model_base_url,
        api_key=api_key,
    )

    if verbose:
        print("=" * 30)
        print(f"Model: {model_name}")
        for k, v in model.config.model_dump().items():
            print(f"  {k}: {v}")
        print("=" * 30)

    print(f"Chatting with {model_name}. Ctrl-C to exit.")

    chat_history: list[ChatMessage] = []
    while True:
        try:
            user_input = input("\n> ")
            chat_history.append(ChatMessageUser(content=user_input))
            model_output = asyncio.run(model.generate(chat_history))
            chat_history.append(model_output.message)
            print(f"\n{model_output.completion}")
        except KeyboardInterrupt:
            print()
            break

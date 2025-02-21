import logging
import os
from datetime import datetime

import click

logger = logging.getLogger(__name__)


def _logger_init():
    log_dir = os.environ.get("CNLP_CHAT_LOG_DIR")
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), "logs", "chat")
        print(
            f"Chat log dir not configured; using '{log_dir}'. Add 'CNLP_CHAT_LOG_DIR' to your .env to configure this option."
        )
    if not os.path.exists(log_dir):
        mkdir = input(
            f"'{log_dir}' is not a valid directory. Would you like to create it? (y/[N]) "
        )
        if not mkdir.lower() == "y":
            raise FileNotFoundError(f"No such directory: '{log_dir}'")
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"{datetime.now().isoformat()}.log")
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        fmt=" %(name)s :: %(asctime)s :: %(levelname)s :: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@click.command(
    "chat",
    context_settings=dict(
        ignore_unknown_options=True,
    ),
)
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
    "--system-message",
    type=str,
    help="Override the default system message.",
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
def chat(
    model_name: str,
    model_base_url: str | None,
    api_key: str | None,
    system_message: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    m: tuple[str] | None = None,
):
    """Start an interactive chat session with a model."""
    import asyncio

    from inspect_ai._cli.util import parse_cli_args
    from inspect_ai.model import ChatMessage, ChatMessageUser, GenerateConfig, get_model
    from rich.console import Console

    console = Console()
    _logger_init()

    config = GenerateConfig(
        system_message=system_message, max_tokens=max_tokens, temperature=temperature
    )

    model_args = parse_cli_args(m)

    model = get_model(
        model=model_name,
        base_url=model_base_url,
        api_key=api_key,
        config=config,
        **model_args,
    )

    logger.info(f"Chatting with {model_name}")
    for k, v in model.config.model_dump().items():
        logger.info(f"{k}: {v}")

    console.print(f"\nChatting with {model_name}.", style="bold")
    if system_message is not None:
        console.print(f"System message: [i]'{system_message}'")
    console.print("Type '/quit' to quit, or '/help' for more options.")

    commands = {
        "/help": "Show this help message.",
        "/quit": "Stop chatting.",
        "/reset": "Clear the chat history.",
        "/regenerate": "Regenerate the last response.",
        "/undo": "Remove the last two messages (user and assistant) from the context.",
    }

    def generate_with_spinner(messages: list[ChatMessage]):
        with console.status("[i]Generating...", spinner="dots"):
            return asyncio.run(model.generate(messages))

    chat_history: list[ChatMessage] = []
    while True:
        user_input = console.input("\n> ")
        logger.info(f"USER :: {user_input}")
        if (cmd := user_input.strip().lower()) in commands.keys():
            match cmd:
                case "/help":
                    for c, d in commands.items():
                        console.print(f"'{c}': {d}")
                    continue
                case "/quit":
                    return
                case "/reset":
                    console.print(
                        f"Cleared {len(chat_history)} messages.", style="green italic"
                    )
                    logger.info(
                        f"Cleared {len(chat_history)} messages from chat history."
                    )
                    chat_history = []
                    continue
                case "/regenerate":
                    if len(chat_history) == 0:
                        console.print(
                            "History is empty, cannot regenerate.", style="red italic"
                        )
                        continue
                    chat_history.pop()
                    logger.info("Removed last message from chat history.")
                case "/undo":
                    if len(chat_history) == 0:
                        console.print(
                            "History is empty, cannot undo.", style="red italic"
                        )
                    else:
                        chat_history.pop()
                        chat_history.pop()
                        logger.info("Removed last two messages from chat history.")
                        continue
        else:
            chat_history.append(ChatMessageUser(content=user_input))
            console.print()

        model_output = generate_with_spinner(chat_history)
        logger.info(f"ASSISTANT :: {model_output.completion}")
        chat_history.append(model_output.message)
        console.print(model_output.completion)

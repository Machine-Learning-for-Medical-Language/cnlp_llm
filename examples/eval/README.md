# Evaluation Examples

> [!TIP]
Many inference parameters can be automatically loaded from a `.env` file in your working directory. See the [example `.env` file](../../.env.example) for an example with links to the Inspect documentation on relevant environment variables.
>
> To create your own `.env` file based on the example, run `cp .env.example .env` from the base directory, and modify the result as needed.

## Security Guide

Copied from [here](https://inspect.ai-safety-institute.org.uk/examples.html#sec-security-guide).

Run with `python examples/security_guide.py` or `inspect eval examples/security_guide.py`

Make sure your current working directory has a [`.env` file](../../.env.example) with `INSPECT_EVAL_MODEL` set to whichever model you want to use for evaluation.

Alternatively, if you run using `inspect eval` you can pass in a model identifier with the `--model` flag.

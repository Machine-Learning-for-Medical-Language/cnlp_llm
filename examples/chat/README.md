# Chat Examples

You can chat with an instruction-tuned model via `cnlp_llm chat`.

## Configuration

### Models

You can choose which model to chat with by passing the `--model` or `-m` flag to `cnlp_llm chat` with a valid [model identifier](https://inspect.ai-safety-institute.org.uk/models.html#using-models).

If the `--model` flag is not set, it will default to the model identifier given by the environment variable `CNLP_CHAT_MODEL`.

#### Example model identifiers

| Provider | Example | Notes |
| --- | --- | --- |
| Huggingface Hub | `hf/TinyLlama/TinyLlama-1.1B-Chat-v1.0` | |
| Huggingface Local Model | `hf/local` | Specify local model path with `-M model_path=./path/to/model` |
| OpenAI | `openai/gpt-3.5-turbo` | Set API key with `OPENAI_API_KEY` environment variable |
| Ollama | `ollama/llama3.1` | Set Ollama host address with `OLLAMA_BASE_URL` environment variable; e.g. `http://localhost:11434/v1` |

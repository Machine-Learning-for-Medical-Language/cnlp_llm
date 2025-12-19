# Inference Examples

## Pirate Assistant ([pirate.py](pirate.py))

This example shows how to start a FastAPI application to serve your Task using `cnlp_llm serve`.

Here's the full source code of the example with explanations:

```python
from inspect_ai import Task
from inspect_ai import task
from inspect_ai.solver import generate, system_message

# Create your task like normal. Task arguments can be passed with `-T`, just like during evaluation.
@task
def pirate_assistant(pirate_name: str = "Tensorbeard") -> Task:
    return Task(
        solver=[
            # Custom system message based on task argument
            system_message(
                f"You are an AI assistant named {pirate_name} that always speaks like a pirate."
            ),
            # Generate a response
            generate(),
        ],
        # No dataset or scorer necessary, because we're not doing evaluation
    )
```

To start the server for the pirate example, run `cnlp_llm serve examples/infer/pirate.py` (or equivalently `cnlp_llm serve examples/infer/pirate.py@pirate_assistant`). Now you can send POST requests to [localhost:8000/evaluate](http://localhost:8000/evaluate) and get a response from the task.

```bash
curl -X POST -H 'Content-Type: application/json' -d '["Hello, who are you?"]' http://localhost:8000/evaluate
```

You can also point your web browser at [localhost:8000](http://localhost:8000) for a simple web interface to send strings of text to your task.

You can pass task arguments with `-T`, for example: `cnlp_llm serve examples/infer/pirate.py -T pirate_name='Jack Sparrow'`

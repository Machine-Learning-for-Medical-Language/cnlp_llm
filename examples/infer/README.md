# Inference Examples

## Pirate Assistant ([pirate.py](pirate.py))

This example shows how to create a simple `Pipeline` for inference. A `Pipeline` object bundles a name, a `Plan` or list of `Solver` objects, and an optional postprocessing step.

A `Pipeline` can then be invoked on a `Dataset`, a list of `Sample` objects, or a list of strings. By default, this will return a list of `TaskState` objects, one for each input value. If a postprocessing function is specified, then it will be used to map each `TaskState` to whatever output you like.

In the special case where the postprocessing function returns a string, you can easily start a FastAPI application to serve your pipeline with the `@servable` decorator.

To start the server for the pirate example, run `cnlp_llm serve examples/infer/pirate.py` (or equivalently `cnlp_llm serve examples/infer/pirate.py@get_pipeline`). Now you can send POST requests to [localhost:8000](http://localhost:8000) and get a response from the pipeline.

```bash
curl -X POST -H 'Content-Type: application/json' -d '["Hello, who are you?"]' http://localhost:8000
```

You can also point your web browser at [localhost:8000](http://localhost:8000) for a simple web interface to send strings of text to your pipeline.

# cnlp_llm

LLMs for Clinical NLP

## Introduction

The goal of this library is to consolidate, extend, and provide helpful wrappers around tools for training and evaluating decoder LLMs for clinical NLP.

## Table of Contents

- [cnlp\_llm](#cnlp_llm)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Development Tools](#development-tools)
  - [Training](#training)
    - [Pretraining (not yet implemented)](#pretraining-not-yet-implemented)
    - [Fine-tuning](#fine-tuning)
  - [Chat, Evaluation, and Serving](#chat-evaluation-and-serving)
    - [Simple Chat REPL](#simple-chat-repl)
    - [Evaluation](#evaluation)
    - [Hosting an Inference Server](#hosting-an-inference-server)

## Installation

Optionally create a conda environment (currently only Python 3.10 is officially supported):

```sh
conda create -n cnlp_llm python=3.10
conda activate cnlp_llm
```

Clone this repo and install using pip:

```sh
git clone https://github.com/Machine-Learning-for-Medical-Language/cnlp_llm.git
cd cnlp_llm
pip install -e ".[dev]"
```

### Development Tools

A few useful development tools are provided via the `Makefile`.

To install pre-commit hooks, run `make hooks`.

To lint, format, and type check your code, run `make check`.

To run tests with pytest, run `make test`.

## Training

### Pretraining (not yet implemented)

...

### Fine-tuning

We use [TRL](https://github.com/huggingface/trl/) for fine-tuning. Refer to the [examples](examples/finetune) or run `cnlp_llm finetune` for more details.

## Chat, Evaluation, and Serving

### Simple Chat REPL

To start an interactive chat session with a model, run `cnlp_llm chat`. See the [examples](examples/chat) for more info.

### Evaluation

For evaluation, we use [Inspect](https://inspect.ai-safety-institute.org.uk/). For more details, see the [examples](examples/eval) or the [Inspect documentation](https://inspect.ai-safety-institute.org.uk/workflow.html#eval-basics).

### Hosting an Inference Server

You can serve inference of an inspect `Task` on your local network with `cnlp_llm serve`. See the [examples](examples/serve) for more info.

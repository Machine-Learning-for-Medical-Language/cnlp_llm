# cnlp_llm

LLMs for Clinical NLP

## Introduction

The goal of this library is to consolidate, extend, and provide helpful wrappers around tools for training and evaluating decoder LLMs for clinical NLP.

>[!WARNING]
Training/fine-tuning are not yet implemented! Stay tuned.

## Table of Contents

- [cnlp\_llm](#cnlp_llm)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Training](#training)
    - [Pretraining](#pretraining)
    - [Fine-tuning](#fine-tuning)
  - [Inference - Chat and Evaluation](#inference---chat-and-evaluation)
    - [Simple Chat REPL](#simple-chat-repl)
    - [Evaluation](#evaluation)

## Installation

Clone this repo and install using pip (Python 3.10):

```sh
git clone https://github.com/ianbulovic/cnlp_llm
cd cnlp_llm
pip install -e .
```

## Training

### Pretraining

Coming soon!

### Fine-tuning

Coming soon!

## Inference - Chat and Evaluation

> [!TIP]
Many inference parameters can be automatically loaded from a `.env` file in your working directory. See the [example `.env` file](.env.example) for an example with links to the Inspect documentation on relevant environment variables.

### Simple Chat REPL

To start an interactive chat session with a model, run `cnlp_llm chat`. This will use the model specified by the environment variable `INSPECT_EVAL_MODEL`, or you can specify a [model name](https://inspect.ai-safety-institute.org.uk/models.html#using-models) via the `--model` flag.

### Evaluation

For evaluation, see the [Inspect documentation](https://inspect.ai-safety-institute.org.uk/workflow.html#eval-basics).
More details and examples are coming soon!

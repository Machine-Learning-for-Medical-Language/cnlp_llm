# cnlp_llm

LLMs for Clinical NLP

## Introduction

The goal of this library is to consolidate, extend, and provide helpful wrappers around tools for training and evaluating decoder LLMs for clinical NLP.

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

Optionally create a conda environment:

```sh
conda create -n cnlp_llm python=3.10
conda activate cnlp_llm
```

Clone this repo and install using pip:

```sh
git clone https://github.com/ianbulovic/cnlp_llm
cd cnlp_llm
pip install -e .
```

## Training

### Pretraining

Not yet implemented.

### Fine-tuning

We use [TRL](https://github.com/huggingface/trl/) for fine-tuning. Run `cnlp_llm finetune` for more details.

## Inference - Chat and Evaluation

> [!TIP]
Many inference parameters can be automatically loaded from a `.env` file in your working directory. See the [example `.env` file](.env.example) for an example with links to the Inspect documentation on relevant environment variables.

### Simple Chat REPL

To start an interactive chat session with a model, run `cnlp_llm chat`. This will use the model specified by the environment variable `INSPECT_EVAL_MODEL`, or you can specify a [model name](https://inspect.ai-safety-institute.org.uk/models.html#using-models) via the `--model` flag.

### Evaluation

For evaluation, cnlp_llm provides an alias to the Inspect CLI via `cnlp_llm evaluation`.
More details and examples are coming soon! In the meantime, see the [Inspect documentation](https://inspect.ai-safety-institute.org.uk/workflow.html#eval-basics).

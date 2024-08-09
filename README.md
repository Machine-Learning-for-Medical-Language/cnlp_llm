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
git clone https://github.com/Machine-Learning-for-Medical-Language/cnlp_llm.git
cd cnlp_llm
pip install -e .
```

## Training

### Pretraining

Not yet implemented.

### Fine-tuning

We use [TRL](https://github.com/huggingface/trl/) for fine-tuning. Refer to the [examples](examples/finetune) or run `cnlp_llm finetune` for more details.

## Inference - Chat and Evaluation

### Simple Chat REPL

To start an interactive chat session with a model, run `cnlp_llm chat`. See the [examples](examples/chat) for more info.

### Evaluation

For evaluation, we use [Inspect](https://inspect.ai-safety-institute.org.uk/). For more details, see the [examples](examples/eval) or the [Inspect documentation](https://inspect.ai-safety-institute.org.uk/workflow.html#eval-basics).

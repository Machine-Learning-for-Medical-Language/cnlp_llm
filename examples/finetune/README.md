# Finetuning Examples

## SFT with PEFT

The [`tinyllama_shakespeare_config.yaml`](tinyllama_shakespeare_config.yaml) file is a SFT config file for fine-tuing TinyLlama on the tiny_shakespeare dataset using PEFT.

Run the finetuning with `cnlp_llm finetune sft --config examples/finetune/tinyllama_shakespeare_config.yaml`.

## Continued pretraining with PEFT
1. Create a data directory with train.csv, dev.csv, test.csv. Each file should have a column labeled ```text``` and the entries are just sentences from your dataset that you want to pretrain on.
2. Create a .yaml file following the tinyllama_toy_cpt_config.yaml example. Replace the model with the model you want to build on (probably should be a base model). Replace the dataset_name with the path to your dataset. Replace the output_dir variable with the path where you want the adapter saved.
3. Run ```cnlp_llm finetune sft --config <path/to/your/yaml>``` pointing to the config file you created in step 2.
4. Merge the file with the chat/instruct version of your model with ```examples/finetune/merge_lora.py```
5. Run ```cnlp_llm chat -m hf/local -M model_path="<merged model directory"``` to experiment with model. That path can be used as a model path for any inspect/cnlp_llm command.

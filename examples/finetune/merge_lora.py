import sys


def main(args):
    if len(args) < 3:
        sys.stderr.write(
            "3 required arguments: <model> <lora dir> <merged (output) dir>\n"
        )
        sys.stderr.write(
            "3 required arguments: <model> <lora dir> <merged (output) dir>\n"
        )
        sys.exit(-1)

    import torch
    from peft.peft_model import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

    # base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    # base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    base_model_name = args[0]
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16
    )
    # lora_adapter_path="output"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16
    )
    # lora_adapter_path="output"
    lora_adapter_path = args[1]
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    merge_model: PreTrainedModel = model.merge_and_unload()  # type: ignore
    output_dir = args[2]
    merge_model.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])

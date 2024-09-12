from inspect_ai.log import EvalSample


def extract_final_output(sample: EvalSample):
    return sample.output.completion

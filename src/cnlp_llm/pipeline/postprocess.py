from inspect_ai.log import EvalSample


def extract_final_answer(sample: EvalSample):
    return sample.messages[-1].content

import sys

from inspect_ai.dataset import Sample

from cnlp_llm.cli import init_dotenv
from cnlp_llm.eval import cnlp_dataset
from cnlp_llm.eval.tournament import ComparisonPrompt, Tournament
from cnlp_llm.eval.tournament.scheduler import RandomScheduler

if __name__ == "__main__":
    init_dotenv()

    cola_dataset = cnlp_dataset(
        sys.argv[1],
        "acceptable",
    )

    def sample_to_binary(sample: Sample):
        return sample.target == "Yes"

    comparison_prompt = ComparisonPrompt(
        template='You are an expert linguist deciding whether sentences are grammatically acceptable or not. Your task is to take in a pair of sentences and decide which is more acceptable. The output format should be {"choice": <Sentence>, "reasoning": <your reasoning>}, where <Sentence> should be the more positive or less negative review, either "Sentence 1" or "Sentence 2". Here are the two sentences.\nSentence 1: {text1}\n\nSentence 2: {text2}',
        p1_indicator="Sentence 1",
        p2_indicator="Sentence 2",
    )

    tournament = Tournament(
        model="ollama/llama3.2:1b",
        dataset=cola_dataset,
        sample_to_binary=sample_to_binary,
        comparison_prompt=comparison_prompt,
        max_players=None,
    )

    tournament.run(rounds=10, scheduler=RandomScheduler())

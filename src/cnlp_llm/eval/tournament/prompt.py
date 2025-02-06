from dataclasses import dataclass
from typing import Literal

from inspect_ai.dataset import Sample


@dataclass
class ComparisonPrompt:
    template: str
    p1_indicator: str
    p2_indicator: str

    def get_prompt(self, sample_1: Sample, sample_2: Sample):
        return self.template.replace("{text1}", str(sample_1.input)).replace(
            "{text2}", str(sample_2.input)
        )

    def extract_winner(self, judgement: str) -> Literal["p1", "p2"] | None:
        if self.p1_indicator in judgement:
            return "p1"
        elif self.p2_indicator in judgement:
            return "p2"
        else:
            return None

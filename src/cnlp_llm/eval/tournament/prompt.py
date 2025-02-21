import json
from dataclasses import dataclass
from typing import Literal

from inspect_ai.dataset import Sample


@dataclass
class ComparisonPrompt:
    template: str
    p1_indicator: str
    p2_indicator: str
    extraction_strategy: Literal["first", "last"] = "first"

    @classmethod
    def from_file(cls, filepath: str):
        with open(filepath) as f:
            file_json = json.load(f)
        return ComparisonPrompt(**file_json)

    def get_prompt(self, sample_1: Sample, sample_2: Sample):
        return self.template.replace("{text1}", str(sample_1.input)).replace(
            "{text2}", str(sample_2.input)
        )

    def extract_winner(self, judgement: str) -> Literal["p1", "p2"] | None:
        if self.extraction_strategy == "first":
            p1_idx = judgement.find(self.p1_indicator)
            p2_idx = judgement.find(self.p2_indicator)

            if p1_idx != p2_idx:
                if p2_idx == -1 or -1 < p1_idx < p2_idx:
                    return "p1"
                else:
                    return "p2"
            else:
                return None
        else:
            p1_idx = judgement.rfind(self.p1_indicator)
            p2_idx = judgement.rfind(self.p2_indicator)

            if p1_idx != p2_idx:
                if p1_idx > p2_idx:
                    return "p1"
                else:
                    return "p2"
            else:
                return None

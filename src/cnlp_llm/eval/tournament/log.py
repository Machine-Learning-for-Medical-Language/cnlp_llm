import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from types import TracebackType
from typing import Any

from inspect_ai.model import ModelOutput

from .prompt import ComparisonPrompt


def _clean_nans(obj):
    if isinstance(obj, dict):
        return {k: _clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_nans(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return "nan"
    return obj


@dataclass(frozen=True)
class MatchResult:
    pid1: str
    pid2: str
    winner: str | None
    p1_elo_before: float
    p1_elo_after: float
    p2_elo_before: float
    p2_elo_after: float
    rematches: int
    model_input: str
    model_output: ModelOutput | None

    def to_dict(self, omit_pids: bool = False):
        d = {
            "winner": self.winner,
            "p1_elo_before": self.p1_elo_before,
            "p1_elo_after": self.p1_elo_after,
            "p2_elo_before": self.p2_elo_before,
            "p2_elo_after": self.p2_elo_after,
            "rematches": self.rematches,
            "model_input": self.model_input,
            "model_output": self.model_output.model_dump()
            if self.model_output
            else None,
        }
        if not omit_pids:
            d["pid1"] = self.pid1
            d["pid2"] = self.pid2
        return d


@dataclass
class Round:
    start_time: datetime
    results: dict[tuple[str, str], MatchResult | None]
    finish_time: datetime | None = None
    standings: dict[str, float] | None = None
    metrics: dict[str, Any] | None = None
    _usage: tuple[int, int] | None = None

    @property
    def n_completed(self):
        return len([m for m in self.results.values() if m is not None])

    @property
    def n_matches(self):
        return len(self.results)

    @property
    def finished(self):
        return self.finish_time is not None

    @property
    def duration(self):
        if self.finish_time is not None:
            return self.finish_time - self.start_time
        return None

    @property
    def usage(self):
        if self._usage is not None:
            return self._usage

        in_, out = 0, 0

        for result in self.results.values():
            if result and result.model_output and result.model_output.usage:
                in_ += result.model_output.usage.input_tokens
                out += result.model_output.usage.output_tokens

        if self.finished:
            self._usage = in_, out
        return in_, out

    def to_dict(self):
        return {
            "start_time": self.start_time.isoformat(),
            "finish_time": self.finish_time.isoformat() if self.finish_time else None,
            "standings": self.standings,
            "metrics": _clean_nans(self.metrics),
            "results": {
                ", ".join(pair): result.to_dict(omit_pids=True) if result else None
                for pair, result in self.results.items()
            },
        }


@dataclass
class TournamentLog:
    log_dir: str
    tournament_id: str
    model: str
    dataset_path: str
    samples: int
    n_rounds: int
    scheduler: str
    prompt: ComparisonPrompt
    start_standings: dict[str, float]
    start_metrics: dict[str, Any]
    rounds: list[Round] = field(default_factory=list)
    start_time: datetime | None = None
    finish_time: datetime | None = None
    error: dict[str, Any] | None = None
    log_file: str | None = None

    def log_match(
        self,
        result: MatchResult,
    ):
        self.rounds[-1].results[result.pid1, result.pid2] = result

    def log_round_start(self, matchups: list[tuple[str, str]]):
        self.rounds.append(
            Round(start_time=datetime.now(), results={tup: None for tup in matchups})
        )

    def log_round_end(
        self,
        standings: dict[str, float] | None,
        metrics: dict[str, Any] | None,
    ):
        self.rounds[-1].standings = standings
        self.rounds[-1].metrics = metrics
        if standings is not None:
            self.current_standings = standings
        if metrics is not None:
            self.current_metrics = metrics
        self.rounds[-1].finish_time = datetime.now()
        self.write_to_file()

    @property
    def current_round(self):
        if len(self.rounds) == 0:
            return 0, None
        return len(self.rounds), self.rounds[-1]

    def model_usage(self):
        in_, out = 0, 0
        for round in self.rounds:
            i, o = round.usage
            in_ += i
            out += o
        return in_, out

    def __enter__(self):
        self.start_time = datetime.now()
        self.current_standings = self.start_standings
        self.current_metrics = self.start_metrics

        def clean(s: str):
            return s.replace("_", "-").replace("/", "-").replace(":", "-")

        logger_id = f"{clean(self.start_time.isoformat(timespec='seconds'))}_tournament_{clean(self.tournament_id)}"
        self.log_file = os.path.join(self.log_dir, f"{logger_id}.json")
        self.write_to_file()
        return self

    def __exit__(
        self,
        ex_type: type[BaseException] | None,
        ex_val: BaseException | None,
        ex_tb: TracebackType | None,
    ):
        self.end_time = datetime.now()
        if ex_type or ex_val:
            self.error = dict(
                type=ex_type.__name__ if ex_type else None,
                value=str(ex_val) if ex_val else None,
            )
        self.write_to_file()

    def to_dict(self):
        in_, out = self.model_usage()
        return {
            "tournament_id": self.tournament_id,
            "model": self.model,
            "dataset_path": self.dataset_path,
            "n_samples": self.samples,
            "n_rounds": self.n_rounds,
            "scheduler": self.scheduler,
            "prompt": self.prompt.to_dict(),
            "start_standings": self.start_standings,
            "start_metrics": _clean_nans(self.start_metrics),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "finish_time": self.finish_time.isoformat() if self.finish_time else None,
            "rounds": [round.to_dict() for round in self.rounds],
            "error": self.error,
            "usage": {"input_tokens": in_, "output_tokens": out},
        }

    def write_to_file(self):
        if self.log_file is None:
            raise RuntimeError("cannot write log before tournament starts")

        with open(self.log_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

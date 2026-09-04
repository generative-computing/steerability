"""Scorers assign a scalar score to one or more candidate prompts."""
from aisteer360.algorithms.input_control.common.scorers.base import BaseScorer
from aisteer360.algorithms.input_control.common.scorers.task_evaluation import TaskEvaluationScorer

__all__ = ["BaseScorer", "TaskEvaluationScorer"]

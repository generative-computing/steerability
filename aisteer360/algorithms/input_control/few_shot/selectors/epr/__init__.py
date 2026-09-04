"""EPR, Efficient Prompt Retrieval (Rubin, Herzig, Berant 2021).

Reference:

  - "Learning To Retrieve Prompts for In-Context Learning"
    Ohad Rubin, Jonathan Herzig, Jonathan Berant
    [https://arxiv.org/abs/2112.08633](https://arxiv.org/abs/2112.08633)
"""
from aisteer360.algorithms.input_control.few_shot.selectors.epr.selector import EPRSelector

__all__ = ["EPRSelector"]

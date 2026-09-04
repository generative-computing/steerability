from aisteer360.algorithms.structural_control.wrappers.trl.ppotrainer.args import PPOArgs
from aisteer360.algorithms.structural_control.wrappers.trl.ppotrainer.control import PPO

STEERING_METHOD = {
    "category": "structural_control",
    "name": "ppo",
    "control": PPO,
    "args": PPOArgs,
}

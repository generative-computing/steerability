from .args import SystemPromptArgs
from .control import SystemPrompt

STEERING_METHOD = {
    "category": "input_control",
    "name": "system_prompt",
    "control": SystemPrompt,
    "args": SystemPromptArgs,
}

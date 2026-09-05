"""
The TRL wrapper implements a variety of methods from Hugging Face's [TRL library](https://huggingface.co/docs/trl/index).

The current functionality spans the following methods:

- **SFT (Supervised Fine-Tuning)**: Standard supervised learning to fine-tune language models on demonstration data
- **DPO (Direct Preference Optimization)**: Trains models directly on preference data without requiring a separate reward model
- **APO (Anchored Preference Optimization)**: A variant of DPO that uses an anchor model to improve training stability and performance
- **PPO (Proximal Policy Optimization)**: Reinforcement learning against a sequence-classification reward model (and value model)
- **GRPO (Group-Relative Policy Optimization)**: Critic-free reinforcement learning that takes a callable reward function (no reward model and no value model); used by PRewrite for its metric-in-the-loop reward

For documentation information, please refer to the [TRL page](https://huggingface.co/docs/trl/index).

"""

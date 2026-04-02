# My Role in Agentic RL

Code is the easy part. It always was. You just couldn't tell before because it used to
take so long.

## What I Do That No Coding Tool Can

1. **Choose what to build**: "Let's train tool use via RL, not SFT"
2. **Design the learning signal**: Pure outcome reward, no tool-use bonus
3. **Decide what the model should learn**: loss_mask=1 for model tokens, 0 for tool output
4. **Set the experimental space**: max_turns, temperature, n_samples, reward_scaling
5. **Run experiments and interpret results**: "15% accuracy advantage over pure GRPO"
6. **Identify what's broken or missing**: "We need async tool calls for throughput"
7. **Make architectural bets**: "Same workflow for train and eval, always"

## The Reframe

Faster implementation means I can **try more ideas per week**, which makes my judgment
and experimental intuition *more* valuable, not less.

My role is to be the person who looks at a math model that gets 45% accuracy and thinks:
*"What if I gave it a Python interpreter and let RL figure out when to use it?"* — and
then knows how to verify whether that worked.

## Remember

- The commented-out `tool_using = 0.01` reward bonus in `train_tir.py` — someone tried
  it, decided against it. That decision is research. The code is trivial either way.
- The `loss_mask=0` on tool outputs — the entire intellectual core of TIR in one line.
- The TODO list in the README — identifying *what matters next* is the real work.

Every design choice in this codebase embeds a decision that no coding tool would have
made on its own.
Header="""This task is to extract the final predicted label from a given rationale text, using a predefined list of candidate labels."""

Instruction="""Instruction:
- You are given a list of possible labels and a rationale text that explains a prediction.
- Identify the single most appropriate label from the label list that best matches the final conclusion in the rationale.
- Use only the given label list.
- If multiple labels apply, output all labels, each on a new line.
- Do not include explanations, reasoning steps, or any additional text."""

FewShot="""Example:
"""

DataContainer="""Rationale:
{rationale}

Possible Labels:
{labels}"""

def get_prompt(rationale, labels, fewshot=False):
    prompt = Header + '\n' + Instruction
    if fewshot:
        prompt = prompt + '\n' + fewshot
    user_prompt = DataContainer.format(rationale=rationale, labels=labels)
    return prompt, user_prompt
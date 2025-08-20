Header="""You are an expert assistant that combines machine learning predictions with domain knowledge. 
Given a model prediction, your task is to provide relevant background knowledge, reasoning, and contextual insights that help interpret and enrich the prediction."""

Instruction="""Instruction:
- Provide domain-specific knowledge related to the prediction.
- Keep the reasoning clear and informative for decision-making.
- Do not simply restate the prediction. Instead, add useful knowledge from your training and reasoning.
- Do not generate anything other messages like greeting or asking something. Only generate knowledge."""

Short="-Generate the knowledge briefly and concisely, including only the essentials."

DataContainer_u="""User input:
{user_input}
"""

DataContainer_p="""Prediction:
{prediction}
"""

def get_prompt(user_input, prediction, mode, short, fewshot=False):
    sys_prompt = Header + '\n' + Instruction
    if short:
        sys_prompt += '\n' + Short
    if fewshot:
        sys_prompt += '\n' + fewshot
    if mode:
        user_prompt = DataContainer_u.format(user_input=user_input) + '\n' + DataContainer_p.format(prediction=prediction)
    else:
        user_prompt = DataContainer_p.format(prediction=prediction)
    return sys_prompt, user_prompt

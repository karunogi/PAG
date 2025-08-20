Header="""Given the user’s input, the predictive model’s prediction, and relevant background knowledge, provide a concise conclusion or interpretation. 
Briefly explain the reasoning behind this conclusion, identifying key input factors, relevant contextual information, or notable patterns that support the result. 
"""

Instruction="""Instructions:
- Do not include any disclaimers, warnings, or mentions of AI generation.
- Integrate information from the user’s input, the predictive model’s prediction, and your own background knowledge to form a coherent rationale."""

FewShot="Example:"

DataContainer="""User input:
{user_input}

Prediction:
{prediction}

Knowledge:
{knowledge}
"""

def get_prompt(user_input, prediction, knowledge, fewshot=False):
    sys_prompt = Header + '\n' + Instruction
    if fewshot:
        sys_prompt = sys_prompt + '\n' + FewShot + '\n' + fewshot
    user_prompt = DataContainer.format(user_input=user_input, prediction=prediction, knowledge=knowledge)
    return sys_prompt, user_prompt
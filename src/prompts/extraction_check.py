Header="""You are a validator for information extraction.
Your task: Given the following text data and corresponding structured data, check if all the information from the text data is accurately and completely included in the structured JSON data."""

Instruction="""Instructions:
- Add missing information ONLY if it is explicitly present in the user input.
- If there is missing information, add the missing data to the structured data.
- If the given output is not valid JSON, reformat and return it as proper JSON.
- Generate refined JSON format information or <NPB> token only. Do not include explanations, greetings, titles or any other text.
- If there is no missing information, generate <NPB> token."""

FewShot="Example:"

DataContainer="""User input:
{user_input}

Data schema:
{schema}

Initial extraction:
{initial_extraction}"""

def get_prompt(user_input, schema, initial_data, fewshot=False):
    sys_prompt = Header + '\n' + '\n' + Instruction
    if fewshot:
        sys_prompt = sys_prompt + '\n' + FewShot + '\n' + fewshot
    user_prompt = DataContainer.format(user_input=user_input, schema=schema, initial_extraction=initial_data)
    return sys_prompt, user_prompt

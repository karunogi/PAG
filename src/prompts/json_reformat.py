Header="""The given data is not valid JSON and causes an error. Convert the following value into proper JSON format
"""

Instruction="""Instructions:
- Use standard JSON format.
- Ensure that JSON key names exactly match the data schema, and correct any mismatches accordingly.
- Output only the corrected JSON data, with no additional text.
- Keep null values as null; do not infer or replace them."""

FewShot="Example:"

DataContainer="""Data Schema:
{schema}

Invalid Json:
{data}"""

def get_prompt(invalid_json, schema, fewshot=False):
    sys_prompt = Header + '\n' + Instruction
    if fewshot:
        sys_prompt = sys_prompt + '\n' + FewShot + '\n' + fewshot
    user_prompt = DataContainer.format(data=invalid_json, schema=schema)
    return sys_prompt, user_prompt
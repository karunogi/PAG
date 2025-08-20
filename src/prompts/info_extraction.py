Header="""You are an information extraction system.
Given a user’s utterance, extract the relevant information and fill it into the following JSON schema:"""

Instruction="""Instructions:
- Only use fields defined in the schema.
- Strictly follow the data types specified in the schema for each field.
- If the information is not present in the utterance, leave the corresponding field as null.
- Do not infer or guess.
- Output only valid JSON that strictly follows the schema datatype.
- Extract JSON format information only. Do not include explanations, greetings, titles or any other text."""

FewShot="Example:"

def get_prompt(schema, fewshot=False):
    prompt = Header + '\n' + schema + '\n' + Instruction
    if fewshot:
        prompt = '\n' + prompt + '\n' + FewShot + '\n' + fewshot
    return prompt
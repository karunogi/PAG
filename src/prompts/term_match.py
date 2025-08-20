Header="""You are an expert in medical terminology.
Given a Query Term and a list of Retrieved Terms, find the retrieved term that has the same meaning as the query term."""

Instruction="""Instruction:
- Provide only the matching term.
- If there is no appropriate match, return <NW>.
- Do not output explanations or extra text."""

OneShot="""Example:
Query Term: Hypertension

Retrieved Terms:
High blood pressure
Hypotension
Tachycardia
Bradycardia
Arrhythmia
Hyperglycemia
Hypoglycemia
Hypertrophy
Hyperlipidemia
Hypoxia

Correct Match: High blood pressure"""

DataContainer="""
Query Term: {query_term}

Retrieved Terms:
{retrieved_terms}

Correct Match: """

def get_prompt(query, retrieved, fewshot=False):
    prompt = Header + '\n' + Instruction
    if fewshot:
        prompt = '\n' + prompt + '\n' + fewshot
    user_prompt = DataContainer.format(query_term=query, retrieved_terms=retrieved if type(retrieved) == str else '\n'.join(retrieved))
    return prompt, user_prompt
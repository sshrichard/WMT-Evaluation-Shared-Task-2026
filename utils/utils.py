"""Utility functions."""


import pickle
import wandb



# prompt for error severity and confidence assessment
def make_prompt_all_errors(source, translation, source_language, target_language, error_types, error_type_defs):
    prompt = f'You will be shown a source text (language: {source_language}) and a translation (language: {target_language}). \n\n'
    prompt += f"Source text: {source}\n\n"
    prompt += f"Translation: {translation}\n\n"
    prompt += (
        "TASK: For EACH of the following error categories, assess the overall severity of errors "
        "of that type (A: NOT APPLICABLE, B: NEUTRAL, C: MINOR, D: MAJOR, E: CRITICAL) across the "
        "entire translation. If multiple errors of a given type are present, base your rating on "
        "their combined impact on translation quality. Then assess your confidence for each category "
        "independently; your judgment on one category should not influence your judgment on another.\n\n"
    )
    prompt += "Error categories:\n"
    for key in error_types:
        label = key.replace("_", " ")
        prompt += f"- {key} ({label}): {error_type_defs[key]}\n"
    prompt += "\n"
    prompt += (
        'INSTRUCTION:\noutput solely a JSON object with one key per error category listed above. '
        'Each value must be an object of the form {"answer": "X", "confidence": 0.0} where "answer" '
        'is one of "A", "B", "C", "D", or "E", and "confidence" is a float between 0 and 1. '
        'DO NOT INCLUDE ANY OTHER TEXT.\n'
    )
    prompt += "Indications for confidence rating (do not be overconfident by default):\n"
    prompt += 'Better than even (0.0-0.2)\n'
    prompt += 'Likely (0.2-0.4)\n'
    prompt += 'Very good chance (0.4-0.6)\n'
    prompt += 'Highly likely (0.6-0.8)\n'
    prompt += 'Almost certain (0.8-1.0)\n\n'
    prompt += 'Your answer: '
    return prompt


# prompt for language assessment
def make_prompt_language_detection(translation, target_language):
    prompt = f'You will be shown a text and you must assess whether the language is {target_language}. \n\n'
    prompt += f"Text: {translation}\n\n"
    prompt += ('INSTRUCTION:\noutput solely a JSON object of the form {"answer": true} or {"answer": false}, '
               'where the boolean reflects whether the text is written in the target language. '
               'If you are unsure, answer True by default.\n')
    prompt += 'Your answer (JSON): '
    return prompt


def save(object, file):
    with open(f'{wandb.run.dir}/{file}', 'wb') as f:
        pickle.dump(object, f)
    wandb.save(f'{wandb.run.dir}/{file}')
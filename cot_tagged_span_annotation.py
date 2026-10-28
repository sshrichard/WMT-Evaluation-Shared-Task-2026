import argparse
import pandas as pd
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
import torch
from transformers import DataCollatorForLanguageModeling
from typing import List
import json
import os
import re

# Pydantic models for structured output
class ErrorAnnotation(BaseModel):
    severity: str
    category: str

class ErrorResponse(BaseModel):
    annotated_translation: str
    errors: List[ErrorAnnotation]
    omission: str

few_shots = {
    "ende": {
        "source_lang": "English",
        "source_seg": "I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.",
        "target_lang": "German",
        "target_seg": "Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.",
        "answer": '''{
        "annotated_translation": "Ich entschuldige mich dafür, wir müssen die Erlaubnis<ES></ES> einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen <ES>wäre</ES>, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit <ES>dir</ES> <ES>involvement</ES>.",
        "errors": [
        {"severity": "Major", "category": "accuracy/omission"},
        {"severity": "Minor", "category": "fluency/grammar"},
        {"severity": "Minor", "category": "fluency/register"},
        {"severity": "Major", "category": "accuracy/mistranslation"}
        ],
        "omission": "Major"  
        }'''
    },
    "encs": {
        "source_lang": "English",
        "source_seg": "Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.",
        "target_lang": "Czech",
        "target_seg": "Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemž obě partaje se snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních.",
        "answer": '''{
        "annotated_translation": "Ve Vídni se <ES>ve Vídni</ES> obnovily rozhovory o oživení jaderného paktu, přičemž obě <ES>partaje</ES> se snaží posoudit vyhlídky na úspěch po posledních výměnách v<ES></ES> jednáních.",
        "errors": [
        {"severity": "Major", "category": "accuracy/addition"},
        {"severity": "Minor", "category": "terminology/inappropriate for context"},
        {"severity": "Major", "category": "accuracy/omission"}
        ],
        "omission": "Major" 
        }'''
    },
    "zhen": {
        "source_lang": "Chinese",
        "source_seg": "大众点评乌鲁木齐家居卖场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评",
        "target_lang": "English",
        "target_seg": "Urumqi Home Furnishing Store Channel provides you with the latest business information such as the address, telephone number, business hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.",
        "answer": '''{
        "annotated_translation": "Urumqi Home Furnishing Store Channel provides you with the latest business information such as the address, telephone number, business hours, <ES>etc.,</ES> <ES>of high-speed rail,</ES> and find a decoration company, and <ES>go to the reviews</ES>.",
        "errors": [
        {"severity": "Minor", "category": "style/awkward"},
        {"severity": "Major", "category": "accuracy/addition"},
        {"severity": "Major", "category": "accuracy/mistranslation"}
        ],
        "omission": "Null" 
        }'''
    },
    "enes": {
        "source_lang": "English",
        "source_seg": "According to the terms outlined in the agreement, the supplier shall deliver all components no later than thirty days after receiving the initial purchase order, and any delays must be communicated in writing at least five business days in advance.",
        "target_lang": "Spanish",
        "target_seg": "De acuerdo con los términos establecidos en el acuerdo, el proveedor deberá entregar todos los componentes a más tardar treinta días después de recibir la orden de compra inicial, y cualquier retraso deberá comunicarse por escrito con al menos cinco días hábiles de antelación.",
        "answer": '''{
        "annotated_translation": "De acuerdo con los términos establecidos en el acuerdo, el proveedor deberá entregar todos los componentes a más tardar treinta días después de recibir la orden de compra inicial, y cualquier retraso deberá comunicarse por escrito con al menos cinco días hábiles de antelación.",
        "errors": [
        ],
        "omission": "Null" 
        }'''
    },
}

language_mapping_wmt_24 = {
    'en': 'English',
    'en_GB': 'English',
    'en_US': 'English',
    'cs': 'Czech',
    'cs_CZ': 'Czech',
    'ja': 'Japanese',
    'ja_JP': 'Japanese',
    'is_IS': 'Icelandic',
    'de_DE': 'German',
    'uk_UA': 'Ukrainian',
    'uk': 'Ukrainian',
    'ar_EG': 'Arabic',
    'bho_IN': 'Bhojpuri',
    'et_EE': 'Estonian',
    'ko_KR': 'Korean',
    'ru_RU': 'Russian',
    'ru': 'Russian',
    'zh_CN': 'Chinese',
    'zh': 'Chinese',
    'it_IT': 'Italian',
    'sr_Cyrl_RS': 'Serbian',
    'mas_KE': 'Maasai',
    'es': 'Spanish',
    'es_ES': 'Spanish',
    'hi': 'Hindi',
    'hi_IN': 'Hindi'
}

language_mapping = {
    "eng_Latn": "English",
    "arz_Arab": "Arabic, Egyptian",
    "bel_Cyrl": "Belarusian",
    "zho_Hans": "Chinese, Simplified",
    "zho_Hant_TW": "Chinese, Traditional Taiwan",
    "ces_Latn": "Czech",
    "hye_Armn": "Eastern Armenian",
    "ekk_Latn": "Estonian",
    "deu_Latn": "German",
    "isl_Latn": "Icelandic",
    "ind_Latn": "Indonesian",
    "jpn_Jpan": "Japanese",
    "kaz_Cyrl": "Kazakh",
    "kor_Hang": "Korean",
    "lld_Latn": "Ladin, Val Badia, Italy",
    "lij_Latn": "Ligurian, Italy",
    "sme_Latn": "Northern Sámi",
    "rus_Cyrl": "Russian",
    "tha_Thai": "Thai",
    "ukr_Cyrl": "Ukrainian",
    "vie_Latn": "Vietnamese"
}

def build_prompt(src_line: str, src_lang: str, tgt_line: str, tgt_lang: str):
    system_prompt = """
    You are a careful and balanced annotator for machine translation quality. Your task is to identify translation errors with appropriate confidence.

    ## EVALUATION GUIDELINES:
    - Be thorough but precise: Only mark errors when you are confident they are incorrect
    - Consider context and domain: Some variations may be acceptable depending on context
    - Distinguish between errors and acceptable alternatives: Multiple valid translations may exist
    - Focus on clear, objective errors rather than subjective preferences
    - Verify each potential error against the source text before marking
    - When in doubt, err on the side of not marking an error

    ## Error Categories:
    - Accuracy: addition, mistranslation, omission, untranslated text
    - Fluency: character encoding, grammar, inconsistency, punctuation, register, spelling
    - Style: awkward phrasing
    - Terminology: inappropriate for context, inconsistent use
    - Other: non-translation, other issues
    
    ## Severity Classification:
    - Major: Errors that impact meaning or usability but do not render the text unusable
    - Minor: Errors that do not impact meaning or usability

    ## Omission  Classification:
    - Null: No Errors detected in the annotated text
    - Major: Errors that appears in the error category as accuracy error like mistranslation, omission, untranslated text
    - Minor: Errors that do not impact meaning or usability
    
    ## CRITICAL OUTPUT REQUIREMENTS:
    - Mark errors only when you have clear evidence they are incorrect
    - Consider whether alternative translations could be equally valid
    - Apply strict standards: better to miss a minor error than create a false positive
    - Wrap error spans in the translation with tags: <ES></ES>
    - For omissions, place empty tags <vN></vN> where the missing text should have been
    - NO comments, explanations, or additional text
    - **Mark only the minimal substring that contains the clear error; do not include extra context**
    """.strip()

    prompts = [{"role": "system", "content": system_prompt}]

    shot_tpl = "**Source ({source_lang}):** {source_seg}\n**Translation ({target_lang}):** {target_seg}"
    for shot in few_shots.values():
        prompts.append({"role": "user",
                        "content": shot_tpl.format(**shot)})
        prompts.append({"role": "assistant",
                        "content": shot["answer"].strip()})

    query = {
        "source_lang": language_mapping.get(src_lang, src_lang),
        "source_seg": src_line,
        "target_lang": language_mapping.get(tgt_lang, tgt_lang),
        "target_seg": tgt_line
    }
    user_input = (
        "**Source ({source_lang}):** {source_seg}\n"
        "**Translation ({target_lang}):** {target_seg}"
    ).format(**query)

    prompts.append({"role": "user", "content": user_input})
    return prompts

def process_item(idx, row, model, tokenizer):
    result_row = row.to_dict()
    try:
        src = row['source_segment']
        tgt = row['hypothesis_segment']
        ref = row.get('reference_segment', None)
        src_lang = row['source_lang']
        tgt_lang = row['target_lang']
        translation_error = build_prompt(src, src_lang, tgt, tgt_lang)
        
        text = tokenizer.apply_chat_template(
            translation_error,
            tokenize=False,
            add_generation_prompt=True,
            response_format=ErrorResponse,
            enable_thinking=False
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    except Exception as e:
        print(f"[{idx}] Error processing item: {e}")
        result_row["answer"] = {"error": str(e)}

    return content
        
def offset_process(marker_text):
    pattern_s = r"<ES>"
    pattern_e = r"</ES>"

    marker_s = []
    string = ""
    for match in re.finditer(pattern_s, marker_text):                                                                                              
        marker_s.append(match.span())
    marker_e = []
    for match in re.finditer(pattern_e, marker_text):
        marker_e.append(match.span())

    spans= []
    for item_s, item_e in zip(marker_s, marker_e):
        string = string + "<ES>"
        off_s = item_s[1] - len(string)
        string = string + "</ES>"
        off_e = item_e[1] - len(string)
        spans.append((off_s, off_e))

    return spans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Input JSON file path")
    parser.add_argument("--end", type=int, help="End index (exclusive). If not provided, process all sentences")
    parser.add_argument("--source_lang", help="Source language to filter (e.g., 'en'). If not provided, process all source languages")
    parser.add_argument("--target_lang", help="Target language to filter (e.g., 'de'). If not provided, process all target languages")
    args = parser.parse_args()

    # JSON Output File
    file_path = "path_to_directory"
    base_fn = os.path.splitext(os.path.basename(args.input_json))
    file_name = file_path + base_fn[0] + '.jsonl'  
    file_out = open(file_name, 'a', encoding='utf-8')

    # Read JSON Line from File
    with open(args.input_json, 'r') as file:
        data = [json.loads(line) for line in file]
        new_data = []
        for i, item in enumerate(data):
            if i > 230:
                new_item.append(item)
        data = new_data
    
    # Load Model and Tokenizer 
    model_name = "Qwen/Qwen3-32B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,                                                                                                                                                       
        device_map="auto"
    )
    total_complete = 0
    for idx, item in enumerate(data):
        print(item["item_id"])
        src_lang = item["item_id"].split("_###_")[1]
        tgt_lang = item["item_id"].split("_###_")[2]
        src = item["src"]
        
        all_systems = {}
        for sys, tgt in item["hyps"].items():
            translation_error = build_prompt(src, src_lang, tgt, tgt_lang)

            text = tokenizer.apply_chat_template(
                translation_error,
                tokenize=False,
                add_generation_prompt=True,
                response_format=ErrorResponse,
                enable_thinking=False
            )

            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1024 
            )
            
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
                
            segment_error = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            try:
                segment_error = json.loads(segment_error, strict=False)
                if len(segment_error["errors"]) > 0:
                    spans = offset_process(segment_error["annotated_translation"])
                    all_err = []
                    for idx, span in enumerate(spans):
                        offset_dict = {
                            "start": span[0],
                            "end": span[1],
                            "severity": segment_error["errors"][idx]["severity"],
                            "category": segment_error["errors"][idx]["category"]
                        }
                        all_err.append(offset_dict)
                        
                    span_dict = {"errors": all_err, "omission": "Null"}
                    all_systems.update({sys: span_dict})
                else:
                    empty_dict = {"errors": [], "omission": "Null"}  
                    all_systems.update({sys: empty_dict})
            except Exception as e:
                print(f"Unexpected error reading '{segment_error}': {e}")
                empty_dict = {"errors": [], "omission": "Null"}
                all_systems.update({sys: empty_dict})

        error_dict = {
            "item_id": item["item_id"],
            "task1_pred": all_systems
        }
        
        if isinstance(error_dict, dict):
            json.dump(error_dict, file_out, ensure_ascii=False)
            file_out.write('\n')
            total_complete += 1

    print("Completed")

if __name__ == "__main__":
    main()

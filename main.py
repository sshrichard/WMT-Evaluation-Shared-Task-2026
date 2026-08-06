#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import langcodes
import json
import itertools
import random
import asyncio
import fasttext
import pickle
from huggingface_hub import hf_hub_download
from scipy import stats
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from openai import AsyncOpenAI, RateLimitError, APIConnectionError
from utils import utils
import joblib
import time
from iso639 import Language

start_time= time.time()

_glotlid = fasttext.load_model(hf_hub_download("cis-lmu/glotlid", "model.bin"))
GLOTLID_THRESHOLD = 0.5

label_list = [l.split("__label__")[1].split("_")[0] for l in _glotlid.labels]

random.seed(42)
np.random.seed(42)


client = AsyncOpenAI(
    base_url="http://localhost:8001/v1",
    api_key="ThisIsJustAl0calm0del",
)


# we prompt OpenAI API by batch
BATCH_SIZE = 250


# From https://themqm.org/the-mqm-typology/
#----------------------------------------------------------------------#
mqm_errors_typology = {
    "terminology": (
        "Errors arising when a term does not conform to normative subject field or "
        "organizational terminology standards or when a term in the target content is not "
        "the correct, normative equivalent of the corresponding term in the source content."),

    "accuracy": (
        "Errors occurring when the target content does not accurately correspond to the "
        "propositional content of the source text because of distortion, omission, or "
        "addition to the message."),

    "linguistic_conventions": (
        "Errors related to the linguistic well-formedness of the text, including problems "
        "with grammaticality, idiomaticity, and mechanical correctness."),

    "style": (
        "Errors occurring in a text that are grammatically acceptable but are inappropriate "
        "because they deviate from organizational style guides or exhibit inappropriate "
        "language style."),

    "locale_conventions": (
        "Errors occurring when the translation product violates locale-specific content or "
        "formatting requirements for data elements."),

    "audience_appropriateness": (
        "Errors where content inappropriately uses a culture-specific reference that will not "
        "be understandable to the intended audience."),

    "design_and_markup": (
        "Errors related to the physical design or presentation of a translation product, "
        "including character, paragraph, and UI element formatting and markup, integration "
        "of text with graphical elements, and overall page or window layout."),
}

error_types = list(mqm_errors_typology.keys())
num_error_types = len(error_types)
#----------------------------------------------------------------------#



# Severity scale
#----------------------------------------------------------------------#
# A = not applicable
# B = neutral
# C = minor
# D = major
# E = critical
PENALTY_MAP = {"B": 0, "C": 1, "D": 5, "E": 25} # we follow MQM framework (0->1->5->25)
#----------------------------------------------------------------------#



# All information is contained in one long vector
dim_per_category = 4 # [NA, severity, confidence, format_error]
VECTOR_LENGTH = num_error_types * dim_per_category + 1  # +1 for language-mismatch assessment



# Parsing output JSON
#----------------------------------------------------------------------#
# if we cannot extract the severity of an error type (unexpected value, wrong format, ...)
# , we still retrieve the values of the other error types
def get_ai_response_all(model_response_text, error_types):
    if not model_response_text:
        print("NULL CONTENT — check if reasoning field has the JSON in it", flush=True)
        return None, True
    try:
        parsed = json.loads(model_response_text)
    except json.JSONDecodeError:
        return None, True
    results = {}
    has_error = False
    for key in error_types:
        entry = parsed.get(key) if isinstance(parsed, dict) else None
        if (not isinstance(entry, dict) or set(entry.keys()) != {"answer", "confidence"} or entry.get("answer") not in ["A", "B", "C", "D", "E"]):
                has_error = True
                continue
        try:
            confidence = float(entry["confidence"])
        except (ValueError, TypeError):
            has_error = True
            continue
        if not (0.0 <= confidence <= 1.0):
            has_error = True
            continue
        results[key] = {"answer": entry["answer"], "confidence": confidence}
    return results, has_error
#----------------------------------------------------------------------#




# using glotlid model for language assessment
#----------------------------------------------------------------------#
def macro(code):
    try: return Language.from_part3(code).macrolanguage or code
    except Exception: return code

from collections import defaultdict
accept_map = defaultdict(set)
for g in set(label_list):
    accept_map[g].add(g)
    accept_map[macro(g)].add(g)

def accept_set(target_lang):
    got = accept_map.get(target_lang)
    if not got:
        return target_lang
    return got

def check_language(translation, target_language):
    labels, probs = _glotlid.predict(translation.replace("\n", " "), k=1)
    lang = labels[0].replace("__label__", "").split("_")[0]
    if probs[0] < GLOTLID_THRESHOLD:
        return True
    return lang in  accept_set(target_language)
#----------------------------------------------------------------------#




# We force the structure/format of the model's output:
# A JSON in which each key corresponds to an error type, and the value is of the following format:
# {"answer": "X", "confidence": 0.0}
#----------------------------------------------------------------------#
def build_error_schema(error_types):
    per_category_schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["answer", "confidence"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "mqm_error_assessment",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {key: per_category_schema for key in error_types},
                "required": error_types,
                "additionalProperties": False,
            },
        },
    }

ERROR_SCHEMA = build_error_schema(error_types)
#----------------------------------------------------------------------#




# calling the model
#----------------------------------------------------------------------#
async def check_error_severity_and_confidence(source, translation, source_language, target_language, temperature): # async: we send multiple prompts by batch
    delay = 0.1
    while True:
        try:
            severity_assessment = await client.chat.completions.create(
                model="qwen3.6-27b",
                messages=[
                    {"role": "system",
                        "content": "You are a helpful assistant that assesses the severity of translation errors."},
                    {"role": "user",
                        "content": utils.make_prompt_all_errors(source, translation, source_language, target_language, mqm_errors_typology.keys(), mqm_errors_typology)}],
                response_format=ERROR_SCHEMA,
                seed=42,
                temperature=temperature,
                max_tokens=4096,
            ) # Default and recommended temperature for Qwen/Qwen3.6-27B is 1.0 and Tower-Babel/Babel-83B is 0.0

            return severity_assessment.choices[0].message.content
        except (RateLimitError, APIConnectionError): # if there are connection issues
                                                    # or whether we have submitted too many requests in a short amount of time
                                                    #  or whatever
            await asyncio.sleep(delay)
            delay = min(delay * 2, 2.0)
#----------------------------------------------------------------------#







#----------------------------------------------------------------------#
# MAIN
#----------------------------------------------------------------------#
async def main(source, translation, source_language, target_language, temperature):
    """
    vector (see returns) contains [is_NA, severity_penalty, confidence, is_format_error] for all error types
    """

    #---------------------#
    # language assessment #
    #---------------------#
    is_correct_language = check_language(translation, langcodes.Language.get(target_language).to_alpha3())

    language_mismatch = 0.0 if is_correct_language else 1.0

    if not is_correct_language:
        vector = np.zeros(VECTOR_LENGTH)
        vector[-1] = language_mismatch  # still 1.0 here
        return vector, bool(language_mismatch), None # True = there is a language mismatch problem



    #---------------------#
    # severity assessment #
    #---------------------#

    model_response_text = await check_error_severity_and_confidence(source, translation, source_language, target_language, temperature)
    results, has_error = get_ai_response_all(model_response_text, error_types)

    vector = np.zeros(VECTOR_LENGTH)
    for i, key in enumerate(error_types):
        base = i * dim_per_category

        if results is None or key not in results:
            vector[base + 3] = 1.0
            continue

        entry = results[key]
        answer = entry["answer"]
        confidence = entry["confidence"]

        if answer == "A":
            vector[base + 0] = 1.0
        else:
            vector[base + 1] = PENALTY_MAP[answer]

        vector[base + 2] = confidence

    vector[-1] = language_mismatch


    return vector, bool(language_mismatch), results
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#







#---------------------#
      # training #
#---------------------#
"""
temperature = 1.0 # Default and recommended temperature for Qwen/Qwen3.6-27B is 1.0 and Tower-Babel/Babel-83B is 0.0

# retrieve data
#----------------------------------------------------------------------#
def parse_doc_id(doc_id):
    lang_part, domain, document, segment_id = doc_id.split("_#_")
    source_language_full, target_language_full = lang_part.split("-", 1)
    return source_language_full.split("_")[0], target_language_full.split("_")[0]
#----------------------------------------------------------------------#



#Loading data
#----------------------------------------------------------------------#
# we perform training using wmt25-genmt-humeval.jsonl (see https://github.com/wmt-conference/wmt25-general-mt/blob/main/README.md)
with open("wmt25-genmt-humeval.jsonl", "r", encoding="utf-8") as f:
    all_lines = f.readlines()
translations, source_sentences, human_scores, references = [], [], [], []

source_languages, target_languages = [], []


# number of considered sources
NUM_TRAIN =1000# 1000000


# we draw sources and save indices
sampled_indices = random.sample(range(len(all_lines)), min(NUM_TRAIN, len(all_lines)))
sampled_lines = [all_lines[i] for i in sampled_indices]

with open("sampled_indices.json", "w") as f:
    json.dump(sampled_indices, f)



# We retrieve data
for line in sampled_lines:
    record = json.loads(line)
    
    # translation
    target_text = record["tgt_text"]

    # source
    source_text = record["src_text"]
    
    # check for references
    ref_keys = [k for k in target_text.keys() if k.lower().startswith("ref")]
    reference = target_text[ref_keys[0]] if ref_keys else None


    # scores (two human annotations per proposed translation)
    scores_pair = record["scores"]

    # source and translation languages
    source_language, target_language = parse_doc_id(record["doc_id"])

    system_keys = list(target_text.keys())
    for system in system_keys:
        # if there is no human annotation, we discard the corresponding translation
        if system not in scores_pair:
            continue
        score_list = [a["score"] for a in scores_pair[system]]
        if not score_list:
            continue

        translation = target_text[system]
        translations.append(translation)
        source_sentences.append(source_text)
        references.append(reference) # for each translation, we may have a reference that we can use when applying baseline methods like COMET

        score_list = [a["score"] for a in scores_pair[system]]
        human_scores.append(sum(score_list) / len(score_list)) # we take the average over the pair of annotations
        source_languages.append(source_language)
        target_languages.append(target_language)

human_scores = np.array(human_scores)
#----------------------------------------------------------------------#



# data generation and preparation
#----------------------------------------------------------------------#
# We stack the data vector associated with each processed translation, resulting in a big data matrix (see np.vstack(vectors))
async def build_feature_matrix():
    vectors = []
    language_mismatches = []
    raw_responses = []
    for start in range(0, len(translations), BATCH_SIZE):
        idx = range(start, min(start + BATCH_SIZE, len(translations)))
        print(f"batch {idx.start}-{idx.stop - 1}")
        # calling main
        results = await asyncio.gather(*[main(source_sentences[i], translations[i], source_languages[i], target_languages[i], temperature) for i in idx])
        for vector, mismatch, raw in results:
            vectors.append(vector)
            language_mismatches.append(mismatch)
            raw_responses.append(raw)
    return np.vstack(vectors), language_mismatches, raw_responses

y = human_scores
X, language_mismatches, raw_responses = asyncio.run(build_feature_matrix())
#----------------------------------------------------------------------#



# nonlinear regression (training)
#----------------------------------------------------------------------#
confidence_index = 2
confidence_indices = [i * dim_per_category + confidence_index for i in range(num_error_types)]
X_no_confidences = np.delete(X, confidence_indices, axis=1)

# network architecture chosen from https://sklearner.com/sklearn-mlpclassifier-hidden_layer_sizes-parameter/
def make_model():
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            max_iter=5000,
            random_state=42))
#----------------------------------------------------------------------#

end_time = time.time()




# SAVING
#----------------------------------------------------------------------#
training_record = {
    "sampled_indices": sampled_indices,          # which lines of the jsonl were used
    "source_sentences": source_sentences,
    "translations": translations,
    "references": references,
    "Duration": end_time - start_time,
    "source_languages": source_languages,
    "target_languages": target_languages,
    "human_scores": human_scores,                # y
    "raw_responses": raw_responses,              
    "language_mismatches": language_mismatches,
    "X": X,                                       # full feature matrix
    "X_no_confidences": X_no_confidences,
    "confidence_indices": confidence_indices,
    "error_types": error_types,
    "penalty_map": PENALTY_MAP,
    "dim_per_category": dim_per_category,
    "vector_length": VECTOR_LENGTH,
}


with open("training_record.pkl", "wb") as f:
    pickle.dump(training_record, f)

model_full = make_model()
model_full.fit(X, y)
joblib.dump(model_full, "model_full.joblib")


model_no_conf = make_model()
model_no_conf.fit(X_no_confidences, y)
joblib.dump(model_no_conf, "model_no_conf.joblib")
"""
#----------------------------------------------------------------------#







#---------------------#
# testing #
#---------------------#

temperature = 1.0 # Default and recommended temperature for Qwen/Qwen3.6-27B is 1.0 and Tower-Babel/Babel-83B is 0.0

# retrieve data
#----------------------------------------------------------------------#
def parse_doc_id(doc_id):
    set_id, src_lang, tgt_lang, domain, document_id, segment_id = doc_id.split("_###_")
    return src_lang.split("_")[0], tgt_lang.split("_")[0]
translations, source_sentences, references = [], [], []
system_names_per_doc = []
source_languages, target_languages = [], []
doc_id_list = []
#----------------------------------------------------------------------#



#Loading data
#----------------------------------------------------------------------#
#with open("mteval-test26.jsonl", "r", encoding="utf-8") as f:
#    all_lines = f.readlines()
import gzip
with gzip.open("mteval-test26.jsonl.gz", "rt", encoding="utf-8") as f:
    all_lines = f.readlines()


NUM = len(all_lines) # number of tested sources

if NUM < len(all_lines):
    lines_indices = random.sample(range(len(all_lines)), NUM)
    lines = [all_lines[i] for i in lines_indices]
    #with open("sampled_indices.json", "w") as f:
    #    json.dump(lines_indices, f)
else:
    lines = all_lines

number_of_translations_per_doc = np.zeros(len(lines))


# We retrieve data
i = 0
for line in lines:
    record = json.loads(line)

    # translation
    #target_text = record["tgt_text"]
    target_text = record["hyps"]

    # source
    #source_text = record["src_text"]
    source_text = record["src"]

    # check for references
    ref = record.get("ref")
    reference = ref["text"] if ref else None

    # source and translation language
    source_language, target_language = parse_doc_id(record["item_id"])
    doc_id_list.append(record["item_id"])

    system_keys = list(target_text.keys())
    system_names_per_doc.append(system_keys)

    for system in system_keys:
        number_of_translations_per_doc[i] += 1

        translation = target_text[system]
        translations.append(translation)

        references.append(reference) # for each translation, we may have a reference that we can use when applying baseline methods like COMET
        source_sentences.append(source_text)

        source_languages.append(source_language)
        target_languages.append(target_language)
    i += 1

#----------------------------------------------------------------------#



# data generation and preparation
#----------------------------------------------------------------------#
# We stack the data vector associated with each processed translation, resulting in a big data matrix (see np.vstack(vectors))
async def build_feature_matrix():
    vectors = []
    language_mismatches = []
    raw_responses = []
    for start in range(0, len(translations), BATCH_SIZE):
        idx = range(start, min(start + BATCH_SIZE, len(translations)))
        print(f"batch {idx.start}-{idx.stop - 1}")
        # calling main
        results = await asyncio.gather(*[main(source_sentences[i], translations[i], source_languages[i], target_languages[i], temperature) for i in idx])
        for vector, mismatch, raw in results:
            vectors.append(vector)
            language_mismatches.append(mismatch)
            raw_responses.append(raw)
    return np.vstack(vectors), language_mismatches, raw_responses
#----------------------------------------------------------------------#

end_time = time.time()




# SAVING
#----------------------------------------------------------------------#
#y = human_scores
X, language_mismatches, raw_responses = asyncio.run(build_feature_matrix())
confidence_index = 2
confidence_indices = [i * dim_per_category + confidence_index for i in range(num_error_types)]
X_no_confidences = np.delete(X, confidence_indices, axis=1)


model_full = joblib.load("model_full_QWEN.joblib")
predictions_test_full = model_full.predict(X)

model_no_conf = joblib.load("model_no_conf_QWEN.joblib")
predictions_test_nc = model_no_conf.predict(X_no_confidences)



testing_record = {
    "doc_ids": doc_id_list,          # which lines of the jsonl were used
    "number_of_translations_per_doc": number_of_translations_per_doc,
    "source_sentences": source_sentences,
    "translations": translations,
    "references": references,
    "source_languages": source_languages,
    "target_languages": target_languages,
    "Duration": end_time - start_time,
    "raw_responses": raw_responses,
    "language_mismatches": language_mismatches,
    "X": X,                                       # full feature matrix
    "X_no_confidences": X_no_confidences,
    "predictions_test_full": predictions_test_full,
    "predictions_test_nc": predictions_test_nc,
    "confidence_indices": confidence_indices,
    "error_types": error_types,
    "penalty_map": PENALTY_MAP,
    "dim_per_category": dim_per_category,
    "vector_length": VECTOR_LENGTH,
    "temperature": temperature,
}


with open("testing_record_GPT_OSS_REF.pkl", "wb") as f:
    pickle.dump(testing_record, f)


# well formatted file for submission
with open("task2_predictions_full.jsonl", "w", encoding="utf-8") as f:
    start = 0
    for item_id, n, system_names in zip(doc_id_list, number_of_translations_per_doc, system_names_per_doc):
        n = int(n)
        doc_predictions = (predictions_test_full[start: start + n]).tolist()
        start += n
        record = {
            "item_id": item_id,
            "task2_pred": dict(zip(system_names, doc_predictions))
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

with open("task2_predictions_no_conf.jsonl", "w", encoding="utf-8") as f:
    start = 0
    for item_id, n, system_names in zip(doc_id_list, number_of_translations_per_doc, system_names_per_doc):
        n = int(n)
        doc_predictions = (predictions_test_nc[start: start + n]).tolist()
        start += n
        record = {
            "item_id": item_id,
            "task2_pred": dict(zip(system_names, doc_predictions))
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

#----------------------------------------------------------------------#


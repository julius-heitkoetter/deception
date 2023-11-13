import typing as T
import json
import os
from pathlib import Path
from datetime import datetime

from datasets import load_dataset
import requests

ETHICS_CATEGORIES = ['commonsense', 'deontology', 'justice', 'utilitarianism', 'virtue']
MMLU_CATEGORIES = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

def load_hf_dataset(dataset_path: str, category: str = '', split: str =''):
    '''
    Loads dataset from huggingface

    Inputs:
        dataset_path: path to dataset on hugging face (e.g. 'cais/mmlu') 
        category (optional) (e.g. 'all', 'bio') 
    '''
    if category != '':
        return load_dataset(dataset_path, category)
    return load_dataset(dataset_path, split=split)

def mmlu_to_json(dataset_name, dataset_path, category, split):
    json_dict_correct = {'metadata': {'dataset': f'{dataset_name}-{category}', 'correct': True}, 'data':[]}
    json_dict_incorrect = {'metadata': {'dataset': f'{dataset_name}-{category}', 'correct': False}, 'data':[]}
    for question_data in load_dataset(dataset_path, category, split=split):
        for answer_i, answer in enumerate(question_data['choices']):
            is_correct = answer_i == question_data['answer'] # True if answer choice is correct, else False
            new_question = [{
                'question': question_data['question'],
                'answer': answer,
                'correct': is_correct
            }]
            if is_correct:
                json_dict_correct['data'].append(new_question)
            else:
                json_dict_incorrect['data'].append(new_question)
    return json.dumps(json_dict_correct, indent=4), json.dumps(json_dict_incorrect, indent=4) 

def commonsense_data_extractor(question_data, json_dict_correct, json_dict_incorrect, correct_label = 0):
    label = question_data['label']
    if label == correct_label:
        correct_answer = "ethical"
        incorrect_answer = "unethical"
    else:
        correct_answer = "unethical"
        incorrect_answer = "ethical"
    
    json_dict_correct['data'].append({
        'question': question_data['input'],
        'answer': correct_answer,
        'correct': True
    })
    json_dict_incorrect['data'].append({
        'question': question_data['input'],
        'answer': incorrect_answer,
        'correct': False
    })
    return json_dict_correct, json_dict_incorrect

def deontology_data_extractor(question_data, json_dict_correct, json_dict_incorrect):
    # I AM UNSURE HOW TO SPLIT THIS ONE
    
    label = question_data['label']
    if label == 0:
        json_dict_correct['data'].append({
            'question': question_data['scenario'],
            'answer':  question_data['excuse'],
            'correct': True
        })
    else:
        json_dict_incorrect['data'].append({
            'question': question_data['scenario'],
            'answer': question_data['excuse'],
            'correct': False
        })

def justice_data_extractor(question_data, json_dict_correct, json_dict_incorrect):
    return commonsense_data_extractor(question_data, json_dict_correct, json_dict_incorrect, correct_label = 1)

def utilitarianism_data_extractor(question_data, json_dict_correct, json_dict_incorrect):
    # I AM UNSURE HOW TO SPLIT THIS ONE UP
    # Fixed order is likely bad.
    option_A = question_data['baseline']
    option_B = question_data['less_pleasant']
    question = f"Which is more pleasant:\nA. {option_A}\nB. {option_B}"
    json_dict_correct['data'].append({
        'question': question,
        'answer': 'A',
        'correct': True
    })
    json_dict_incorrect['data'].append({
        'question': question,
        'answer': 'B',
        'correct': False
    })

def virtue_data_extractor(question_data, json_dict_correct, json_dict_incorrect):
    is_correct = question_data['label'] == 1
        json_dict_correct['data'].append({
        'question': question,
        'answer': is_correct,
        'correct': True
    })
    json_dict_incorrect['data'].append({
        'question': question,
        'answer': not is_correct,
        'correct': False
    })

ethics_data_extractor = {
    'commonsense': commonsense_data_extractor,
    'deontology': deontology_data_extractor,
    'justice': justice_data_extractor,
    'utilitarianism': utilitarianism_data_extractor, 
    'virtue': virtue_data_extractor 
}

def ethics_to_json(dataset_name, dataset_path, category, split):
    json_dict_correct = {'metadata': {'dataset': f'{dataset_name}-{category}', 'correct': True}, 'data':[]}
    json_dict_incorrect = {'metadata': {'dataset': f'{dataset_name}-{category}', 'correct': False}, 'data':[]}
    for question_data in load_dataset(dataset_path, category, split=split):
        for answer_i, answer in enumerate(question_data['choices']):
            ethics_data_extractor[category](question_data, json_dict_correct, json_dict_incorrect)
    return json.dumps(json_dict_correct, indent=4), json.dumps(json_dict_incorrect, indent=4)

#test_json_c, test_json_i = mmlu_to_json('mmlu','cais/mmlu', 'college_mathematics', 'dev')
test_json_c, test_json_i = mmlu_to_json('mmlu','cais/mmlu', 'college_mathematics', 'dev')

print(test_json_c)
print(test_json_i)


datetime_string = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

#a = load_hf_dataset('cais/mmlu', 'college_mathematics')

#print(load_dataset('cais/mmlu', 'college_mathematics')['auxiliary_train'][0])
#print(dataset_to_json)


'''
Input of this module is a dataset
Output of this module:
- QA dataset (json format)
- example name: qa-incorrect-mmlu-datetime
- json: 'metadata: {'dataset': 'MMLU-bio'} 
    - 'data': [{'question': question str, 'answer': answer, 'correct': bool}]

'''
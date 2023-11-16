import sys
import os
import platform

import typing as T
import json
from abc import ABC
from datasets import load_dataset


if platform.system() == "Windows":
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))
    from utils import save_json_locally, get_json_locally, upload_json_to_hf
else:
    from lib.utils import save_json_locally, get_json_locally, upload_json_to_hf


ETHICS_CATEGORIES = ['commonsense', 'deontology', 'justice', 'utilitarianism', 'virtue']
MMLU_CATEGORIES = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

class DataLoader(ABC):
    """
    Creates qa
    """
    
    def __init__(self, dataset_name, hf_dataset_src_path):
        self.dataset_name = dataset_name
        self.hf_dataset_src_path = hf_dataset_src_path

    def __call__(self,):
        pass

class MMLULoader(DataLoader):
    def __init__(self):
        """
        Creates an instance of a dataloader for MMLU dataset
        """
        super().__init__(
            dataset_name = 'mmlu',
            hf_dataset_src_path = 'cais/mmlu'
        )

    def __call__(self, category: str, data_split: str ='test'):
        """
        Returns json files of correct (and incorrect) responses.
        json structure: {
            'metadata': {'dataset': 'dataset_name-category', 'correct': True/False}
            'data': [{'question': 'question text goes here', 'answer': answer text goes here, 'correct': True/False}, ...]
        }

        Inputs:
            category (str) - 
            data_split (str)

        Outupts:
        - json file in the format 
        """
        assert category in MMLU_CATEGORIES
        return self.mmlu_to_json(self.dataset_name, self.hf_dataset_src_path, category, data_split)

    def mmlu_to_json(self, dataset_name, dataset_path, category, split):
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

class EthicsLoader(DataLoader):
    def __init__(self):
        super().__init__(
            dataset_name = 'ethics',
            hf_dataset_src_path = 'hendrycks/ethics'
        )

        self.ethics_data_extractors = {
            'commonsense': self.commonsense_data_extractor,
            'deontology': self.deontology_data_extractor,
            'justice': self.justice_data_extractor,
            'utilitarianism': self.utilitarianism_data_extractor, 
            'virtue': self.virtue_data_extractor 
        }

    def __call__(self, category, data_split='test'):
        # input: dataset category
        # save_locally_flag=False
        # num_samples_max
        # 
        # output: dataset

        assert category in ETHICS_CATEGORIES
        return self.ethics_to_json(self.dataset_name, self.hf_dataset_src_path, category, split=data_split)

    def ethics_to_json(self, dataset_name, dataset_path, category, split):
        json_dict_correct = {'metadata': {'dataset': f'{dataset_name}-{category}', 'correct': True}, 'data':[]}
        json_dict_incorrect = {'metadata': {'dataset': f'{dataset_name}-{category}', 'correct': False}, 'data':[]}
        for question_data in load_dataset(dataset_path, category, split=split):
            self.ethics_data_extractors[category](question_data, json_dict_correct, json_dict_incorrect)
        return json.dumps(json_dict_correct, indent=4), json.dumps(json_dict_incorrect, indent=4)

    def commonsense_data_extractor(self, question_data, json_dict_correct, json_dict_incorrect, correct_label = 0):
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

    def deontology_data_extractor(self, question_data, json_dict_correct, json_dict_incorrect):
        # I AM UNSURE HOW TO SPLIT THIS ONE
        
        label = question_data['label']
        if label == 1:
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

    def justice_data_extractor(self, question_data, json_dict_correct, json_dict_incorrect):
        return self.commonsense_data_extractor(question_data, json_dict_correct, json_dict_incorrect, correct_label = 1)

    def utilitarianism_data_extractor(self, question_data, json_dict_correct, json_dict_incorrect):
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

    def virtue_data_extractor(self, question_data, json_dict_correct, json_dict_incorrect):
        is_correct = question_data['label'] == 1
        answer_strings = (
            "The emotion matches the scenario.",
            "The emotion does not match the scenario."
        )
        json_dict_correct['data'].append({
            'question': question_data['question'],
            'answer': answer_strings[int(is_correct)],
            'correct': True
        })
        json_dict_incorrect['data'].append({
            'question': question_data['question'],
            'answer': answer_strings[int(not is_correct)],
            'correct': False
        })
import sys
import os
import platform

import copy
from datetime import datetime
import typing as T
import json
from abc import ABC
from datasets import load_dataset
import numpy as np


if platform.system() == "Windows":
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))
    from utils import save_json_locally, get_json_locally, upload_json_to_hf
else:
    from lib.utils import save_json_locally, get_json_locally, upload_json_to_hf


ETHICS_CATEGORIES = ['commonsense', 'deontology', 'justice', 'utilitarianism', 'virtue']
MMLU_CATEGORIES = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
                    'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science',
                    'college_mathematics', 'college_medicine', 'college_physics', 'computer_security',
                    'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics',
                    'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry',
                    'high_school_computer_science', 'high_school_european_history', 'high_school_geography',
                    'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics',
                    'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics',
                    'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 
                    'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 
                    'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 
                    'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 
                    'professional_psychology', 'public_relations', 'security_studies', 'sociology', 
                    'us_foreign_policy', 'virology', 'world_religions']

MMLU_DATA_SPLITS = {'auxiliary_train', 'test', 'validation', 'dev'}
ETHICS_DATA_SPLITS = {'train', 'test', 'validation'}

class DataLoader(ABC):
    """
    Creates qa
    """
    
    def __init__(self, dataset_name, hf_dataset_src_path, storage_path):
        self.dataset_name = dataset_name
        self.hf_dataset_src_path = hf_dataset_src_path
        self.json_template = {
            'metadata': 
            {
                'dataset': dataset_name,
                'category': "",
                'src_path': hf_dataset_src_path,
                'correct': True
            },
            'data': []
        }
        
        self.final_storage_path = f"./data/{dataset_name}" if not storage_path else storage_path
    
    def __call__(self):
        pass

    def new_json(self, category, correct):
        new_json_from_template = copy.deepcopy(self.json_template)
        new_json_from_template['metadata']['category'] = category
        new_json_from_template['metadata']['correct'] = correct
        return new_json_from_template
    
    def upload_file(self, data: dict, save_locally: bool = False, save_on_hf: bool = False):
        date_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        correct_incorrect = "correct" if data['metadata']['correct'] else "incorrect"
        category = data['metadata']['category']
        file_name = f"qa-{correct_incorrect}-{self.dataset_name}-{category}-{date_id}.json"
        full_storage_path = f"{self.final_storage_path}/{file_name}"

        if save_locally:
            save_json_locally(storage_path=self.final_storage_path, filename=file_name, data=data)
        if save_on_hf:
            upload_json_to_hf(
                obj=data,   # SHOULD I JSON DUMP HERE?
                path_in_repo=full_storage_path,
                repo_id="laker-julius-misha/correlated-errors",
                repo_type="dataset")
        return full_storage_path
        


class MMLULoader(DataLoader):
    def __init__(self, storage_path:str = ""):
        """
        Creates an instance of a dataloader for MMLU dataset
        """
        super().__init__(
            dataset_name = 'mmlu',
            hf_dataset_src_path = 'cais/mmlu',
            storage_path = storage_path,

        )

    def __call__(self, category: str, data_split: str ='dev', save_locally: bool = False, 
                 save_on_hf: bool = True, num_samples: T.Optional[int] = None):
        """
        Returns json files of correct (and incorrect) responses.
        json structure: {
            'metadata': {'dataset': 'dataset_name-category', 'correct': True/False}
            'data': [{'question': 'question text goes here', 'answer': answer text goes here, 'correct': True/False}, ...]
        }

        Inputs:
            category (str) the MMLU category (e.g. "astronomy", "high_school_us_history")
            data_split (str) which split from hugging face to download
                - must be in {'auxiliary_train', 'test', 'validation', 'dev'}
            num_samples (int) how many samples to use. Samples are always shuffled. All 
                              samples used if this is None

        Outupts:
            (tuple[str]) path/filenames to json files 
        """
        assert category in MMLU_CATEGORIES, f"category: {category} not in MMLU categories"
        assert data_split in MMLU_DATA_SPLITS, f"data split: {data_split} not in MMLU data splits"

        json_correct, json_incorrect = self._mmlu_to_json(category, data_split)

        if num_samples is not None:

            data_correct = np.random.shuffle(json_correct['data'])[:num_samples]
            json_correct = {'metadata': json_correct['metadata'], 'data':data_correct}

            data_incorrect = np.random.shuffle(json_incorrect['data'])[:num_samples]
            json_incorrect = {'metadata': json_incorrect['metadata'], 'data':data_incorrect}

        correct_path = self.upload_file(json_correct, save_locally=save_locally, save_on_hf=save_on_hf)
        incorrect_path = self.upload_file(json_incorrect, save_locally=save_locally, save_on_hf=save_on_hf)

        return correct_path, incorrect_path

    def _mmlu_to_json(self, category, split):
        """
        Converts desired category to json format by loading from hugging face
        Organizes into correct and incorrect responses
        """
        json_dict_correct = self.new_json(category=category, correct=True)
        json_dict_incorrect = self.new_json(category=category, correct=False)
        
        for question_data in load_dataset(self.hf_dataset_src_path, category, split=split):
            for i, answer in enumerate(question_data['choices']):
                is_correct = i == question_data['answer'] # True if answer choice is correct, else False
                new_question = [{
                    'question': question_data['question'],
                    'answer': answer,
                    'correct': is_correct
                }]
                if is_correct:
                    json_dict_correct['data'].append(new_question)
                else:
                    json_dict_incorrect['data'].append(new_question)

        return json_dict_correct, json_dict_incorrect 

class EthicsLoader(DataLoader):
    def __init__(self, storage_path:str = ""):
        super().__init__(
            dataset_name = 'ethics',
            hf_dataset_src_path = 'hendrycks/ethics',
            storage_path = storage_path,
        )

        self.ethics_data_extractors = {
            'commonsense': self.commonsense_data_extractor,
            'deontology': self.deontology_data_extractor,
            'justice': self.justice_data_extractor,
            'utilitarianism': self.utilitarianism_data_extractor, 
            'virtue': self.virtue_data_extractor 
        }

    def __call__(self, category, data_split='test', save_locally: bool = False, save_on_hf: bool = True, 
                 num_samples: T.Optional[int] = None):
        """
        Returns json files of correct (and incorrect) responses.
        json structure: {
            'metadata': {'dataset': 'dataset_name-category', 'correct': True/False}
            'data': [{'question': 'question text goes here', 'answer': answer text goes here, 'correct': True/False}, ...]
        }

        Inputs:
            category (str) the Ethics category (e.g. "astronomy", "high_school_us_history")
            data_split (str) which split from hugging face to download
                - must be in {'auxiliary_train', 'test', 'validation', 'dev'}
            num_samples (int) how many samples to use. Samples are always shuffled. All 
                              samples used if this is None

        Outupts:
            (tuple[str]) path/filenames to json files 
        """

        assert category in ETHICS_CATEGORIES
        assert data_split in ETHICS_DATA_SPLITS, f"data split: {data_split} not in Ethics data splits"

        json_correct, json_incorrect = self._ethics_to_json(category, data_split)

        if num_samples is not None:

            data_correct = np.random.shuffle(json_correct['data'])[:num_samples]
            json_correct = {'metadata': json_correct['metadata'], 'data':data_correct}

            data_incorrect = np.random.shuffle(json_incorrect['data'])[:num_samples]
            json_incorrect = {'metadata': json_incorrect['metadata'], 'data':data_incorrect}

        correct_path = self.upload_file(json_correct, save_locally=save_locally, save_on_hf=save_on_hf)
        incorrect_path = self.upload_file(json_incorrect, save_locally=save_locally, save_on_hf=save_on_hf)

        return correct_path, incorrect_path

    def _ethics_to_json(self, category, split):
        json_dict_correct = self.new_json(category=category, correct=True)
        json_dict_incorrect = self.new_json(category=category, correct=False)
        for question_data in load_dataset(self.hf_dataset_src_path, category, split=split):
            self.ethics_data_extractors[category](question_data, json_dict_correct, json_dict_incorrect)
        return json_dict_correct, json_dict_incorrect

    def generic_data_extractor(self, question_data, json_dict_correct, json_dict_incorrect, flipped_label: bool):
        correct_label = int(flipped_label)
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

    def commonsense_data_extractor(self, question_data, json_dict_correct, json_dict_incorrect):
        # TODO: Debug loading from hugging face
        return self.generic_data_extractor(self, question_data, json_dict_correct, json_dict_incorrect, correct_label = True)

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
        return self.generic_data_extractor(self, question_data, json_dict_correct, json_dict_incorrect, correct_label = False)

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
        is_correct = question_data['label'] == 1    # returns True if label is correct
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
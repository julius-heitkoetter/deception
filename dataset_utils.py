import json
import csv
import os

from config import prompt_to_get_explanation, prompt_to_check_truth

def save_pairs(storage_path, filename, pairs):

    with open(os.path.join(storage_path, filename), 'w') as f:
        json.dump(pairs, f)

def get_pairs(storage_path, filename):

    with open(os.path.join(storage_path, filename), 'r') as f:
        pairs = json.load(f)

    return pairs


def generate_question_answer_from_data(data_path, data_name):

    correct_qa_pairs = []
    incorrect_qa_pairs = []
    choice_indices = {'A':1, 'B':2, 'C':3, 'D':4}

    with open(os.path.join(data_path, data_name), 'r') as f:
        
        reader = csv.reader(f)
        
        for row in reader:
            assert len(row) == 6

            question = row[0]
            correct_choice = row[-1]
            correct_index = choice_indices.get(correct_choice, -1)

            if correct_index != -1:

                for i in range(1,5):
                    if i == correct_index:
                        correct_qa_pairs.append({"question":question, "answer":row[i][:-2]}) #TODO: why do I have to drop the last two characters?
                    else:
                        incorrect_qa_pairs.append({"question":question, "answer":row[i][:-2]})

    return correct_qa_pairs, incorrect_qa_pairs

def generate_question_answer_explanation_from_llm(llm, qa_pairs):

    qae_pairs = []

    for qa_pair in qa_pairs:

        prompt = prompt_to_get_explanation.format(
                Question=qa_pair["question"], 
                Answer=qa_pair["answer"]
        )

        explanation = llm(prompt=prompt)

        qae_pairs.append({
            "question":qa_pair["question"], 
            "answer":qa_pair["answer"], 
            "explanation":explanation
        })

    return qae_pairs

def check_truth_from_question_answer_explanation(llm, qae_pairs):

    qaet_pairs = []

    for qae_pair in qae_pairs:

        prompt = prompt_to_check_truth.format(
                Question=qae_pair["question"],
                Answer=qae_pair["answer"],
                Explanation=qae_pair["explanation"]
        )

        predicted_truth = llm(prompt=prompt)

        qaet_pairs.append({
            "question":qae_pair["question"],
            "answer":qae_pair["answer"],
            "explanation":qae_pair["explanation"],
            "predicted_truth":predicted_truth,
        })

    return qaet_pairs

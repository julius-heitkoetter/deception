import typing as T
import numpy as np
import random
import json
import os

import lib.utils as utils
from lib.chain import HumanEvaluator


def make_filename_pairs(data_dir):
    filenames = [f"{data_dir}/{filename}" for filename in os.listdir(data_dir) if not os.path.isdir(f"{data_dir}/{filename}") and filename != ".DS_STORE"]
    filenames = sorted(filenames)

    # Collect pairs of (correct_filename, incorrect_filename),
    # since we need both to compute the model's capability.
    use_correct_datasets = False
    filename_pairs = []
    for i in range(0, len(filenames), 2):
        if "mmlu_business-ethics_qaeve_gpt-3.5-turbo_gpt-3.5-turbo" in filenames[i]: #TODO: fix this, it's terrible coding!
            print("Swapping correct and incorrect filename")
            filename_pairs.append((filenames[i+1], filenames[i]))
            assert utils.get_json_locally("", filenames[i+1])["metadata"]["correct"] == True, f"Error creating correct/incorrect filename pairs: {filenames[i+1]}"
            assert utils.get_json_locally("", filenames[i])["metadata"]["correct"] == False, f"Error creating correct/incorrect filename pairs: {filenames[i]}."
            continue
        filename_pairs.append((filenames[i], filenames[i+1]))
        # verify that the datasets are incorrect and correct as assumed
        # (an error here can indicate a hidden .DS_STORE file)
        assert utils.get_json_locally("", filenames[i])["metadata"]["correct"] == True, f"Error creating correct/incorrect filename pairs: {filenames[i]}"
        assert utils.get_json_locally("", filenames[i+1])["metadata"]["correct"] == False, f"Error creating correct/incorrect filename pairs: {filenames[i+1]}."
        assert len(utils.get_json_locally("", filenames[i])["data"]) > 0, f"Error dataset {filenames[i]} has zero questions"
        assert len(utils.get_json_locally("", filenames[i+1])["data"]) > 0, f"Error: dataset {filenames[i+1]} has zero questions."

    return filename_pairs

if __name__ == "__main__":
    data_dir = "data/data-final/feb-data-deduped/llama_70b_supervisor"

    filename_pairs = make_filename_pairs(data_dir)

    # organize files by deceiver model then by category then by correct-incorrect
    qaeve = {}

    # print some information about each dataset
    for correct_filename, incorrect_filename in filename_pairs:
        _, category, _, deceiver, _, _ = utils.atoms_from_filename(correct_filename)
        _, incorrect_category, _, _, _, _ = utils.atoms_from_filename(correct_filename)
        assert category == incorrect_category
        if deceiver not in qaeve:
            qaeve[deceiver] = [(correct_filename, incorrect_filename, category)]
        else:
            qaeve[deceiver].append((correct_filename, incorrect_filename, category))
    
    # for each model, for each category, prepare answer-explanation pairs to evaluate
    prepared_samples = {}
    samples_per_model_per_category = 20  # split evenly among correct and incorrect
    for model, categories in qaeve.items():
        prepared_samples[model] = []
        for correct_filename, incorrect_filename, category in categories:
            for filename in [correct_filename, incorrect_filename]:
                data = utils.get_json_locally("", filename)["data"]
                samples = random.sample(data, samples_per_model_per_category//2)
                for sample in samples:
                    sample["category"] = category
                prepared_samples[model] += samples

    # collect human evaluations of the samples in a random order
    human = HumanEvaluator()
    collected_samples = {model: [] for model, _ in prepared_samples.items()}
    while any([len(samples) > 0 for samples in list(prepared_samples.values())]):
        # select random model, then random question to ask
        model = random.choice(list(prepared_samples.keys()))
        if len(prepared_samples[model]) == 0:
           continue
        sample_index = random.randint(0, len(prepared_samples[model])-1)
        sample = prepared_samples[model][sample_index]
        prepared_samples[model].pop(sample_index)

        # sample human
        evaluation = human(sample)
        collected_samples[model].append(evaluation)
        
        # store each intermediate sample
        with open("human_evals-2-10-2024.json", "w") as f:
            f.write(json.dumps(collected_samples))


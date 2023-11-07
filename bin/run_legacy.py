#from models import LlamaLLM
from config import llama_base_config, gpt_35_turbo_base_config
from dataset_utils import generate_question_answer_explanation_from_llm, generate_question_answer_from_data, save_pairs, get_pairs, check_truth_from_question_answer_explanation
from models import LlamaLLM, OpenAILLM

import time

DATA_PATH = "MC_data"
STORAGE_PATH = "saved_datasets"

def test_error_catching_full(generating_llm, checking_llm, dataset_name, n=3):

    print("INFO : starting error catching for ", dataset_name)

    correct_qa_pairs, incorrect_qa_pairs = generate_question_answer_from_data(DATA_PATH, dataset_name + ".csv")
    save_pairs(STORAGE_PATH, "incorrect_qa_pairs_" + dataset_name + ".json", incorrect_qa_pairs)

    start = time.time()
    incorrect_qae_pairs = generate_question_answer_explanation_from_llm(generating_llm, incorrect_qa_pairs[:n])
    save_pairs(STORAGE_PATH, "incorrect_qae_llama_pairs_" + dataset_name + ".json", incorrect_qae_pairs)
    print("Time to generate qae dataset: ", time.time() - start)

    incorrect_qae_pairs = get_pairs(STORAGE_PATH, "incorrect_qae_llama_pairs_" + dataset_name + ".json")

    start = time.time()
    incorrect_qeat_pairs = check_truth_from_question_answer_explanation(checking_llm, incorrect_qae_pairs)
    save_pairs(STORAGE_PATH, "incorrect_qaet_llama_llama_pairs_" + dataset_name + ".json", incorrect_qeat_pairs)
    print("Time to generate qaet dataset: ", time.time() - start)

def test_error_catching_evaluation(checking_llm, qae_pairs_filename, dataset_name, n=5): #TODO: get rid of dataset name

    incorrect_qae_pairs = get_pairs(STORAGE_PATH, qae_pairs_filename)[:n]

    start = time.time()
    incorrect_qeat_pairs = check_truth_from_question_answer_explanation(checking_llm, incorrect_qae_pairs)
    save_pairs(STORAGE_PATH, "incorrect_qaet_llama_" + checking_llm.name + "_pairs_" + dataset_name + ".json", incorrect_qeat_pairs)
    print("Time to generate qaet dataset: ", time.time() - start)



llama = LlamaLLM(**llama_base_config)
gpt_35 = OpenAILLM(**gpt_35_turbo_base_config)

test_error_catching_evaluation(llama, "incorrect_qae_llama_pairs_high_school_biology_test.json", "high_school_biology_test")
test_error_catching_evaluation(gpt_35, "incorrect_qae_llama_pairs_high_school_biology_test.json", "high_school_biology_test")

test_error_catching_evaluation(llama, "incorrect_qae_llama_pairs_high_school_physics_test.json", "high_school_physics_test")
test_error_catching_evaluation(gpt_35, "incorrect_qae_llama_pairs_high_school_physics_test.json", "high_school_physics_test")

test_error_catching_evaluation(llama, "incorrect_qae_llama_pairs_high_school_world_history_test.json", "high_school_world_history_test")
test_error_catching_evaluation(gpt_35, "incorrect_qae_llama_pairs_high_school_world_history_test.json", "high_school_world_history_test")

#test_error_catching_full(llama, llama, "high_school_physics_test", n=50)

#test_error_catching_full(llama, llama, "high_school_biology_test", n=50)

#test_error_catching_full(llama, llama, "high_school_world_history_test", n=50)

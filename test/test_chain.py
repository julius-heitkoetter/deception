import sys
print(sys.path)
import json
from lib.models import OpenAILLM
from config import gpt_35_turbo_base_config
from lib.chain import Supervisor, Deceiver, Evaluator, run_on_dataset

with open("test/data/qa_testsmall_2023-11-08_09_07_32.json", "r") as f:
    dataset = json.load(f)

llm = OpenAILLM(**gpt_35_turbo_base_config)
deceiver = Deceiver(llm)
supervisor = Supervisor(llm)
evaluator = Evaluator(llm)

print("Dataset at step 0:")
print(dataset)

# Step 1: create "explanation"
dataset = run_on_dataset(dataset, deceiver)
print("Dataset at step 1: create explanations")
print(dataset)

with open("test/data/qae_testsmall", "w") as f:
    json.dump(dataset, f, indent=4)

# Step 2: create "verdict"
dataset = run_on_dataset(dataset, supervisor)
print("Dataset at step 2: create verdicts")
print(dataset)

with open("test/data/qaev_testsmall", "w") as f:
    json.dump(dataset, f, indent=4)

# Step 3: create "verdict-without-explanation"
dataset = run_on_dataset(dataset, evaluator)
print("Dataset at step 0: create verdicts without explanation")
print(dataset)

with open("test/data/qaev_testsmall_complete.json", "w") as f:
    json.dump(dataset, f, indent=4)


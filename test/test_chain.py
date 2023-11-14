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
print(dataset)

exit()
# Step 2: create "verdict"
dataset = run_on_dataset(dataset, supervisor)
print(dataset)

# Step 3: create "verdict-without-explanation"
dataset = run_on_dataset(dataset, evaluator)
print(dataset)

import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))
from dataset import MMLULoader, EthicsLoader


mmlu_dataloader = MMLULoader()
ethics_dataloader = EthicsLoader()

print(mmlu_dataloader('astronomy', data_split='dev'))
print(ethics_dataloader('deontology', data_split='test'))
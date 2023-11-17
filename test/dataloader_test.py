import sys
import os
import platform

if platform.system() == "Windows":
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))
    from utils import save_json_locally, get_json_locally, upload_json_to_hf
    from dataset import MMLULoader, EthicsLoader

else:
    from lib.utils import save_json_locally, get_json_locally, upload_json_to_hf
    from lib.dataset import MMLULoader, EthicsLoader


mmlu_dataloader = MMLULoader()
ethics_dataloader = EthicsLoader()

astronomy_correct, astronomy_incorrect = mmlu_dataloader('astronomy', 
    data_split='dev', save_locally=True, save_on_hf=False)
bio_correct, bio_incorrect = mmlu_dataloader('high_school_biology', 
    data_split='dev', save_locally=True, save_on_hf=False)
us_history_correct, us_history_incorrect = mmlu_dataloader('high_school_us_history', 
    data_split='dev', save_locally=True, save_on_hf=False)

print(bio_incorrect)
#print(ethics_dataloader('commonsense', data_split='test'))
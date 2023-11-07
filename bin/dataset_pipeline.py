##Example pipeline usage:

from config import {
    llama_base_config, 
    gpt_35_turbo_base_config, 
    CoT_base_config,

    deceiver_base_config,
    supervisor_base_config,
}
from bin.modles import LlamaLLM, OpenAILLM, CoT
from bin.chain import DataLoader, Deceiver, Supervisor, Evaluator


deceiver_llm = LlamaLLM(**llama_base_config)
supervisor_llm =  OpenAILLM(**gpt_35_turbo_base_config)
supervisor_CoT_llm = CoT(supervisor_llm, **CoT_base_config)

deceiver = Deceiver(deceiver_llm, **deceiver_base_config)
supervisor = Supervisor(supervisor_CoT_llm, **supervisor_base_config)

deceiver_evaluator = Evaluator(deceiver_llm, **evaluator_base_config)
supervisor_evaluator = Evaluator(supervisor_CoT_llm, **evaluator_base_config)

loader = DataLoader(??)

dataset_names = ["LIST", "OF", "NAMES"]

#For uploading everything to huggingface
for dataset_name in dataset_names:

    qa_dataset_name = loader(dataset_name)
    qae_dataset_name = deceiver(qa_dataset_name)
    qaev_dataset_name = supervisor(qae_dataset_name)
    
    qav_deceiver_dataset_name = deceiver_evaluator(qa_dataset_name)
    qav_supervisor_dataset_name = supervisor_evaluator(qa_dataset_name)


#For uploading everything to local storage
local_storage_path = "path/to/folder"

for dataset_name in dataset_names:

    qa_dataset_name = loader(dataset_name, local_storage_path = local_storage_path)
    qae_dataset_name = deceiver(qa_dataset_name, local_storage_path = local_storage_path)
    qaev_dataset_name = supervisor(qae_dataset_name, local_storage_path = local_storage_path)
    
    qav_deceiver_dataset_name = deceiver_evaluator(qa_dataset_name, local_storage_path = local_storage_path)
    qav_supervisor_dataset_name = supervisor_evaluator(qa_dataset_name, local_storage_path = local_storage_path)

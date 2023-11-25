import typing as T
from abc import ABC
from config import (
    deceiver_base_config,
    supervisor_base_config,
    evaluator_base_config,
)
from lib.utils import (
    atoms_from_filename, 
    next_filename_in_chain,
    download_json_dataset_from_hf,
    upload_json_to_hf,
    save_json_locally,
    get_json_locally,
)
import copy
import os
from tqdm import tqdm

class DataLoader:
    """
    Creates qa
    """

    def __init__(self,):
        pass

    def __call__(self,):
        pass

class DatasetLLM(ABC):
    """
    Abstract base class for an LLM that extends a dataset, for instance, qa -> qae.
    """

    def __init__(self, llm, **kwargs):
        self.llm = llm

    def __call__(self, item: T.Dict):
        raise NotImplementedError("")
    
    def update_metadata(self, metadata: T.Dict) -> T.Dict:
        """
        Update the metadata to describe what is happening during the stage.
        """
        raise NotImplementedError("")

    def run_on_dataset(self, dataset: T.List[T.Dict]) -> T.List[T.Dict]:
        """
        Run an on an entire dataset to add the relevant key-value pair to each item in the list. Mutates dataset.
        """
        data = dataset["data"]
        updated_data = []
        for i in tqdm(range(len(data)), desc = "Running..."):
            updated_data.append(self(data[i]))

        updated_metadata = self.update_metadata(dataset["metadata"])

        dataset["data"] = updated_data
        dataset["metadata"] = updated_metadata

        return dataset

    def run_on_dataset_name(self, dataset_path: str, save_locally: bool = False, save_on_hf: bool = True,
                            repo_id: str = "laker-julius-misha/correlated-errors") -> str:
        """
        Run on an entire datset to add the relevant key-value pair to each item in the list.

        This method takes in a full filepath, fetches that file either locally or from hf, runs on the
        dataset, then writes the new file back to either local storage or to hf
        """        
        
        # get filename and path 
        filename = os.path.basename(dataset_path)
        storage_path = os.path.dirname(dataset_path)

        # check if either uploaded locally or on huggingface (but not both!)
        if not (save_locally or save_on_hf):
            raise ValueError("Either save_locally or save_on_hf flags must be set to true")
        if save_locally and save_on_hf:
            raise ValueError("Cannot get dataset from both local storage and hugging face simultaneously")        

        # get dataset
        if save_locally:
            dataset = get_json_locally(storage_path, filename)
        elif save_on_hf:
            dataset = download_json_dataset_from_hf(os.path.join(storage_path, filename), repo_id)

        # modify the dataset
        dataset = self.run_on_dataset(dataset)
        out_filename = self.get_next_filename_in_chain(filename)

        # save the dataset
        if save_locally:
            save_json_locally(storage_path, out_filename, dataset)
        elif save_on_hf:
            upload_json_to_hf(dataset, os.path.join(storage_path, out_filename), repo_id, "dataset")

        # return the path of the output dataset
        return os.path.join(storage_path, out_filename)

    def get_next_filename_in_chain(filename):
        raise NotImplementedError("")

class Deceiver(DatasetLLM):
    """
    qa -> qae
    """

    def __init__(self, llm):
        super().__init__(llm)
    
    def __call__(self, qa: T.Dict):
        assert "question" in qa
        assert "answer" in qa

        explanation_prompt = deceiver_base_config["explanation_prompt"].format(
            Question = qa["question"],
            Answer = qa["answer"],
        )

        evaluation_prompt = evaluator_base_config["prompt"].format(
            Question = qa["question"],
            Answer = qa["answer"],
        )

        qa["explanation"] = self.llm(prompt=explanation_prompt)
        qa["evaluation-deceiver"] = self.llm(prompt=evaluation_prompt)

        return qa
    
    def update_metadata(self, metadata: T.Dict) -> T.Dict:
        """
        Update the metadata to describe what is happening during the stage.

        Specifically, we want to mention the deceiver llm, the explanation
        prompt, and the deceiver evaluation prompt
        """

        # create deep copy as to not mutate inputs
        updated_metadata = copy.deepcopy(metadata)

        updated_metadata["deceiver_llm"] = self.llm.name
        updated_metadata["explanation_prompt"] = deceiver_base_config["explanation_prompt"]
        updated_metadata["deciever_evaluation_prompt"] = evaluator_base_config["prompt"]

        return updated_metadata

    def get_next_filename_in_chain(self, filename):
        """
        Deceiver class specific wrapper for the util next_filename_in_chain
        """
        return next_filename_in_chain(filename, deciever_model = self.llm.name)

class Supervisor(DatasetLLM):
    """
    qae -> qaev
    """
    
    def __init__(self, llm):
        super().__init__(llm)

    def __call__(self, qae: T.Dict):
        assert "question" in qae
        assert "answer" in qae
        assert "explanation" in qae

        prompt = supervisor_base_config["prompt"].format(
            Question = qae["question"],
            Answer = qae["answer"],
            Explanation = qae["explanation"],
        )

        qae["verdict"] = self.llm(prompt=prompt)

        return qae
    
    def update_metadata(self, metadata: T.Dict) -> T.Dict:
        """
        Update the metadata to describe what is happening during the stage.

        Specifically, we want to mention the supervisor llm and verdict prompt,

        """

        # create deep copy as to not mutate inputs
        updated_metadata = copy.deepcopy(metadata)

        updated_metadata["supervisor_llm"] = self.llm.name
        updated_metadata["verdict_prompt"] = supervisor_base_config["prompt"]

        return updated_metadata

    def get_next_filename_in_chain(self, filename):
        """
        Supervisor class specific wrapper for the util next_filename_in_chain
        """
        _, _, _, deciever_model, _, _ = atoms_from_filename(filename)
        return next_filename_in_chain(filename, deciever_model = deciever_model, supervisor_model = self.llm.name)


class Evaluator(DatasetLLM):
    """
    qa -> qav
    """

    def __init__(self, llm):
        super().__init__(llm)

    def __call__(self, qa: T.Dict):
        assert "question" in qa
        assert "answer" in qa

        prompt = evaluator_base_config["prompt"].format(
            Question = qa["question"],
            Answer = qa["answer"],
        )

        qa["evaluation-supervisor"] = self.llm(prompt=prompt)

        return qa
    
    def update_metadata(self, metadata: T.Dict) -> T.Dict:
        """
        Update the metadata to describe what is happening during the stage.

        Specifically, we want to mention the evaluator llm and evaluator prompt,

        """

        # create deep copy as to not mutate inputs
        updated_metadata = copy.deepcopy(metadata)

        updated_metadata["evaluator_llm"] = self.llm.name
        updated_metadata["evaluator_prompt"] = evaluator_base_config["prompt"]

        return updated_metadata

    def get_next_filename_in_chain(self, filename):
        """
        Evaluator class specific wrapper for the util next_filename_in_chain
        """

        _, _, _, deciever_model, supervisor_model, _ = atoms_from_filename(filename)
        return next_filename_in_chain(filename, deciever_model = deciever_model, supervisor_model = supervisor_model)

import typing as T
from abc import ABC
from config import (
    deceiver_base_config,
    supervisor_base_config,
    evaluator_base_config,
)
import copy

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
        updated_data = [self(item) for item in data]

        updated_metadata = self.update_metadata(dataset["metadata"])

        dataset["data"] = updated_data
        dataset["metadata"] = updated_metadata

        return dataset


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

        Specifically, we want to mension the deceiver llm, the explanation
        prompt, and the deceiver evaluation prompt
        """

        # create deep copy as to not mutate inputs
        updated_metadata = copy.deepcopy(metadata)

        updated_metadata["deceiver_llm"] = self.llm.name
        updated_metadata["explanation_prompt"] = deceiver_base_config["explanation_prompt"]
        updated_metadata["deciever_evalaution_prompt"] = evaluator_base_config["prompt"]

        return updated_metadata

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

        Specifically, we want to mension the supervisor llm and verdict prompt,

        """

        # create deep copy as to not mutate inputs
        updated_metadata = copy.deepcopy(metadata)

        updated_metadata["supervisor_llm"] = self.llm.name
        updated_metadata["verdict_prompt"] = supervisor_base_config["prompt"]

        return updated_metadata


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

        Specifically, we want to mension the evaluator llm and evaluator prompt,

        """

        # create deep copy as to not mutate inputs
        updated_metadata = copy.deepcopy(metadata)

        updated_metadata["evaluator_llm"] = self.llm.name
        updated_metadata["evaluator_prompt"] = evaluator_base_config["prompt"]

        return updated_metadata



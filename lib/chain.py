import typing as T
from abc import ABC
from config import (
    deceiver_base_config,
    supervisor_base_config,
    evaluator_base_config,
)

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

    def run_on_dataset(self, dataset: T.List[T.Dict]) -> T.List[T.Dict]:
        """
        Run an on an entire dataset to add the relevant key-value pair to each item in the list. Mutates dataset.
        """
        data = dataset["data"]
        updated_data = [self(item) for item in data]
        dataset["data"] = updated_data

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

        prompt = deceiver_base_config["prompt"].format(
            Question = qa["question"],
            Answer = qa["answer"],
        )

        qa["explanation"] = self.llm(prompt=prompt)

        return qa


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

        qa["verdict-without-explanation"] = self.llm(prompt=prompt)

        return qa



from abc import ABC, abstractmethod


class BasePipeline(ABC):
    """Abstract base class for main pipelines"""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def run(self) -> None:
        """run the pipeline."""
        pass
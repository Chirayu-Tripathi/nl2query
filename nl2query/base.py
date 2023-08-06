from abc import ABC, abstractmethod


class MethodNotImplementedError(Exception):
    """
    Raised when a method is not implemented.

    Args:
        Exception (Exception): MethodNotImplementedError
    """


class QueryLanguage:
    """Base class to implement multiple querying languages"""

    @abstractmethod
    def load_model(self) -> object:
        """
        Load the corresponding model for query generation

        Raises:
            MethodNotImplementedError: load_model method has not been implemented
        """
        raise MethodNotImplementedError("load_model method has not been implemented")

    @abstractmethod
    def generate_query(
        self,
        textual_query: str,
        num_beams: int,
        max_length: int,
        repetition_penalty: int,
        length_penalty: int,
        early_stopping: bool,
        top_p: int,
        top_k: int,
        num_return_sequences: int,
    ) -> str:
        """
        Execute the CodeT5 to generate the query in the corresponding language.

        Raises:
            MethodNotImplementedError: generate_query method has not been implemented
        """
        raise MethodNotImplementedError(
            "generate_query method has not been implemented"
        )

    @abstractmethod
    def preprocess(self, text) -> str:
        """
        Pre-Process the user's textual query by converting all to lowercase and inserting columns/attributes/keys in the query itself.

        Raises:
            MethodNotImplementedError: load_model method has not been implemented
        """
        raise MethodNotImplementedError("load_model method has not been implemented")

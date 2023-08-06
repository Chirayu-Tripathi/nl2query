import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .base import QueryLanguage


class MongoQuery(QueryLanguage):
    """Base QueryLanguage class extended to perform query generation for MongoDB"""

    def __init__(
        self,
        collection_keys: list,
        collection_name: str,
        path: str = "Chirayu/MQL_best_pipe",
    ):
        """Constructor for MongoQuery class"""
        self.path = path
        self.collection_keys = collection_keys
        self.collection_keys.remove("index")
        # self.keys_mapping = {'"'+key.lower()+'"' : '"'+key+'"' for key in self.collection_keys}
        self.keys_mapping = {key.lower(): key for key in self.collection_keys}
        self.collection_name = collection_name
        self._load_model()
        # self.db = db

    def _load_model(self) -> object:
        """Constructor for MongoQuery class"""
        model = AutoModelForSeq2SeqLM.from_pretrained(self.path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        return self.model, self.tokenizer

    def preprocess(self, text: str) -> str:
        """Pre-Process the user's textual query by converting all the text to lowercase and inserting all the keys of collections in the query itself."""
        text = (
            "mongo: "
            + text
            + " | "
            + self.collection_name
            + " : "
            + ", ".join(self.collection_keys)
        )
        # TO-DO: Won't work nicely for (improve the splitting) - what 'pclass' has average age of passengers greater than the age of person named 'Braund, Mr. Owen Harris'?
        upper_text = {i.lower(): i for i in text.split() if i.lower() != i}
        # print(upper_text)
        # print(text.lower())
        return text.lower(), upper_text

    def generate_query(
        self,
        textual_query: str,
        num_beams: int = 20,
        max_length: int = 256,
        repetition_penalty: int = 2.5,
        length_penalty: int = 1,
        early_stopping: bool = True,
        top_p: int = 0.95,
        top_k: int = 50,
        num_return_sequences: int = 1,
    ) -> str:
        """Execute the CodeT5 to generate the query for the MongoDB framework."""
        query, upper_text = self.preprocess(textual_query)
        input_ids = self.tokenizer.encode(
            query, return_tensors="pt", add_special_tokens=True
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = input_ids.to(device)

        generated_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
        )
        query = [
            self.tokenizer.decode(
                generated_id,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            for generated_id in generated_ids
        ][0]
        # print(query)
        # print(self.keys_mapping)
        pattern = "|".join(
            re.escape(key) for key in {**self.keys_mapping, **upper_text}.keys()
        )
        query = re.sub(
            pattern, lambda x: {**self.keys_mapping, **upper_text}[x.group()], query
        )
        return query

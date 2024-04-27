"""
This module contains classes for generating MongoDB queries using different models.

Classes:
    MongoQueryT5: Uses T5 model to generate MongoDB queries.
    MongoQueryPhi2: Uses Phi2 model to generate MongoDB queries.
    MongoQuery: Factory class to create an instance of either MongoQueryT5 or MongoQueryPhi2.

Each class has methods to load the model, preprocess the input, and generate the query.

Example:
    To create an instance of MongoQueryT5:
    >>> mq = MongoQuery('T5', collection_keys=['key1', 'key2'], collection_name='my_collection')

    To generate a query:
    >>> query = mq.generate_query('Find all documents where key1 is "value1"')

"""

import re
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from .base import QueryLanguage


class MongoQueryT5(QueryLanguage):
    """Base QueryLanguage class extended to perform query generation for MongoDB"""

    def __init__(
        self,
        collection_keys: list,
        collection_name: str,
        path: str = "Chirayu/nl2mongo",
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
        """Helper function to load the model for MongoQuery class"""
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


class MongoQueryPhi2(QueryLanguage):
    """Base QueryLanguage class extended to perform query generation for MongoDB using Phi2 model."""

    def __init__(
        self,
        path: str = "Chirayu/phi-2-mongodb",
    ):
        """Constructor for MongoQuery class"""

        # self.db_schema = db_schema
        self.adapter = path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
        # self.db = db

    def _load_model(self) -> object:
        """Helper function to load the model for MongoQuery class"""

        base_model_id = "microsoft/phi-2"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            trust_remote_code=True,
            quantization_config=bnb_config,
            revision="refs/pr/23",
            device_map={"": 0},
            torch_dtype="auto",
            flash_attn=True,
            flash_rotary=True,
            fused_dense=True,
        )

        self.model = PeftModel.from_pretrained(model, self.adapter).to(self.device)
        return self.model, self.tokenizer

    def preprocess(self, db_schema: str, text: str) -> str:
        """Pre-Process the db_schema by removing new line and extra spaces, and creates a prompt for the model."""
        db_schema = db_schema.replace("\n", " ").replace("  ", "")

        prompt_template = f"""<s> 
        Task Description:
        Your task is to create a MongoDB query that accurately fulfills the provided Instruct while strictly adhering to the given MongoDB schema. Ensure that the query solely relies on keys and columns present in the schema. Minimize the usage of lookup operations wherever feasible to enhance query efficiency.

        MongoDB Schema: 
        {db_schema}

        ### Instruct:
        {text}

        ### Output:
        """

        return prompt_template

    def generate_query(
        self,
        db_schema: str,
        textual_query: str,
        max_length: int = 1024,
        no_repeat_ngram_size: int = 10,
        repetition_penalty: int = 1.02,
    ) -> str:
        """Execute the Phi2 to generate the query for the MongoDB framework."""
        prompt = self.preprocess(db_schema, textual_query)
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **model_inputs,
            max_length=max_length,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )[0]
        query = self.tokenizer.decode(output, skip_special_tokens=False)
        start_idx = query.index("Output")
        try:
            stop_idx = query.index("</s>")
        except Exception as e:
            print(e)
            stop_idx = len(query)
        return query[start_idx + 8 : stop_idx].strip()


class MongoQuery:
    """Primary class to call the appropriate model"""

    def __new__(cls, model_type, **kwargs):
        if model_type == "T5":
            return MongoQueryT5(**kwargs)
        elif model_type == "Phi2":
            return MongoQueryPhi2(**kwargs)
        else:
            raise ValueError("Invalid model_type. Expected 'T5' or 'Phi2'")

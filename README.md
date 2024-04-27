# nl2query

 > Convert natural language text inputs to Pandas, MongoDB, Kusto, and Cypher(Neo4j) queries. The models used are fine-tuned versions of CodeT5+ 220m and Phi2 model.


[![Downloads](https://static.pepy.tech/badge/nl2query)](https://pepy.tech/project/nl2query)
[![Build Status][build-image]][build-url]
[![][stars-image]][stars-url]
[![PyPI version][pypi-image]][pypi-url]
[![Support Python versions][versions-image]][versions-url]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![nl2query](images/logo.png?raw=true)

## Getting started

You can [get `nl2query` from PyPI](https://pypi.org/project/nl2query), using

```bash
python -m pip install nl2query
```


## Example usage

## 1. Pandas Query
Suppose you want to convert the textual question to pandas query, follow the code below

```py
from nl2query import PandasQuery

titanic = pd.read_csv('/path/titanic.csv')
queryfier = PandasQuery(titanic, 'titanic')

queryfier.generate_query('''list all people who paid more fare than the fare paid by 'Braund, Mr. Owen Harris' ''')
queryfier.generate_query('''find the names of passengers with age greater than 35 and containing Heath in their name''')
queryfier.generate_query('''which cabinet has average age less than 21?''') #Groupby Query

```

## 2. MongoDB Query
Suppose you want to convert the textual question to Mongo query, follow the instruction code below

### MongoDB query using CodeT5

The generate_query method takes a textual query and returns a MongoDB query. It also accepts optional parameters to control the generation process, such as num_beams, max_length, repetition_penalty, length_penalty, early_stopping, top_p, top_k, and num_return_sequences.

NOTE: GPU will be required to run Phi2 as quantization is enabled using *load_in_4bit*.

```py
from nl2query import MongoQuery
import pymongo # import if performing analysis using python client
keys = ['_id', 'index', 'passengerid', 'survived', 'Pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked'] #keys present in the collection to be queried.
queryfier = MongoQuery('T5', collection_keys = keys, collection_name = 'titanic')
queryfier.generate_query('''which pclass has the minimum average fare?''')

keys = ['_id', 'index', 'total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size'] 
queryfier = MongoQuery('T5', collection_keys = keys, collection_name = 'titanic')
queryfier.generate_query('''find the day on which combined sales was highest''')

```
In the above code the keys can be found by running the following piece `db.tips.find_one({}).keys()`

### MongoDB query using Phi2

The generate_query method takes a database schema and a textual query and returns a MongoDB query. It also accepts optional parameters to control the generation process, such as max_length, no_repeat_ngram_size, and repetition_penalty. *The Phi2 model performs better than the CodeT5+ model.*

```py
from nl2query import MongoQuery
schema = shipwreck = '''{
  "collections": [
    {
      "name": "shipwrecks",
      "indexes": [
        {
          "key": {
            "_id": 1
          }
        },
        {
          "key": {
            "feature_type": 1
          }
        },
        {
          "key": {
            "chart": 1
          }
        },
        {
          "key": {
            "latdec": 1,
            "londec": 1
          }
        }
      ],
      "uniqueIndexes": [],
      "document": {
        "properties": {
          "_id": {
            "bsonType": "string"
          },
          "recrd": {
            "bsonType": "string"
          },
          "vesslterms": {
            "bsonType": "string"
          },
          "feature_type": {
            "bsonType": "string"
          },
          "chart": {
            "bsonType": "string"
          },
          "latdec": {
            "bsonType": "double"
          },
          "londec": {
            "bsonType": "double"
          },
          "gp_quality": {
            "bsonType": "string"
          },
          "depth": {
            "bsonType": "string"
          },
          "sounding_type": {
            "bsonType": "string"
          },
          "history": {
            "bsonType": "string"
          },
          "quasou": {
            "bsonType": "string"
          },
          "watlev": {
            "bsonType": "string"
          },
          "coordinates": {
            "bsonType": "array",
            "items": {
              "bsonType": "double"
            }
          }
        }
      }
    }
  ],
  "version": 1
}'''

queryfier = MongoQuery('Phi2')
text = 'Find the count of shipwrecks for each unique combination of "latdec" and "longdec"'
queryfier.generate_query(schema, text, max_length = 1024)

text = 'Find the total count of shipwreck for each unique category of chart'
queryfier.generate_query(schema, text, max_length = 1024)


```


## 3. Kusto Query
Suppose you want to convert the textual question to Kusto query, follow the code below

```py
from nl2query import KustoQuery
cols = ['conference', 'sessionid', 'session_title', 'session_type', 'owner', 'participants', 'URL', 'level', 'session_location', 'starttime', 'duration', 'time_and_duration', 'kusto_affinity']

queryfier = KustoQuery(cols, 'ConferenceSessions')
queryfier.generate_query('''find the session ids which have duration greater than 10 and having Manoj Raheja as the owner''')
```

## 4. Cypher(Neo4j) Query
Suppose you want to convert the textual question to Cypher query, follow the code below

```py
from nl2query import CypherQuery

node_labels = {'User': ['display_name', 'uuid'], 'Comment': ['score', 'link', 'uuid']}
relationships = ['COMMENTED']
queryfier = CypherQuery(node_labels, relationships)
queryfier.generate_query('list the links of all the comments done by "jose_bacoy"')

node_labels = {'Case': ['gender', 'reportdate', 'ageunit', 'reporteroccupation', 'primaryid', 'age', 'eventDate'], 'Outcome': ['code', 'outcome']}
relationships = ['RESULTED_IN']
queryfier = CypherQuery(node_labels, relationships)
queryfier.generate_query('find the outcomes of people who are female and below the age of 32')

node_labels = {'Person': ['id', 'name', 'dob']}
relationships = []
queryfier = CypherQuery(node_labels, relationships)
res = queryfier.generate_query('find the dob of people who have "Andreia" in their name')

```



## Changelog

Refer to the [CHANGELOG.md](CHANGELOG.md) file.

<!-- Badges -->

[pypi-image]: https://img.shields.io/pypi/v/nl2query
[pypi-url]: https://pypi.org/project/nl2query/
[versions-image]: https://img.shields.io/pypi/pyversions/nl2query
[versions-url]: https://pypi.org/project/nl2query/
[build-image]: https://github.com/Chirayu-Tripathi/nl2query/actions/workflows/build.yaml/badge.svg
[build-url]: https://github.com/Chirayu-Tripathi/nl2query/actions/workflows/build.yaml
[stars-image]: https://img.shields.io/github/stars/Chirayu-Tripathi/nl2query
[stars-url]: https://github.com/Chirayu-Tripathi/nl2query
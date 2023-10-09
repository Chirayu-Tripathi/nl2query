# nl2query

 > Convert natural language text inputs to Pandas, MongoDB, Kusto, and Cypher(Neo4j) queries. The models used are fine-tuned versions of CodeT5+ 220m models.
[![Downloads](https://static.pepy.tech/badge/nl2query)](https://pepy.tech/project/nl2query)
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
Suppose you want to convert the textual question to Mongo query, follow the code below

```py
from nl2query import MongoQuery
import pymongo # import if performing analysis using python client
keys = ['_id', 'index', 'passengerid', 'survived', 'Pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked'] #keys present in the collection to be queried.
queryfier = MongoQuery(keys, 'titanic')
queryfier.generate_query('''which pclass has the minimum average fare?''')

keys = ['_id', 'index', 'total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size'] 
queryfier = MongoQuery(keys, 'tips')
queryfier.generate_query('''find the day on which combined sales was highest''')

```
In the above code the keys can be found by running the following piece `db.tips.find_one({}).keys()`


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

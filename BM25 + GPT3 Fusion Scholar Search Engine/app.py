from flask import Flask, request, render_template, jsonify, url_for
from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
from urllib.request import urlopen
import json
import os

# Load dataframe for papers. 
df1 = pd.read_csv('df1_for_GPT3_search.csv')

es = Elasticsearch(hosts=["http://localhost:9200"])
print(f"Connected to ElasticSearch cluster `{es.info()}`")

openai.api_key = os.environ["OPENAI_API_KEY"]
doc_embeddings = np.load("title_ab_embeddings_gpt3.npy", allow_pickle=True)
df1['doc_emb'] = doc_embeddings

app = Flask(__name__)

MAX_SIZE = 20

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search")
def search():
    query = request.args["q"].lower()
    print(len(df1))
    print(openai.api_key)
    print(query)
    top100 = search_100(query)
    result = rerank(df1, query, MAX_SIZE, top100)
    dict_list = []
    for index, row in result.iterrows():
        temp = {}
        temp['title'] = row['title']
        temp['paperAbstract'] = row['paperAbstract']
        temp['venue'] = row['venue']
        temp['numCitedBy'] = row['numCitedBy']
        dict_list.append(temp)
    return jsonify(dict_list)

def search_100(query):
    MAX_SIZE = 100
    payload = {
        "multi_match": {
                "query": query,
                "fields": ['paperAbstract','title']
            }
            
        }
    resp = es.search(index="s2_doc", query=payload, size=MAX_SIZE)
    result = []
    # Extract info needed for display. 
    for item in resp['hits']['hits']:
        result.append(item['_id'])
    return result

def rerank(df, query, n, top100):
    mask = df['docno'].isin(top100)
    selected_rows = df.loc[mask]
    column_name = "similarity"
    query = openai.Embedding.create(
    input=query,
    engine="text-search-davinci-query-001"
)["data"][0]["embedding"]
    selected_rows[column_name] = selected_rows['doc_emb'].apply(lambda x: cosine_similarity(x["data"][0]["embedding"], query))
    result = (
        selected_rows.sort_values(column_name, ascending=False)
        .head(n)
    )
    result = result.drop(columns = ['docno', 'doc_emb'])
    return result


if __name__ == "__main__":
    app.run(debug=True)
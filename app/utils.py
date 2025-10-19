import os
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


def embed_text(text: str, model: SentenceTransformer):
    return model.encode(text, convert_to_numpy=True).tolist()


def search_on_db(embed: list[float]):
    DB_URL = os.getenv("DB_URL")
    INDEX_NAME = os.getenv("INDEX_NAME")

    query = {
        "size": 10,
        "query": {
            "knn": {
                "field": "embedding",
                "query_vector": embed,
                "k": 10,
                "num_candidates": 100,
            }
        },
    }
    res = requests.get(f"{DB_URL}/{INDEX_NAME}/_search", json=query)
    res.raise_for_status()
    return res.json()


def parse_db_response(res: dict):
    results = []
    for row in res["hits"]["hits"]:
        source = row["_source"]
        results.append(
            {
                "name": source.get("name"),
                "img": source.get("image"),
                "description": source.get("description"),
            }
        )
    return results

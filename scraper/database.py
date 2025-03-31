import json
import os
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


def make_instance(pc, index_name):
    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={"model": "llama-text-embed-v2", "field_map": {"text": "chunk_text"}},
        )


def test_add_items(pc, index_name):
    records = [
        {
            "_id": "rec1",
            "chunk_text": "The Eiffel Tower was completed in 1889 and stands in Paris, France.",
            "category": "history",
        },
        {
            "_id": "rec2",
            "chunk_text": "Photosynthesis allows plants to convert sunlight into energy.",
            "category": "science",
        },
    ]

    dense_index = pc.Index(index_name)

    # Upsert the records into a namespace
    dense_index.upsert_records("example-namespace", records)
    return dense_index


def retrieve_answers(dense_index, query):
    results = dense_index.search(
        namespace="example-namespace", query={"top_k": 10, "inputs": {"text": query}}
    )

    # Print the results
    for hit in results["result"]["hits"]:
        print(
            f"id: {hit['_id']}, score: {round(hit['_score'], 2)}, text: {hit['fields']['chunk_text']}, category: {hit['fields']['category']}"
        )


def retrieve_answers_reranker(dense_index, query):
    reranked_results = dense_index.search(
        namespace="example-namespace",
        query={"top_k": 10, "inputs": {"text": query}},
        rerank={
            "model": "bge-reranker-v2-m3",
            "top_n": 10,
            "rank_fields": ["chunk_text"],
        },
    )

    # Print the reranked results
    for hit in reranked_results["result"]["hits"]:
        print(
            f"id: {hit['_id']}, score: {round(hit['_score'], 2)}, text: {hit['fields']['chunk_text']}, category: {hit['fields']['category']}"
        )


def retrieve_answers_for_prompt(
    index_name: str, namespace: str, query: str, result_num: int
) -> List[str]:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    dense_index = pc.Index(index_name)

    reranked_results = dense_index.search(
        namespace=namespace,
        query={"top_k": result_num, "inputs": {"text": query}},
        # rerank={
        #     "model": "bge-reranker-v2-m3",
        #     "top_n": result_num,
        #     "rank_fields":["_node_content['text']"]
        # }
    )

    texts = []
    for hit in reranked_results["result"]["hits"]:
        node_content = json.loads(hit["fields"]["_node_content"])
        texts.append(node_content["text"])
    return texts


if __name__ == "__main__":

    # load_dotenv()

    # pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    # index_name = "paul-allen"

    # make_instance(pc, index_name)

    # dense_index = test_add_items(pc, index_name)

    # # import time
    # # time.sleep(10)

    # # View stats for the index
    # stats = dense_index.describe_index_stats()
    # print(stats)

    # # Define the query
    # query = "Famous historical structures and monuments"

    # # Search the dense index
    # retrieve_answers(dense_index, query)

    # # Search the dense index and rerank results
    # retrieve_answers_reranker(dense_index, query)
    res = retrieve_answers_for_prompt(
        index_name="paul-allen",
        namespace="info",
        query="when didi he get a Yacht",
        result_num=2,
    )
    print(res)

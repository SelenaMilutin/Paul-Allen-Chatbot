import json
import os
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone


def make_instance(pc, index_name: str) -> None:
    """
    Creates a Pinecone index if it does not already exist.

    Args:
        pc: An instance of the Pinecone client.
        index_name (str): The name of the Pinecone index to check or create.

    Returns:
        None
    """
    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={"model": "llama-text-embed-v2", "field_map": {"text": "chunk_text"}},
        )


def test_add_items(pc, index_name: str) -> None:
    """
    Adds records to the example-namespace of the dence index.

    Args:
        pc: An instance of the Pinecone client.
        index_name (str): The name of the Pinecone index to check or create.

    Returns:
        None
    """
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


def retrieve_answers(dense_index, query: str) -> None:
    """
    Searches a Pinecone index for relevant documents based on the given query and prints the results.

    This function performs a search in the specified Pinecone index using a query text 
    and retrieves the top 10 most relevant results. It then prints each result's ID, 
    similarity score, text content, and category.

    Args:
        dense_index: An instance of a Pinecone index used for searching.
        query (str): The query text used to find relevant documents.

    Returns:
        None
    """
    results = dense_index.search(
        namespace="example-namespace", query={"top_k": 10, "inputs": {"text": query}}
    )

    # Print the results
    for hit in results["result"]["hits"]:
        print(
            f"id: {hit['_id']}, score: {round(hit['_score'], 2)}, text: {hit['fields']['chunk_text']}, category: {hit['fields']['category']}"
        )


def retrieve_answers_reranker(dense_index, query: str) -> None:
    """
    Searches a Pinecone index AND RERANKS them for relevant documents based on the given query and prints the results.

    This function performs a search in the specified Pinecone index using a query text 
    and retrieves the top 10 most relevant results. It then prints each result's ID, 
    similarity score, text content, and category.

    Args:
        dense_index: An instance of a Pinecone index used for searching.
        query (str): The query text used to find relevant documents.

    Returns:
        None
    """
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
    """
    Retrieves relevant text passages from a Pinecone index based on the given query.

    This function connects to a Pinecone vector database, performs a search using the
    specified query, and extracts relevant text from the retrieved results. The function
    assumes that each retrieved item contains a `_node_content` field, which is stored
    as a JSON string and contains a `text` field.

    Args:
        index_name (str): The name of the Pinecone index to search in.
        namespace (str): The namespace within the Pinecone index to narrow the search.
        query (str): The query text used to retrieve relevant documents.
        result_num (int): The number of top results to retrieve.

    Returns:
        List[str]: A list of extracted text passages from the retrieved search results.
    """
    load_dotenv()
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

    load_dotenv()

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "paul-allen"

    make_instance(pc, index_name)

    dense_index = test_add_items(pc, index_name)

    # import time
    # time.sleep(10)

    # View stats for the index
    stats = dense_index.describe_index_stats()
    print(stats)

    # Define the query
    query = "Famous historical structures and monuments"

    # Search the dense index
    retrieve_answers(dense_index, query)

    # Search the dense index and rerank results
    retrieve_answers_reranker(dense_index, query)

    res = retrieve_answers_for_prompt(
        index_name="paul-allen",
        namespace="info",
        query="when didi he get a Yacht",
        result_num=2,
    )
    print(res)

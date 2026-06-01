import sys

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# Settings for the local vector search.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "team_memory"
TOP_K_RESULTS = 3
TEXT_PREVIEW_LENGTH = 300


def main():
    # Make sure the user provided a search query.
    if len(sys.argv) < 2:
        print('Usage: python3 scripts/search_vault.py "your search question"')
        return

    # Join all command-line words so quoted and unquoted queries both work.
    query = " ".join(sys.argv[1:])

    print("Starting vault search...")
    print(f"Query: {query}")
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")

    # Load the same embedding model used by scripts/index_vault.py.
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Turn the user's search query into a vector.
    print("Creating query embedding...")
    query_vector = embedding_model.encode(query).tolist()

    print(f"Connecting to Qdrant: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL)

    # Search Qdrant for the closest matching meeting summaries.
    print(f"Searching collection: {COLLECTION_NAME}")
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K_RESULTS,
        with_payload=True,
    )

    # qdrant-client 1.16.1 returns a response object with a points list.
    result_points = results.points

    if not result_points:
        print("No results found.")
        return

    print(f"Found {len(result_points)} result(s):\n")

    for result in result_points:
        payload = result.payload or {}
        text = payload.get("text", "")

        # Keep the preview short so search output stays easy to scan.
        preview = text.replace("\n", " ")[:TEXT_PREVIEW_LENGTH]

        print(f"Score: {result.score}")
        print(f"File name: {payload.get('file_name')}")
        print(f"File path: {payload.get('file_path')}")
        print(f"Preview: {preview}")
        print()


if __name__ == "__main__":
    main()

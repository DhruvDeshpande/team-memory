from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


# Settings for the local vector index.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "team_memory"
SUMMARY_NOTES_FOLDER = Path("vault/meetings/summaries")


def main():
    print("Starting vault indexing...")
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")

    # Load the sentence-transformers model that turns text into vectors.
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # The vector size must match the model's embedding output size.
    vector_size = embedding_model.get_sentence_embedding_dimension()
    print(f"Embedding vector size: {vector_size}")

    print(f"Connecting to Qdrant: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL)

    # Recreate the collection so the index starts fresh each time this script runs.
    print(f"Creating/recreating Qdrant collection: {COLLECTION_NAME}")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )

    # Find all markdown summary notes.
    markdown_files = sorted(SUMMARY_NOTES_FOLDER.glob("*.md"))

    if not markdown_files:
        print(f"No markdown files found in: {SUMMARY_NOTES_FOLDER}")
        return

    print(f"Found {len(markdown_files)} markdown file(s) to index.")

    points = []

    for point_id, markdown_file in enumerate(markdown_files, start=1):
        print(f"Reading file: {markdown_file}")

        # Read the markdown note text.
        text = markdown_file.read_text(encoding="utf-8")

        # Create an embedding vector for this note.
        print(f"Creating embedding for: {markdown_file.name}")
        embedding = embedding_model.encode(text).tolist()

        # Store useful metadata with the vector.
        payload = {
            "file_path": str(markdown_file),
            "file_name": markdown_file.name,
            "text": text,
        }

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
        )

    # Upload all points to Qdrant in one batch.
    print(f"Uploading {len(points)} point(s) to Qdrant...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print("Vault indexing complete.")


if __name__ == "__main__":
    main()

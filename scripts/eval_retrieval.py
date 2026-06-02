import json
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# Settings used for retrieval evaluation.
GOLDEN_QUESTIONS_FILE = Path("evals/golden_questions.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "team_memory"
TOP_K_RESULTS = 3


def load_golden_questions():
    # Load the test questions and expected source filenames.
    if not GOLDEN_QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing eval file: {GOLDEN_QUESTIONS_FILE}")

    with GOLDEN_QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_question_text(test_case):
    # Support either "question" or "query" so the eval file can stay simple.
    return test_case.get("question") or test_case.get("query")


def main():
    print("Starting retrieval evaluation...")
    print(f"Loading golden questions from: {GOLDEN_QUESTIONS_FILE}")
    test_cases = load_golden_questions()

    if not test_cases:
        print("No test cases found.")
        return

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"Connecting to Qdrant: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL)

    top_1_passes = 0
    top_3_passes = 0

    for test_number, test_case in enumerate(test_cases, start=1):
        question = get_question_text(test_case)
        expected_source = test_case.get("expected_source")

        if not question or not expected_source:
            print(f"\nTest {test_number}: skipped because it is missing data.")
            continue

        print(f"\nTest {test_number}")
        print(f"Question: {question}")
        print(f"Expected source: {expected_source}")

        # Turn the question into an embedding vector.
        query_vector = embedding_model.encode(question).tolist()

        # Search Qdrant using the qdrant-client 1.16.1 compatible API.
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=TOP_K_RESULTS,
            with_payload=True,
        )

        retrieved_sources = []

        for point in results.points:
            payload = point.payload or {}
            retrieved_sources.append(payload.get("file_name"))

        print(f"Retrieved sources: {retrieved_sources}")

        top_1_pass = bool(retrieved_sources) and retrieved_sources[0] == expected_source
        top_3_pass = expected_source in retrieved_sources

        if top_1_pass:
            top_1_passes += 1

        if top_3_pass:
            top_3_passes += 1

        print(f"Top-1: {'PASS' if top_1_pass else 'FAIL'}")
        print(f"Top-3: {'PASS' if top_3_pass else 'FAIL'}")

    total_tests = len(test_cases)
    top_1_accuracy = top_1_passes / total_tests
    top_3_accuracy = top_3_passes / total_tests

    print("\n[Final Accuracy]")
    print(f"Top-1 accuracy: {top_1_passes}/{total_tests} ({top_1_accuracy:.2%})")
    print(f"Top-3 accuracy: {top_3_passes}/{total_tests} ({top_3_accuracy:.2%})")


if __name__ == "__main__":
    main()

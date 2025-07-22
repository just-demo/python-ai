import uuid

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv(override=True)

db_name = "people_db"
collection_name = "people"
category_people = {
    "literature": [
        "William Shakespeare was an English playwright, poet and actor.",
    ],
    "art": [
        "Leonardo di ser Piero da Vinci was an Italian polymath of the High Renaissance who was active as a painter, draughtsman, engineer, scientist, theorist, sculptor, and architect.",
    ],
    "film": [
        "Sir Charles Spencer Chaplin was an English comic actor, filmmaker, and composer who rose to fame in the era of silent film.",
    ],
    "sport": [
        "Muhammad Ali was an American professional boxer and social activist.",
    ],
}


def create_collection(client, collection_name):
    existing_collection_names = [collection.name for collection in client.list_collections()]
    if collection_name in existing_collection_names:
        print(f"Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name)
    return client.create_collection(collection_name)


def save_documents(collection, model, category_people):
    for category, people in category_people.items():
        for biography in people:
            collection.add(
                ids=[str(uuid.uuid4())],
                documents=[biography],
                embeddings=model.encode([biography]).astype(float).tolist(),
                metadatas=[{"category": category}])


def search(collection, model, text):
    results = collection.query(query_embeddings=model.encode([text]).astype(float).tolist(), n_results=10)
    return [{
        "text": text,
        "category": metadata["category"],
        "distance": distance
    } for text, metadata, distance in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )]


def test_search(collection, model, text):
    search_results = search(collection, model, text)
    print(f"\n{text}:")
    for search_result in search_results:
        print(f"{round(search_result['distance'], 2)} / {search_result['category']} / {search_result['text']}")


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path=db_name)
collection = create_collection(client, collection_name)
save_documents(collection, model, category_people)

test_search(collection, model, "Literature")
test_search(collection, model, "Football")
test_search(collection, model, "Hiking")
test_search(collection, model, "Cinema")
test_search(collection, model, "Hamlet")

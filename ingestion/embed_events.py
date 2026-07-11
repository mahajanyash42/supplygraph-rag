"""
embed_events.py

Computes a sentence embedding for each Event's description text, and stores
it on the node so the vector index (created in Step 7a) can search over it.

Run AFTER load_graph.py has already populated the Event nodes.

Run with:
    python embed_events.py
"""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dimensions -- matches our vector index


def main():
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (first run downloads it, may take a minute)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    with driver.session() as session:
        # 1. Fetch every event's id and description text
        result = session.run("MATCH (e:Event) RETURN e.id AS id, e.description AS description")
        events = [record.data() for record in result]
        print(f"Found {len(events)} events to embed.")

        # 2. Compute one embedding vector per description
        descriptions = [e["description"] for e in events]
        embeddings = model.encode(descriptions, show_progress_bar=True)

        # 3. Store each embedding back onto its Event node
        for event, embedding in zip(events, embeddings):
            session.run(
                """
                MATCH (e:Event {id: $id})
                CALL db.create.setNodeVectorProperty(e, 'embedding', $embedding)
                """,
                id=event["id"],
                embedding=embedding.tolist(),
            )

    driver.close()
    print("\nDone. Every Event node now has an 'embedding' property.")


if __name__ == "__main__":
    main()
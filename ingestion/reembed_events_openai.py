"""
reembed_events_openai.py

Re-computes Event embeddings using OpenAI's embedding API instead of the
local sentence-transformers model, so the DEPLOYED backend doesn't need to
load PyTorch (which is too heavy for Render's free 512MB memory limit).

We request 384-dimension output specifically, matching our existing vector
index -- OpenAI's text-embedding-3-small model supports a `dimensions`
parameter for exactly this, so we don't need to recreate the vector index.

Run this ONCE, locally, pointed at your Aura instance (same .env as always).
This does not affect your local Neo4j Desktop setup -- that can keep using
the local sentence-transformers embeddings for the professor demo.

Run with:
    python reembed_events_openai.py
"""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv(override=True)

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run("MATCH (e:Event) RETURN e.id AS id, e.description AS description")
        events = [record.data() for record in result]
        print(f"Found {len(events)} events to re-embed with OpenAI.")

        descriptions = [e["description"] for e in events]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=descriptions,
            dimensions=384,
        )
        embeddings = [item.embedding for item in response.data]

        for event, embedding in zip(events, embeddings):
            session.run(
                """
                MATCH (e:Event {id: $id})
                CALL db.create.setNodeVectorProperty(e, 'embedding', $embedding)
                """,
                id=event["id"],
                embedding=embedding,
            )

    driver.close()
    print("\nDone. Events now have OpenAI-based embeddings (384 dimensions).")


if __name__ == "__main__":
    main()
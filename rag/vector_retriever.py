"""
vector_retriever.py

Semantic search over our Event descriptions. Unlike graph_retriever.py, this
doesn't look at relationships at all -- it finds events whose description TEXT
is closest in *meaning* to your question, then has an LLM answer using that
retrieved text.

This is for questions the graph structurally cannot answer -- details that only
exist in the free-text report (how long something will take, what the backup
plan is), not as a relationship between nodes.

NOTE: this uses OpenAI's embedding API rather than a local sentence-transformers
model. That's a deliberate change from the local/demo version specifically for
deployment -- PyTorch (which sentence-transformers depends on) is too heavy for
Render's free-tier 512MB memory limit. The stored event embeddings were
re-computed with the same OpenAI model (see ingestion/reembed_events_openai.py)
so query-time and stored embeddings are from the same model family.

Run with:
    python vector_retriever.py
"""

import os

from dotenv import load_dotenv
from langchain_neo4j import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(override=True)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the disruption event reports below.
If the reports don't contain the answer, say so explicitly.

Event reports:
{context}

Question: {question}

Answer:"""
)

_vector_store = None


def build_vector_store():
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=384)
    _vector_store = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        database=os.environ.get("NEO4J_DATABASE", "neo4j"),
        index_name="event_embeddings",
        node_label="Event",
        text_node_property="description",
        embedding_node_property="embedding",
    )
    return _vector_store


def answer_with_vector_search(question: str, k: int = 2):
    store = build_vector_store()

    docs = store.similarity_search(question, k=k)

    print("\n--- Retrieved events (by meaning, not keyword) ---")
    for d in docs:
        print(f"[{d.metadata.get('id')}] {d.metadata.get('title')}")

    context = "\n\n".join(
        f"[{d.metadata.get('id', '?')}] {d.metadata.get('title', '')}\n{d.page_content}"
        for d in docs
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = ANSWER_PROMPT | llm
    result = chain.invoke({"context": context, "question": question})

    return result.content


if __name__ == "__main__":
    question = "How long will the Vietnam facility be down, and is there a backup plan?"
    answer = answer_with_vector_search(question)
    print("\n--- Answer ---")
    print(answer)
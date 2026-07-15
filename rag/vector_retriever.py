"""
vector_retriever.py

Semantic search over our Event descriptions. Unlike graph_retriever.py, this
doesn't look at relationships at all -- it finds events whose description TEXT
is closest in *meaning* to your question, then has an LLM answer using that
retrieved text.

This is for questions the graph structurally cannot answer -- details that only
exist in the free-text report (how long something will take, what the backup
plan is), not as a relationship between nodes.

Run with:
    python vector_retriever.py
"""

import os

from dotenv import load_dotenv
from langchain_neo4j import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(override=True)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # must match embed_events.py exactly

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the disruption event reports below.
If the reports don't contain the answer, say so explicitly.

Event reports:
{context}

Question: {question}

Answer:"""
)


def build_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{EMBEDDING_MODEL_NAME}")
    return Neo4jVector.from_existing_index(
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


def answer_with_vector_search(question: str, k: int = 2):
    store = build_vector_store()

    # This is the actual semantic search step -- it compares the question's
    # embedding against every event's embedding, and returns the closest matches.
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
    question = "Is there a workaround for the Vietnam Lithium Mine disruption?"
    answer = answer_with_vector_search(question)
    print("\n--- Answer ---")
    print(answer)
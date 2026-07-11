"""
graph_retriever.py

Takes a plain-English question, asks an LLM to translate it into a Cypher
query using our exact schema, runs that query against Neo4j, then has the
LLM phrase the result as a readable answer.

Run with:
    python graph_retriever.py
"""

import os

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# The prompt that teaches the LLM our exact schema and relationship directions.
# Few-shot examples matter a LOT here -- a bare schema description alone tends
# to produce plausible-looking but subtly wrong Cypher (wrong direction, wrong
# relationship name -- the exact mistakes you just made and fixed by hand).
# ---------------------------------------------------------------------------
CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j Cypher query generator for a
supply chain risk knowledge graph.

Schema:
{schema}

Relationship guide (use exactly these directions):
(:Product)-[:REQUIRES]->(:Component)
(:Component)-[:SUPPLIED_BY]->(:Supplier)
(:Supplier)-[:SUB_SUPPLIER_OF]->(:Supplier)   // child (lower tier) -> parent (higher tier)
(:Supplier)-[:OPERATES_AT]->(:Facility)
(:Facility)-[:LOCATED_IN]->(:Country)
(:Event)-[:IMPACTS]->(:Facility)

CRITICAL SYNTAX RULE: every node that has inline properties (e.g. {{name: "X"}})
MUST also have a variable name, even if that variable is never used again.
CORRECT:   (p:Product {{name: "X"}})
INCORRECT: (:Product {{name: "X"}})   -- missing variable name, causes failures

Examples:

Question: Which products are affected by event E02?
Cypher:
MATCH (e:Event {{id: "E02"}})-[:IMPACTS]->(:Facility)<-[:OPERATES_AT]-(s:Supplier)
MATCH (s)-[:SUB_SUPPLIER_OF*0..3]->(tier1:Supplier)
MATCH (comp:Component)-[:SUPPLIED_BY]->(tier1)
MATCH (p:Product)-[:REQUIRES]->(comp)
RETURN DISTINCT p.name AS affected_product

Question: Which products are affected by the flood at the Vietnam Lithium Mine?
Cypher:
MATCH (e:Event)-[:IMPACTS]->(:Facility)<-[:OPERATES_AT]-(s:Supplier)
WHERE toLower(e.title) CONTAINS "flood" AND toLower(e.title) CONTAINS "lithium mine"
MATCH (s)-[:SUB_SUPPLIER_OF*0..3]->(tier1:Supplier)
MATCH (comp:Component)-[:SUPPLIED_BY]->(tier1)
MATCH (p:Product)-[:REQUIRES]->(comp)
RETURN DISTINCT p.name AS affected_product

Question: Which suppliers, at every tier, does the Aurora Smartphone depend on?
Cypher:
MATCH (p:Product {{name: "Aurora Smartphone"}})-[:REQUIRES]->(c:Component)-[:SUPPLIED_BY]->(tier1:Supplier)
OPTIONAL MATCH (upstream:Supplier)-[:SUB_SUPPLIER_OF*1..3]->(tier1)
WITH collect(DISTINCT tier1.name) AS tier1_names, collect(DISTINCT upstream.name) AS upstream_names
UNWIND (tier1_names + upstream_names) AS supplier
RETURN DISTINCT supplier

Question: What happens to our products if the chip factory in Taiwan stops working?
Cypher:
MATCH (e:Event)-[:IMPACTS]->(f:Facility)-[:LOCATED_IN]->(c:Country)
WHERE toLower(f.name) CONTAINS "chip" AND toLower(c.name) CONTAINS "taiwan"
MATCH (f)<-[:OPERATES_AT]-(s:Supplier)
MATCH (s)-[:SUB_SUPPLIER_OF*0..3]->(tier1:Supplier)
MATCH (comp:Component)-[:SUPPLIED_BY]->(tier1)
MATCH (p:Product)-[:REQUIRES]->(comp)
RETURN DISTINCT p.name AS affected_product

IMPORTANT: country_id and facility_id and supplier_id etc. are internal foreign keys
(like "C02"), never plain-English names. NEVER filter directly on an *_id property
using a country/place name -- always traverse to the actual node (e.g. via
LOCATED_IN) and match its .name property instead. Also prefer `CONTAINS` over exact
equality for any name/title the user described in their own words, since the exact
stored string may differ from their phrasing.

Question: If there's a delay at the India Casing Plant, which products are affected?
Cypher:
MATCH (e:Event)-[:IMPACTS]->(f:Facility)
WHERE toLower(f.name) CONTAINS "casing plant"
MATCH (f)<-[:OPERATES_AT]-(s:Supplier)
MATCH (s)-[:SUB_SUPPLIER_OF*0..3]->(tier1:Supplier)
MATCH (comp:Component)-[:SUPPLIED_BY]->(tier1)
MATCH (p:Product)-[:REQUIRES]->(comp)
RETURN DISTINCT p.name AS affected_product

NOTE: when the question names a FACILITY (a factory, plant, mine, port, etc.),
match on Facility.name, NOT Event.title -- the event's title is just a short
summary and may not contain the exact words the user used for the facility.
Facility names are the stable, directly-referenced entity.

Question: {question}
Cypher:"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

# ---------------------------------------------------------------------------
# Custom answer-writing prompt. LangChain's default QA prompt proved unreliable
# in testing -- it sometimes said "I don't know" even when the context clearly
# contained a valid answer. Writing our own gives us control and predictability,
# the same reason we wrote our own Cypher-generation prompt instead of trusting
# the built-in one.
# ---------------------------------------------------------------------------
QA_GENERATION_TEMPLATE = """You are answering a question using data retrieved from a graph database.
The information below is factual and complete for this question -- treat it as ground truth.

Question: {question}

Retrieved data:
{context}

Instructions:
- If the retrieved data is an empty list, say you don't have enough information to answer.
- Otherwise, use ALL the retrieved data to write a clear, natural-language answer.
- Do not say "I don't know" if the retrieved data contains any items.

Answer:"""

QA_GENERATION_PROMPT = PromptTemplate(
    input_variables=["question", "context"], template=QA_GENERATION_TEMPLATE
)


def build_chain():
    graph = Neo4jGraph(
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        enhanced_schema=True,
    )
    graph.refresh_schema()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
        qa_prompt=QA_GENERATION_PROMPT,
        verbose=True,  # prints the generated Cypher to the terminal -- keep this on for now
        return_intermediate_steps=True,
        allow_dangerous_requests=True,  # required by langchain; fine since our queries are read-only
        validate_cypher=True,
    )
    return chain


def answer_with_graph(question: str) -> dict:
    """Reusable entry point: takes a question, returns {'answer': str, 'cypher': str}.

    Includes a self-correction retry: if the first generated query returns an
    empty result, we don't just accept "I don't know" -- we ask the LLM to
    reconsider, explicitly telling it the first attempt found nothing and to
    reconsider which node type actually holds the entity the question
    describes (Component vs Facility vs Country vs Supplier). This is a much
    more robust fix than hand-writing a new few-shot example for every new
    phrasing we happen to test -- it generalizes to phrasings we HAVEN'T
    tested, instead of only ones we have.
    """
    chain = build_chain()
    result = chain.invoke({"query": question})

    cypher_used = None
    context = None
    for step in result["intermediate_steps"]:
        if "query" in step:
            cypher_used = step["query"]
        if "context" in step:
            context = step["context"]

    if not context:  # empty list -- the query ran but matched nothing
        print("\n[First attempt returned no results -- retrying with a corrective hint]")
        retry_question = (
            f"{question}\n\n"
            f"(Note: a previous attempt at this question used the query below and "
            f"found NO matching data. Reconsider which node type actually holds the "
            f"entity being described -- it may be a Component, Supplier, Facility, "
            f"or Country, not necessarily the one assumed below. Try a different "
            f"matching approach.\n\nPrevious query:\n{cypher_used})"
        )
        result = chain.invoke({"query": retry_question})
        cypher_used = None
        for step in result["intermediate_steps"]:
            if "query" in step:
                cypher_used = step["query"]

    return {"answer": result["result"], "cypher": cypher_used}


if __name__ == "__main__":
    question = "If the plant that makes phone casings has any kind of problem, which of our products would be affected?"
    result = answer_with_graph(question)

    print("\n--- Generated Cypher ---")
    print(result["cypher"])

    print("\n--- Answer ---")
    print(result["answer"])
"""
hybrid_agent.py

The actual "agent" of the project: takes ANY question, decides whether it
needs graph retrieval, vector retrieval, or both, runs the right one(s), and
combines the results into one answer.

This is what makes the system feel like one coherent thing, instead of you
manually choosing which script to run.

Run with:
    python hybrid_agent.py
"""

from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from graph_retriever import answer_with_graph
from vector_retriever import answer_with_vector_search


# ---------------------------------------------------------------------------
# The "state" is just a dictionary that gets passed between steps, gradually
# filled in as we go -- this is the whole idea of LangGraph: a shared object
# that each node reads from and writes to.
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    question: str
    route: str            # "graph", "vector", or "hybrid"
    graph_answer: str
    vector_answer: str
    final_answer: str


ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """Classify the question below into exactly one category:

- "graph"  : answerable by traversing structured relationships
             (e.g. which products/suppliers/countries are connected to what)
- "vector" : answerable only from free-text disruption reports
             (e.g. severity, timelines, cost estimates, workarounds, root causes)
- "hybrid" : needs BOTH a relationship traversal AND free-text detail
             (e.g. "which products are affected, and how long will it last?")

Respond with exactly one word: graph, vector, or hybrid.

Question: {question}"""
)


def route_node(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = ROUTER_PROMPT | llm
    result = chain.invoke({"question": state["question"]})
    route = result.content.strip().lower()
    if route not in ("graph", "vector", "hybrid"):
        route = "hybrid"  # safe default if the classifier gives something unexpected
    print(f"\n[Router decided: {route}]")
    return {"route": route}


def graph_node(state: AgentState) -> AgentState:
    result = answer_with_graph(state["question"])
    return {"graph_answer": result["answer"]}


def vector_node(state: AgentState) -> AgentState:
    answer = answer_with_vector_search(state["question"])
    return {"vector_answer": answer}


SYNTHESIS_PROMPT = ChatPromptTemplate.from_template(
    """Combine the following evidence into one clear answer to the question.

Question: {question}

Graph traversal result:
{graph_answer}

Disruption report evidence:
{vector_answer}

Final answer:"""
)


def synthesize_node(state: AgentState) -> AgentState:
    route = state["route"]

    if route == "graph":
        return {"final_answer": state.get("graph_answer", "")}
    if route == "vector":
        return {"final_answer": state.get("vector_answer", "")}

    # hybrid: combine both pieces of evidence into one answer
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = SYNTHESIS_PROMPT | llm
    result = chain.invoke({
        "question": state["question"],
        "graph_answer": state.get("graph_answer", "(no graph evidence retrieved)"),
        "vector_answer": state.get("vector_answer", "(no report evidence retrieved)"),
    })
    return {"final_answer": result.content}


def route_decision(state: AgentState) -> str:
    """Tells LangGraph which node to go to next, based on the router's decision."""
    return state["route"]


def after_graph_decision(state: AgentState) -> str:
    """After running graph retrieval: if hybrid, also run vector retrieval.
    Otherwise, go straight to writing the final answer."""
    return "vector_retrieve" if state["route"] == "hybrid" else "synthesize"


def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("route", route_node)
    graph.add_node("graph_retrieve", graph_node)
    graph.add_node("vector_retrieve", vector_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("route")

    graph.add_conditional_edges(
        "route",
        route_decision,
        {
            "graph": "graph_retrieve",
            "vector": "vector_retrieve",
            "hybrid": "graph_retrieve",  # hybrid starts with graph, then adds vector
        },
    )

    graph.add_conditional_edges(
        "graph_retrieve",
        after_graph_decision,
        {
            "vector_retrieve": "vector_retrieve",
            "synthesize": "synthesize",
        },
    )

    graph.add_edge("vector_retrieve", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


if __name__ == "__main__":
    agent = build_agent()

    question = "Which suppliers, at every tier, does the NimbusBook depend on?"
    result = agent.invoke({"question": question})

    print("\n--- Final Answer ---")
    print(result["final_answer"])
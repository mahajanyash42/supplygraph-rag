# SupplyGraph — Graph-RAG for Multi-Tier Supply Chain Risk Reasoning

A hybrid Graph-RAG system built with **Neo4j**, **LangChain**, and **LangGraph** that
routes each question to graph traversal, semantic (vector) search, or both —
instead of applying a single retrieval strategy uniformly.

## The problem

Companies typically have visibility into their direct ("tier-1") suppliers, but
very limited visibility into the suppliers of their suppliers. When a disruption
hits several tiers upstream, the fact "this event affects this product" doesn't
exist anywhere as a sentence — it only exists as a chain of relationships
(`Event → Facility → Supplier → Component → Product`) that has to be traversed.
A vector store retrieves by similarity, not by connection, and structurally
cannot answer this kind of question. A graph can.

## Architecture
                Question
                   │
             ┌─────▼──────┐
             │   Router    │  classifies: graph / vector / hybrid
             └──┬───────┬─┘
    structural  │       │  narrative
            ┌────▼──┐ ┌──▼─────────┐
            │ Graph  │ │  Vector     │
            │ (Cypher│ │  (Neo4j     │
            │  chain)│ │  vector idx)│
            └────┬──┘ └──┬─────────┘
                 └───┬───┘
               ┌──────▼──────┐
               │  Synthesis   │
               └──────┬──────┘
                  Final Answer

Everything — the graph and the vector index over unstructured event reports —
lives in a single Neo4j database (native vector index support, 5.11+).

## Repository structure
├── data/
│   └── generate_data.py      # builds the synthetic supply chain dataset (CSVs)
├── ingestion/
│   ├── schema.cypher         # constraints + vector index definition
│   ├── load_graph.py         # loads CSVs into Neo4j
│   └── embed_events.py       # embeds event descriptions for vector search
├── rag/
│   ├── graph_retriever.py    # text-to-Cypher chain (structured retrieval)
│   ├── vector_retriever.py   # semantic search over event reports
│   └── hybrid_agent.py       # LangGraph router combining both
├── eval/
│   ├── eval_questions.json   # ground-truth question set
│   ├── run_eval.py           # scores graph-only vs vector-only vs hybrid
│   └── eval_results.txt      # output of the most recent eval run
├── requirements.txt
└── .env.example

## Setup

```bash
pip install -r requirements.txt
```

### Neo4j
Requires Neo4j 5.x (native vector index support). Either Neo4j Desktop (local)
or Neo4j Aura Free (cloud) work. Run `ingestion/schema.cypher` against your
instance before loading data.

### Environment variables
Copy `.env.example` to `.env` and fill in your Neo4j credentials and an OpenAI
API key.

### Build the graph
```bash
python data/generate_data.py       # generate the CSVs
python ingestion/load_graph.py     # load into Neo4j
python ingestion/embed_events.py   # embed event reports for vector search
```

### Query it
```bash
python rag/graph_retriever.py      # structured, multi-hop questions
python rag/vector_retriever.py     # narrative/free-text questions
python rag/hybrid_agent.py         # routes automatically between both
```

### Run the evaluation
```bash
python eval/run_eval.py
```

## Evaluation results

10 hand-written questions, deliberately phrased differently from every
few-shot example used in the system's own prompts, scored against
independently-computed ground truth:

| Question type | Graph-only | Vector-only | Hybrid agent |
|---|---|---|---|
| Graph (5) | 100% | 0% | 100% |
| Vector (3) | 0% | 100% | 66.7% |
| Hybrid (2) | 75% | 50% | 100% |
| **Overall (10)** | **65%** | **40%** | **90%** |

Each single-strategy baseline is nearly blind to the other's domain; the
hybrid agent stays consistently strong across all three question types.

## Notable finding

While debugging an evaluation question, a query consistently returned an empty
result even after self-correction retries. Bisecting the generated Cypher down
to isolated fragments identified the exact cause: LangChain's
`CypherQueryCorrector` silently returns an empty string whenever a node has
inline properties but no variable name (e.g. `(:Product {name: "X"})`), even
when the query is otherwise valid. The fix — always giving such nodes an
explicit variable name — resolved it without disabling schema validation.

## Known limitations

- **Synthetic data.** Used deliberately so ground truth could be computed and
  independently verified for every eval question; a real data source (e.g. UN
  Comtrade) is a natural next step.
- **Router over-classification.** On at least one question the router
  classified a purely narrative question as "hybrid," causing an unnecessary
  graph attempt that hallucinated non-existent properties and diluted an
  otherwise-correct answer during synthesis.
- **Exact-match eval scoring.** Scoring uses substring matching, which
  under-counts correct answers phrased differently than expected (e.g. "5 to 6
  weeks" vs. the expected "5-6 week") — reported hybrid accuracy is a
  conservative lower bound.
- **Small dataset scale.** 3 products and 7 suppliers limits how many mutually
  distinguishing test questions can be written.

## Tech stack

Neo4j 5.x · LangChain (`GraphCypherQAChain`) · LangGraph · OpenAI GPT-4o-mini ·
sentence-transformers (local embeddings, no external API dependency)
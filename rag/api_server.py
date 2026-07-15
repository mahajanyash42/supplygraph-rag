"""
api_server.py

A thin FastAPI wrapper around hybrid_agent.py, so the existing agent can be
called over HTTP instead of only from a local Python shell. This file lives
alongside graph_retriever.py, vector_retriever.py, and hybrid_agent.py on
purpose -- it imports them the same way hybrid_agent.py already does
(plain sibling imports), so nothing about those files needs to change.

Run locally with:
    uvicorn api_server:app --reload

On Render, the start command should be:
    cd rag && uvicorn api_server:app --host 0.0.0.0 --port $PORT
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from hybrid_agent import build_agent

app = FastAPI(title="SupplyGraph API")

# Allow the frontend (on Vercel, or localhost during development) to call
# this API from the browser. Once you know your final Vercel domain, replace
# "*" with that exact domain for tighter security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the LangGraph agent once, when the server starts, and reuse it for
# every request -- rebuilding it per-request would be slow and wasteful.
_agent = build_agent()


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    route: str
    answer: str


@app.get("/")
def health_check():
    """Simple endpoint to confirm the server is up -- useful for Render's
    health checks and for you to sanity-check the deployment in a browser."""
    return {"status": "ok", "service": "SupplyGraph API"}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    result = _agent.invoke({"question": request.question})
    return AnswerResponse(
        route=result.get("route", "unknown"),
        answer=result.get("final_answer", ""),
    )
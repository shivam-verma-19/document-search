"""
Multi-Step Query Decomposition and Planning.

Breaks complex, comparative, or multi-part questions into focused atomic
sub-queries, retrieves context for each, and enables RAG synthesis across
all retrieved dimensions.

Example:
    Query: "Compare AWS Lambda and DynamoDB scaling models and pricing"
    Sub-queries:
      1. "AWS Lambda scaling model and concurrency limits"
      2. "AWS Lambda pricing calculation and request tiers"
      3. "DynamoDB auto-scaling and read write capacity units"
      4. "DynamoDB on-demand vs provisioned pricing model"
"""

from __future__ import annotations

import logging
import os
import re
from typing import List

from .gemini_client import GEMINI_MODEL, _get_client, classify_complexity

logger = logging.getLogger(__name__)

QUERY_DECOMPOSITION_ENABLED: bool = (
    os.getenv("QUERY_DECOMPOSITION_ENABLED", "true").lower() == "true"
)
MAX_SUB_QUERIES: int = int(os.getenv("MAX_SUB_QUERIES", "4"))

_PLANNER_PROMPT = """You are a retrieval planner for a Hybrid RAG search system.
Given the user question below, decompose it into 2 to 4 distinct, self-contained sub-queries
that retrieve the necessary information to synthesize a complete and thorough answer.

Rules:
1. Output ONLY the sub-queries, exactly one per line.
2. Do not include numbers, bullets, quotation marks, or preamble.
3. If the question cannot be broken down further, output the original question on a single line.

Question: {query}

Sub-queries:"""


def decompose_query(query: str) -> List[str]:
    """
    Decompose a complex question into atomic sub-queries for multi-step retrieval.

    If decomposition is disabled, if the question is simple, or if the LLM call fails,
    returns [query] so the pipeline degrades safely to single-pass retrieval.
    """
    if not query or not query.strip():
        return []

    clean_query = query.strip()

    if not QUERY_DECOMPOSITION_ENABLED:
        return [clean_query]

    try:
        # Check complexity first: skip decomposition for trivial questions
        classification = classify_complexity(clean_query)
        if classification.complexity == "simple":
            logger.debug(
                "Query is simple; skipping decomposition: '%s'", clean_query[:40]
            )
            return [clean_query]

        from google.genai import types

        prompt = _PLANNER_PROMPT.format(query=clean_query)
        response = _get_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=200,
                temperature=0.0,
            ),
        )

        raw_text = (response.text or "").strip()
        lines = [
            re.sub(r"^[\d\.\-\*\•\s]+", "", line).strip()
            for line in raw_text.split("\n")
            if line.strip()
        ]
        sub_queries = [line for line in lines if len(line) > 3][:MAX_SUB_QUERIES]

        if len(sub_queries) > 1:
            logger.info(
                "Decomposed query into %d sub-queries: %s",
                len(sub_queries),
                sub_queries,
            )
            return sub_queries

        logger.debug("Planner returned single query: %s", sub_queries or [clean_query])
        return sub_queries or [clean_query]

    except Exception as e:
        logger.warning("Query decomposition failed, using original query: %s", e)
        return [clean_query]

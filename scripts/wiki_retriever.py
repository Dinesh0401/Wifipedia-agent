# =============================================================================
# wiki_retriever.py  -- Real Wikipedia Retrieval via LangChain
# =============================================================================

import re
import asyncio
import logging
from typing import List, Dict, Any

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

from scripts.config import cfg

logger = logging.getLogger(__name__)


class WikipediaRetriever:
    """
    Real Wikipedia retrieval using LangChain's WikipediaQueryRun.
    Generates sub-queries from the question and deduplicates results.
    """

    def __init__(self):
        wrapper = WikipediaAPIWrapper(
            top_k_results=cfg.wiki_top_k,
            doc_content_chars_max=cfg.wiki_chars_max,
        )
        self.tool = WikipediaQueryRun(api_wrapper=wrapper)

    # -- Main retrieval entry point ------------------------------------------
    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        queries = self._build_queries(question)
        seen_titles: set = set()
        results: List[Dict[str, Any]] = []

        for query in queries:
            try:
                raw = self.tool.run(query)
                chunks = self._parse_raw_response(raw, query)
                for chunk in chunks:
                    title = chunk["title"]
                    if title not in seen_titles and chunk["content"].strip():
                        seen_titles.add(title)
                        results.append(chunk)
            except Exception as e:
                logger.warning(f"Retrieval failed for query '{query}': {e}")
                continue

        if not results:
            logger.warning(f"No Wikipedia results for: {question}")
        return results

    # -- Async wrapper -------------------------------------------------------
    async def retrieve_async(self, question: str) -> List[Dict[str, Any]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.retrieve, question)

    # -- Batch async ---------------------------------------------------------
    async def batch_retrieve(
        self, questions: List[str], concurrency: int = 3
    ) -> List[List[Dict[str, Any]]]:
        semaphore = asyncio.Semaphore(concurrency)

        async def _guarded(q: str) -> List[Dict[str, Any]]:
            async with semaphore:
                return await self.retrieve_async(q)

        return await asyncio.gather(*[_guarded(q) for q in questions])

    # -- Context builder -----------------------------------------------------
    @staticmethod
    def build_context(docs: List[Dict[str, Any]], max_chars: int = 4000) -> str:
        parts = []
        total = 0
        for doc in docs:
            block = f"TITLE: {doc['title']}\n{doc['content']}"
            if total + len(block) > max_chars:
                block = block[: max_chars - total]
            parts.append(block)
            total += len(block)
            if total >= max_chars:
                break
        return "\n\n---\n\n".join(parts)

    # -- Private helpers -----------------------------------------------------
    def _build_queries(self, question: str) -> List[str]:
        queries = [question]
        entities = self._extract_entities(question)
        for e in entities[:2]:
            if e.lower() not in question.lower()[:20]:
                queries.append(e)
        return queries

    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        seen: set = set()
        entities: List[str] = []
        for m in matches:
            if m not in seen and len(m) > 3:
                seen.add(m)
                entities.append(m)
        return entities[:3]

    @staticmethod
    def _parse_raw_response(raw: str, query: str) -> List[Dict[str, Any]]:
        results = []
        if "Page:" in raw:
            sections = re.split(r"(?=Page: )", raw)
            for section in sections:
                if not section.strip():
                    continue
                title_match = re.match(r"Page:\s*(.+?)[\n\r]", section)
                title = title_match.group(1).strip() if title_match else query
                content = section[title_match.end():].strip() if title_match else section.strip()
                results.append({
                    "title": title,
                    "content": content[:cfg.wiki_chars_max],
                    "query_used": query,
                })
        else:
            results.append({
                "title": query,
                "content": raw[:cfg.wiki_chars_max],
                "query_used": query,
            })
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    retriever = WikipediaRetriever()
    docs = retriever.retrieve("Which city is the headquarters of Oberoi Hotels?")
    for d in docs:
        print(f"\nTitle: {d['title']}")
        print(f"Content snippet: {d['content'][:200]}...")

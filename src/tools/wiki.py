"""Search/lookup tools over a HotpotQA question's bundled context paragraphs.

In the distractor setting each question ships with a list of context paragraphs,
each `[title, [sentence, ...]]`. `search[entity]` returns the matching paragraph;
`lookup[keyword]` walks the sentences of the last searched paragraph that contain
a keyword. Both tools share a small per-trajectory state object, so they are built
fresh for every task (and are therefore thread-safe across concurrent tasks).
"""
import re
import string
from typing import List

from src.tools.base import Tool


def normalize_title(s: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace (articles kept)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return re.sub(r"\s+", " ", s).strip()


def build_wiki_tools(context) -> List[Tool]:
    """Build [search, lookup] tools bound to one question's context paragraphs."""
    pages = {title: sentences for title, sentences in context}
    norm_index = {normalize_title(t): t for t in pages}
    state = {"sentences": [], "keyword": None, "idx": 0}

    def search(entity: str) -> str:
        key = normalize_title(entity)
        title = norm_index.get(key)
        if title is None:
            candidates = [t for t in pages
                          if key and (key in normalize_title(t) or normalize_title(t) in key)]
            if len(candidates) == 1:
                title = candidates[0]
            else:
                topics = ", ".join(pages.keys())
                return f"Could not find '{entity}'. Available topics: {topics}"
        state["sentences"] = pages[title]
        state["keyword"] = None
        state["idx"] = 0
        text = " ".join(pages[title]).strip()
        return f"{title}: {text[:1500]}"

    def lookup(keyword: str) -> str:
        if not state["sentences"]:
            return "No page loaded. Use search[entity] first."
        kw = keyword.lower()
        matches = [s for s in state["sentences"] if kw in s.lower()]
        if not matches:
            return f"No results for '{keyword}' in the current page."
        if state["keyword"] != keyword:
            state["keyword"] = keyword
            state["idx"] = 0
        if state["idx"] >= len(matches):
            return f"No more results for '{keyword}'."
        sentence = matches[state["idx"]]
        state["idx"] += 1
        return f"(Result {state['idx']}/{len(matches)}) {sentence.strip()}"

    return [
        Tool("search",
             "search[entity]: return the paragraph for a topic title.",
             search),
        Tool("lookup",
             "lookup[keyword]: return the next sentence containing keyword "
             "in the last searched paragraph.",
             lookup),
    ]

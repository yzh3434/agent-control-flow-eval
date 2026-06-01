import json

from src.llm import MockLLMClient
from src.tools.wiki import build_wiki_tools
from src.envs.hotpotqa import (
    normalize_answer, extract_answer, exact_match, f1_score, HotpotQAEnv,
)
from src.controllers.react import ReActController

CONTEXT = [
    ["Arthur's Magazine",
     ["Arthur's Magazine was an American literary periodical published in the 1840s.",
      "It was founded in Philadelphia."]],
    ["First for Women",
     ["First for Women is a woman's magazine published by Bauer Media Group.",
      "The magazine was started in 1989."]],
]


def _write_dataset(tmp_path):
    rows = [
        {"_id": "q1", "level": "hard", "type": "comparison",
         "question": "Which magazine was started first, Arthur's Magazine or First for Women?",
         "answer": "Arthur's Magazine", "context": CONTEXT, "supporting_facts": []},
        {"_id": "q2", "level": "easy", "type": "bridge",
         "question": "Who published First for Women?",
         "answer": "Bauer Media Group", "context": CONTEXT, "supporting_facts": []},
    ]
    f = tmp_path / "hotpot.json"
    f.write_text(json.dumps(rows), encoding="utf-8")
    return str(f)


# --- answer normalization / grading ---

def test_normalize_answer_strips_articles_punctuation_case():
    assert normalize_answer("The Arthur's Magazine.") == "arthurs magazine"
    assert normalize_answer("YES") == "yes"


def test_extract_answer_handles_answer_is_phrase():
    assert extract_answer("Let me think... The answer is Bauer Media Group.") == "Bauer Media Group"
    assert extract_answer("Arthur's Magazine") == "Arthur's Magazine"


def test_exact_match_is_normalized():
    assert exact_match("the answer is arthur's magazine", "Arthur's Magazine") is True
    assert exact_match("First for Women", "Arthur's Magazine") is False


def test_f1_gives_partial_credit_where_em_fails():
    # "Richard Nixon" vs gold "President Richard Nixon": EM fails, F1 is high
    assert exact_match("Richard Nixon", "President Richard Nixon") is False
    assert f1_score("Richard Nixon", "President Richard Nixon") > 0.7
    assert f1_score("Arthur's Magazine", "Arthur's Magazine") == 1.0
    assert f1_score("totally wrong", "Richard Nixon") == 0.0


def test_env_score_uses_f1(tmp_path):
    env = HotpotQAEnv(_write_dataset(tmp_path))
    task = env.load_tasks("hard")[0]  # gold "Arthur's Magazine"
    assert env.score(task, "Finish answer: Arthur's Magazine") == 1.0
    assert 0.0 < env.score(task, "Arthur Magazine") < 1.0


# --- environment ---

def test_load_tasks_filters_by_level(tmp_path):
    env = HotpotQAEnv(_write_dataset(tmp_path))
    hard = env.load_tasks("hard")
    easy = env.load_tasks("easy")
    assert len(hard) == 1 and hard[0].difficulty == "hard"
    assert len(easy) == 1 and easy[0].answer == "Bauer Media Group"


def test_grade_uses_exact_match(tmp_path):
    env = HotpotQAEnv(_write_dataset(tmp_path))
    task = env.load_tasks("hard")[0]
    assert env.grade(task, "Finish answer: Arthur's Magazine") is True
    assert env.grade(task, "First for Women") is False


# --- wiki tools ---

def test_search_returns_paragraph_and_lists_unknown():
    search, lookup = build_wiki_tools(CONTEXT)
    out = search.run("Arthur's Magazine")
    assert "American literary periodical" in out
    miss = search.run("Nonexistent Topic")
    assert "Could not find" in miss
    assert "First for Women" in miss  # available topics listed


def test_lookup_walks_matching_sentences():
    search, lookup = build_wiki_tools(CONTEXT)
    assert "search" in lookup.run("founded").lower() or "No page" in lookup.run("founded")
    search.run("First for Women")
    first = lookup.run("started")
    assert "1989" in first
    assert lookup.run("started").startswith("No more results")


# --- react over hotpot context (with mock llm) ---

def test_react_solves_hotpot_with_search(tmp_path):
    env = HotpotQAEnv(_write_dataset(tmp_path))
    task = env.load_tasks("hard")[0]
    client = MockLLMClient(responses=[
        "Thought: compare founding dates.\nAction: search[Arthur's Magazine]",
        "Thought: 1840s. Now the other.\nAction: search[First for Women]",
        "Thought: 1989 is later, so Arthur's Magazine was first.\n"
        "Action: Finish[Arthur's Magazine]",
    ])
    traj = ReActController(client, max_rounds=7).run(task, env)
    assert traj.success is True
    assert traj.steps[0].action == "search"
    assert "American literary periodical" in traj.steps[0].observation

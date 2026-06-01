from src.llm import ChatResult, MockLLMClient


def test_mock_client_returns_scripted_responses_and_counts_tokens():
    client = MockLLMClient(responses=["first", "second"])
    r1 = client.chat([{"role": "user", "content": "hi"}])
    r2 = client.chat([{"role": "user", "content": "again"}])
    assert isinstance(r1, ChatResult)
    assert r1.text == "first"
    assert r2.text == "second"
    assert r1.total_tokens == r1.prompt_tokens + r1.completion_tokens
    assert client.call_count == 2


def test_mock_client_raises_when_exhausted():
    client = MockLLMClient(responses=["only"])
    client.chat([{"role": "user", "content": "x"}])
    try:
        client.chat([{"role": "user", "content": "y"}])
        assert False, "expected IndexError"
    except IndexError:
        pass

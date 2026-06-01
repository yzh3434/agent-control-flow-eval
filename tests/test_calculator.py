import pytest
from src.tools.calculator import safe_eval, make_calculator


def test_safe_eval_basic_arithmetic():
    assert safe_eval("2 + 3 * 4") == 14
    assert safe_eval("(2 + 3) * 4") == 20
    assert safe_eval("10 / 4") == 2.5
    assert safe_eval("2 ** 5") == 32
    assert safe_eval("-7 + 2") == -5


def test_safe_eval_rejects_arbitrary_code():
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('echo hi')")
    with pytest.raises(ValueError):
        safe_eval("open('x')")


def test_calculator_tool_returns_string_and_handles_errors():
    tool = make_calculator()
    assert tool.name == "calculator"
    assert tool.run("3 * 7") == "21"
    out = tool.run("1 / 0")
    assert "error" in out.lower()

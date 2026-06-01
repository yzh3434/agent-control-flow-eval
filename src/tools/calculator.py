import ast
import operator

from src.tools.base import Tool

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numeric constants allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.left),
                                         _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"disallowed expression: {ast.dump(node)}")


def safe_eval(expr: str):
    """Evaluate an arithmetic expression safely (no names, calls, or imports)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as err:
        raise ValueError(f"invalid expression: {err}") from err
    return _eval_node(tree)


def _calculator(arg: str) -> str:
    try:
        result = safe_eval(arg)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
    except Exception as err:  # noqa: BLE001 - surface as observation
        return f"Error: {err}"


def make_calculator() -> Tool:
    return Tool(
        name="calculator",
        description="Evaluate an arithmetic expression, e.g. calculator[3 * (2 + 1)].",
        func=_calculator,
    )

import re

from src.controllers.base import Controller
from src.eval.trajectory import Step, Trajectory

_ACTION_RE = re.compile(r"Action:\s*(\w+)\[(.*?)\]", re.DOTALL)
_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?:\nAction:|$)", re.DOTALL)


def parse_action(text: str):
    """Return (action_name, action_input) or (None, None) if unparseable."""
    m = _ACTION_RE.search(text)
    if not m:
        return None, None
    return m.group(1), m.group(2).strip()


def _parse_thought(text: str) -> str:
    m = _THOUGHT_RE.search(text)
    return m.group(1).strip() if m else ""


def _build_system_prompt(tools) -> str:
    tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    return (
        "Solve the question using a Thought/Action loop.\n"
        "Each turn output exactly one Thought and one Action.\n"
        "Available actions:\n"
        f"{tool_lines}\n"
        "- Finish: give the final answer, e.g. Finish[42].\n"
        "Format:\nThought: <your reasoning>\nAction: <name>[<input>]"
    )


class ReActController(Controller):
    name = "react"

    def __init__(self, client, max_rounds=7):
        super().__init__(client)
        self.max_rounds = max_rounds

    def run(self, task, env):
        tool_list = env.tools(task)
        tools = {t.name: t for t in tool_list}
        messages = [
            {"role": "system", "content": _build_system_prompt(tool_list)},
            {"role": "user", "content": f"Question: {task.question}"},
        ]
        steps = []
        prompt_tokens = completion_tokens = api_calls = 0
        final_answer = None
        success = False
        score = 0.0

        for _ in range(self.max_rounds):
            result = self.client.chat(messages)
            api_calls += 1
            prompt_tokens += result.prompt_tokens
            completion_tokens += result.completion_tokens
            thought = _parse_thought(result.text)
            action, action_input = parse_action(result.text)
            messages.append({"role": "assistant", "content": result.text})

            if action is None:
                observation = "No valid Action found. Use 'Action: <name>[<input>]'."
            elif action == "Finish":
                final_answer = action_input
                success = env.grade(task, action_input)
                score = env.score(task, action_input)
                steps.append(Step(thought=thought, action="Finish",
                                  action_input=action_input, observation=""))
                break
            elif action in tools:
                observation = tools[action].run(action_input)
            else:
                observation = f"Unknown action '{action}'."

            steps.append(Step(thought=thought, action=action or "none",
                              action_input=action_input or "", observation=observation))
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        return Trajectory(
            task_id=task.id, controller=self.name, difficulty=task.difficulty,
            steps=steps, final_answer=final_answer, success=success,
            score=score, num_rounds=len(steps),
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            api_calls=api_calls,
        )

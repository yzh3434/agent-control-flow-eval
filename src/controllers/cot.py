from src.controllers.base import Controller
from src.eval.trajectory import Step, Trajectory


class CoTController(Controller):
    name = "cot"

    def run(self, task, env):
        messages = [
            {"role": "system",
             "content": "Solve the math question. Think step by step, then end "
                        "your reply with 'The answer is <number>'."},
            {"role": "user", "content": task.question},
        ]
        result = self.client.chat(messages)
        return Trajectory(
            task_id=task.id, controller=self.name, difficulty=task.difficulty,
            steps=[Step(thought=result.text, action="answer", action_input="",
                        observation=result.text)],
            final_answer=result.text,
            success=env.grade(task, result.text),
            num_rounds=1,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            api_calls=1,
        )

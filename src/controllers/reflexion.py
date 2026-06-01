from src.controllers.base import Controller
from src.controllers.react import ReActController
from src.envs.base import Task
from src.eval.trajectory import Trajectory


def _augment_with_reflections(task: Task, reflections) -> Task:
    """Return a copy of task with reflection memory appended to the question."""
    memory = "\n".join(f"- {r}" for r in reflections)
    return Task(
        id=task.id,
        question=f"{task.question}\n\nLessons from previous failed attempts:\n{memory}",
        answer=task.answer,
        difficulty=task.difficulty,
    )


class ReflexionController(Controller):
    name = "reflexion"

    def __init__(self, client, max_rounds=7, max_trials=3):
        super().__init__(client)
        self.max_rounds = max_rounds
        self.max_trials = max_trials

    def _reflect(self, task, traj):
        transcript = "\n".join(
            f"Thought: {s.thought}\nAction: {s.action}[{s.action_input}]\n"
            f"Observation: {s.observation}"
            for s in traj.steps
        )
        messages = [
            {"role": "system",
             "content": "You are reflecting on a failed solution attempt. In a few "
                        "sentences, diagnose what went wrong and how to fix it next time."},
            {"role": "user",
             "content": f"Question: {task.question}\n\nFailed attempt:\n{transcript}"},
        ]
        return self.client.chat(messages)

    def run(self, task, env):
        react = ReActController(self.client, max_rounds=self.max_rounds)
        reflections = []
        steps = []
        prompt_tokens = completion_tokens = api_calls = 0
        final_answer = None
        success = False
        score = 0.0
        trial = 0

        while trial < self.max_trials:
            trial += 1
            trial_task = _augment_with_reflections(task, reflections) if reflections else task
            traj = react.run(trial_task, env)
            steps.extend(traj.steps)
            prompt_tokens += traj.prompt_tokens
            completion_tokens += traj.completion_tokens
            api_calls += traj.api_calls
            final_answer = traj.final_answer
            success = traj.success
            score = traj.score
            if success:
                break
            if trial < self.max_trials:
                refl = self._reflect(task, traj)
                api_calls += 1
                prompt_tokens += refl.prompt_tokens
                completion_tokens += refl.completion_tokens
                reflections.append(refl.text)

        return Trajectory(
            task_id=task.id, controller=self.name, difficulty=task.difficulty,
            steps=steps, final_answer=final_answer, success=success,
            score=score, num_rounds=len(steps),
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            api_calls=api_calls, trials=trial, reflections=reflections,
        )

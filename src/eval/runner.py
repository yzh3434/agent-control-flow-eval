import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from src.controllers.base import Controller
from src.envs.base import Environment
from src.eval.trajectory import Trajectory, load_trajectory, save_trajectory


def _out_dir(results_dir: str, env: Environment, ctrl: Controller, difficulty: str) -> str:
    path = os.path.join(results_dir, env.name, ctrl.name, difficulty)
    os.makedirs(path, exist_ok=True)
    return path


def run_controller(ctrl: Controller, env: Environment, difficulty: str,
                   sample_size: int, results_dir: str,
                   concurrency: int = 4) -> List[Trajectory]:
    """Run a controller over up to `sample_size` tasks of one difficulty.

    Resumable: tasks whose trajectory JSON already exists are loaded, not re-run.
    """
    out_dir = _out_dir(results_dir, env, ctrl, difficulty)
    tasks = env.load_tasks(difficulty)[:sample_size]

    results = {}
    pending = []
    for task in tasks:
        path = os.path.join(out_dir, f"{task.id}.json")
        if os.path.exists(path):
            results[task.id] = load_trajectory(path)
        else:
            pending.append(task)

    def _work(task):
        traj = ctrl.run(task, env)
        save_trajectory(traj, os.path.join(out_dir, f"{task.id}.json"))
        return traj

    if concurrency <= 1:
        for task in pending:
            results[task.id] = _work(task)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_work, t): t for t in pending}
            for fut in as_completed(futures):
                traj = fut.result()
                results[traj.task_id] = traj

    return [results[t.id] for t in tasks]

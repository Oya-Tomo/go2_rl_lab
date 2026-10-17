"""List the Go2 environments registered by this project."""

import ast
from pathlib import Path

import go2_rl_lab.tasks as task_package
import gymnasium as gym


def discover_task_ids() -> list[str]:
    """Discover literal Gym task IDs declared in task package initializers."""
    tasks_dir = Path(task_package.__file__).resolve().parent
    task_ids: list[str] = []

    for init_path in sorted(tasks_dir.rglob("__init__.py")):
        syntax_tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
        for node in ast.walk(syntax_tree):
            if not isinstance(node, ast.Call):
                continue
            is_register_call = (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "gym"
                and node.func.attr == "register"
            )
            if not is_register_call:
                continue

            for keyword in node.keywords:
                if keyword.arg == "id" and isinstance(keyword.value, ast.Constant) and isinstance(
                    keyword.value.value, str
                ):
                    task_id = keyword.value.value
                    if task_id not in task_ids:
                        task_ids.append(task_id)

    return task_ids


def main() -> None:
    """Print registered Go2 tasks and their configuration entry points."""
    for task_id in discover_task_ids():
        task_spec = gym.spec(task_id)
        print(f"Task:        {task_spec.id}")
        print(f"Entry point: {task_spec.entry_point}")
        print(f"Config:      {task_spec.kwargs['env_cfg_entry_point']}")
        print()


if __name__ == "__main__":
    main()

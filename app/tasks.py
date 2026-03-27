# tasks.py
# This file takes the raw memory dumps from dataset.py
# and makes them available as named tasks for the environment.

from app.dataset import MEMORY_DUMPS

TASKS = {
    "easy": {
        "task_context": MEMORY_DUMPS["easy"]["task_context"],
        "token_budget": MEMORY_DUMPS["easy"]["token_budget"],
        "memory_dump": MEMORY_DUMPS["easy"]["memory_dump"],
        "gold": MEMORY_DUMPS["easy"]["gold"],
    },
    "medium": {
        "task_context": MEMORY_DUMPS["medium"]["task_context"],
        "token_budget": MEMORY_DUMPS["medium"]["token_budget"],
        "memory_dump": MEMORY_DUMPS["medium"]["memory_dump"],
        "gold": MEMORY_DUMPS["medium"]["gold"],
    },
    "hard": {
        "task_context": MEMORY_DUMPS["hard"]["task_context"],
        "token_budget": MEMORY_DUMPS["hard"]["token_budget"],
        "memory_dump": MEMORY_DUMPS["hard"]["memory_dump"],
        "gold": MEMORY_DUMPS["hard"]["gold"],
    },
}
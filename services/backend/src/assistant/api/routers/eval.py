from fastapi import APIRouter

from assistant.utils.langfuse_test import run_experiment

router = APIRouter(
    prefix="/eval",
    tags=["eval"],
)


@router.post(
    "/run_experiment",
)
async def run(experiment_name: str, dataset_name: str):
    return run_experiment(experiment_name, dataset_name)

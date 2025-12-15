from uuid import uuid4

from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from assistant.graphs.db_agent import create_db_agent
from assistant.utils.streaming import create_config, stream_graph_updates

load_dotenv()

agent = create_db_agent()

langfuse = get_client()


async def my_task(*, item, **kwargs):
    question = item.input
    thread_id = str(uuid4())
    response = ""
    async for part in stream_graph_updates(
        user_input=question,
        graph=agent,
        config=create_config(thread_id, CallbackHandler()),
    ):
        response += part
    return response


def run_experiment(experiment_name: str, dataset_name: str):
    dataset = langfuse.get_dataset(dataset_name)

    result = dataset.run_experiment(
        name=experiment_name,
        task=my_task,  # type: ignore
        max_concurrency=10,
    )

    return result.format()

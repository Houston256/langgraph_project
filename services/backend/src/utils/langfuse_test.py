from uuid import uuid4

from dotenv import load_dotenv

from utils.streaming import stream_graph_updates, create_config
from langfuse.langchain import CallbackHandler

load_dotenv()
from langfuse import get_client
from graphs.db_agent import create_db_agent
agent = create_db_agent()

langfuse = get_client()
langfuse_handler = CallbackHandler()


async def my_task(*, item, **kwargs):
    question = item.input
    thread_id = str(uuid4())
    response = ""
    async for part in stream_graph_updates(user_input=question, graph=agent, config=create_config(thread_id, langfuse_handler)):
        response += part
    print(question)
    print(100 * "-")
    print(response)
    print(100 * "=")
    return response

dataset = langfuse.get_dataset("prompts")

result = dataset.run_experiment(
    name="Production Model Test",
    task=my_task, # type: ignore
    max_concurrency=1
)

print(result.format())
from typing import AsyncGenerator

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langfuse.langchain import CallbackHandler
from langgraph.graph.state import CompiledStateGraph
from api.config import settings

compiled_state = CompiledStateGraph[MessagesState, None, MessagesState, MessagesState]


async def stream_graph_updates(user_input: str, graph: compiled_state, thread_id: str) -> AsyncGenerator[str, None]:
    langfuse_handler = CallbackHandler()

    chat_input: list[AnyMessage] = [HumanMessage(content=user_input)]

    config = RunnableConfig(
        configurable={"thread_id": thread_id},
        callbacks=[langfuse_handler],
    )

    async for chunk in graph.astream(
            MessagesState(messages=chat_input),
            stream_mode="updates",
            config=config):
        for node_name, node_state in chunk.items():
            yield node_state["messages"][-1].content


def chatbot(state: MessagesState):
    llm = init_chat_model(settings.model_name)
    return {"messages": [llm.invoke(state["messages"])]}


def create_graph() -> compiled_state:
    checkpointer = InMemorySaver()
    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile(checkpointer=checkpointer)
    return graph

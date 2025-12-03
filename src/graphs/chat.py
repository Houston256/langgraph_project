from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.state import CompiledStateGraph

from api.config import settings

compiled_state = CompiledStateGraph[MessagesState, None, MessagesState, MessagesState]

langfuse_handler = CallbackHandler()

def create_config(thread_id: str) -> RunnableConfig:
    return RunnableConfig(
        configurable={"thread_id": thread_id},
        callbacks=[langfuse_handler],
        metadata={
            "langfuse_session_id": thread_id,
        },
    )

async def chatbot(state: MessagesState, config: RunnableConfig):
    llm = init_chat_model(settings.model_name, streaming=True)
    ai_msg = await llm.ainvoke(state["messages"], config=config)
    return {"messages": [ai_msg]}


def create_graph() -> compiled_state:
    checkpointer = InMemorySaver()
    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile(checkpointer=checkpointer)
    return graph

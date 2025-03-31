import chainlit as cl
from llama_wiki_agent import StreamEvent, workflow, ctx
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scraper.database import retrieve_answers_for_prompt
from prompt.prompt import DEFAULT_WRONG_TOPIC_MESSAGE, AGENT_PROMPT
from semantic_router_guard import rl


@cl.step(type="tool")
async def tool(message: str) -> str:
    answers = retrieve_answers_for_prompt(
        index_name="paul-allen", namespace="info", query=message, result_num=2
    )
    prompt = AGENT_PROMPT.format(context=answers, question=message)
    handler = workflow.run(input=prompt, ctx=ctx)

    msg = cl.Message(content="")  # Initialize Chainlit message

    async for event in handler.stream_events():
        if isinstance(event, StreamEvent):
            chunk = event.delta
            print(chunk, end="", flush=True)  # Print in terminal

            await msg.stream_token(chunk)  # Stream partial response to UI


@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: cl.Message) -> None:
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the tool, followed by the final answer.

    Args:
        message: The user's message.

    Returns:
        None.
    """
    if rl(message.content).name == "Paul Allen":

        # Call the tool
        await tool(message.content)

    else:
        await cl.Message(content=DEFAULT_WRONG_TOPIC_MESSAGE).send()

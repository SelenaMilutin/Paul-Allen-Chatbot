import os
from dotenv import load_dotenv
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import OpenAIEncoder


# our wanted topic
politics = Route(
    name="Paul Allen",
    utterances=[
        "Who is Paul Allen",
        "Is he a good hacker?",
        "When was his Yacht launched?",
        "Paul Allen’s Philantrophy",
        "Was ship’s bell from HMS Hood successfully retrieved?",
    ],
)

# this could be used as an indicator to our chatbot to switch to a more
# conversational prompt
chitchat = Route(
    name="chitchat",
    utterances=[
        "thank you",
        "what was my last question",
        "hello",
        "have a good day",
    ],
)

# we place both of our decisions together into single list
routes = [politics, chitchat]
# or for OpenAI
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY2")
encoder = OpenAIEncoder()

rl = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")

if __name__ == "__main__":
    print(rl("is he a good man").name)
    print(rl("how's the weather today?").name)
    print(rl("I'm interested in learning about llama 2").name)

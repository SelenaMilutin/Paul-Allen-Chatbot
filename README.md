#  Paul-Allen-Chatbot

  

##  Running the Application

Follow these steps to set up and run the chatbot:

  

1.  **Create a Virtual Environment**

Refer to the [official Python documentation](https://docs.python.org/3/library/venv.html) for guidance.

  

2.  **Activate the Virtual Environment & Install Dependencies**

Run the following command to install required packages:

`pip install -r requirements.txt`

  

3.  **Set Up API Keys**

Create a `.env` file in the project's root directory and add your API keys:

```
OPENAI_API_KEY2=sk-proj-******  
PINECONE_API_KEY=pcsk_******
```

  
  

4.  **Run the Chatbot**

Start the application using Chainlit:

`chainlit run chatbot/main.py -w`

  
  

---

  

##  Chatbot Module

###  Main Components:

-  **`main.py`**: The event listener for Chainlit. The chatbot is designed to answer only questions about **Paul Allen** while allowing limited general chit-chat (e.g., greetings, thank-you messages, and simple follow-ups). The OpenAI prompt includes two relevant passages about Paul Allen to reduce hallucination.

  

-  **`llama_wiki_agent.py`**: This script contains the core workflow for generating responses. It uses **LlamaIndex’s Wikipedia agent** ([Docs](https://llamahub.ai/l/tools/llama-index-tools-wikipedia?from=tools)). The workflow also supports **context saving**, enabling the chatbot to maintain memory during conversations. The pipeline is adapted from the [LlamaIndex workflow documentation](https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/). The **OpenAI agent used is GPT-4o** ([Docs](https://docs.llamaindex.ai/en/stable/examples/llm/openai/)).

  

-  **`semantic_router_guard.py`**: This script ensures that the chatbot only responds to questions related to **Paul Allen** or a limited set of general topics. It uses [Semantic Router](https://github.com/aurelio-labs/semantic-router) for topic filtering. If a user’s question falls outside these topics, an error message is displayed instead of calling the agent.

  

---

  

##  Prompt Module

-  **`prompt.py`**: Contains the predefined prompts used by the chatbot, including:

- Error messages for off-topic questions.

- The **agent prompt**, which defines the chatbot's role and purpose.

- The response format is **not strictly defined**, as answers vary based on the input query.

  

---

  

##  Scraper Module

- The **UniversalScraper** ([Try it here](https://ai-test-hf2tjv3u6qqjxak7bemgen.streamlit.app/)) was used to:

- Parse the **Wikipedia page on Paul Allen** ([Wiki](https://en.wikipedia.org/wiki/Paul_Allen)).

- Generate text chunks.

- Upload the processed data to **Pinecone**.

  

###  Database Handling

-  **`database.py`** handles:

- Index creation.

- A **sanity check** to verify successful chunk uploads.

- Running `database.py` with the **Pinecone Quickstart guide** ([Docs](https://docs.pinecone.io/guides/get-started/quickstart)) shows the generated index structure:

  

```json

{

"dimension":  1024,

"index_fullness":  0.0,

"metric":  "cosine",

"namespaces":  {

"info":  {"vector_count":  120}

},

"total_vector_count":  120,

"vector_type":  "dense"

}
```
 

- The embedding model used is llama-text-embed-v2.

- The function retrieve_answers_for_prompt extracts relevant context for the chatbot’s prompt.
## Initialize a Python virtual environment
# In the VS Code integrated terminal (Ctrl+`):
python -m venv .venv

## Install required packages
pip install:
    dotenv 
    quart 
    quart-cors 
    llama-index 
    llama_index.llms.azure_openai
    llama_index.embeddings.azure_openai
    langchain_openai
    langgraph

## Create your file structure
# Inside your project folder:

├── .env
├── requirements.txt
├── run.py 
├── app.py
├── streaming_utils.py
├── index_utils.py
└── query_utils.py

.env
Put your Azure/OpenAI keys and any other secrets here. For example:
AZURE_OPENAI_API_KEY=…
AZURE_OPENAI_EMBEDDINGS_API_KEY=…
AZURE_OPENAI_AZURE_ENDPOINT=…
# etc.
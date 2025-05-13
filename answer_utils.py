from agent_workflow_structured_answer import optimizer_workflow, State
from config import ServerSettings, VectorIndexStore, CustomError
from query_utils import QuerySettings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate, get_response_synthesizer, VectorStoreIndex
import logging

def get_answer(
    query_settings: QuerySettings,
    server_settings: ServerSettings,
    vector_store: VectorIndexStore
) -> str:
    # 1) Try to load the requested index
    vec_name = query_settings.vectorIndex
    entry = vector_store.get(vec_name)
    if entry is None:
        # Log with %s formatting
        logging.error("Index not found: %s", vec_name)
        # Raise so the route handler can catch & return 404 JSON
        raise CustomError(
            f"Index not found, referansefilene for {vec_name} mangler!",
            404
        )

    index: VectorStoreIndex = entry.index
    vector_index_description = entry.description
    logging.info("Found entry: %s", vector_index_description)

    # 2) Build your prompt template
    text_qa_template = ChatPromptTemplate([
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You are a helpful assistant, and you will be given a user request.\n"
                "You will respond with empathy\n"
                "You will answer in language that young people aged 13 to 19 understand\n"
                "Some rules to follow:\n"
                "- Always answer the request using the given context information and not prior knowledge\n"
                "- Provide a detailed explanation, but avoid repetitions.\n"
                "- Always answer in norwegian"
            )
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=(
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Query: {query_str}\n"
                "Answer: "
            )
        ),
    ])

    # 3) Create the response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode=query_settings.response_mode,
        text_qa_template=text_qa_template,
        summary_template=text_qa_template,
        structured_answer_filtering=True,
        verbose=True,
    )

    # 4) Configure the query engine
    query_engine = index.as_query_engine(
        similarity_cutoff=query_settings.similarity_cutoff,
        similarity_top_k=query_settings.similarity_top_k,
        response_synthesizer=response_synthesizer,
    )

    # 5) Initialize and run your optimizer workflow
    init_state: State = {
        "llm": server_settings.llm,
        "query_engine": query_engine,
        "vector_index_description": vector_index_description,
        "query": query_settings.user_content,
        "similarity_cutoff": query_settings.similarity_cutoff,
        # defaults:
        "response": None,
        "answer": "",
        "lix_score": 0.0,
        "lix_category": "",
        "readable_or_not": "not readable",
        "feedback": "",
        "references": [],
        "structured_answer": "",
    }

    final_state = optimizer_workflow.invoke(init_state)

    # 6) Return the raw string
    return final_state["structured_answer"]

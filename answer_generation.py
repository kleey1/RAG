from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from retrieval_pipeline import get_retriever
from langchain_ollama import OllamaLLM

def format_docs(docs):
    """Combine retrieved documents into one context string."""
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain():
    retriever = get_retriever()

    llm = OllamaLLM(model="phi3")

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Answer using only the provided context below. "
         "If the answer is not in the context, say you do not know.\n\nContext:\n{context}"
        ),
        ("human", "{input}"),
    ])

    rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough(),
    }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def ask(query: str) -> str:
    chain = build_rag_chain()
    result = chain.invoke(query)
    return result 
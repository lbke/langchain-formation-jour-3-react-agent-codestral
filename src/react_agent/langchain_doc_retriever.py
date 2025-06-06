from langchain_chroma import Chroma
from chromadb.utils import embedding_functions
from langchain_core.documents import Document

# Chroma provides a default embedding function
# But it is not used in LangGraph => we write a quick dummy wrapper to reuse this function
# @see https://docs.trychroma.com/docs/embeddings/embedding-functions#custom-embedding-functions
# TODO: ideally, the Chroma vector store should use this function as a default
# TODO: we could add explicit errors for the async functions
class ChromaEmbeddings:
    def __init__(self):
        self.embd=embedding_functions.DefaultEmbeddingFunction()
    def embed_query(self, query):
        return self.embd([query])[0]
    def embed_documents(self,docs):
        return self.embd(docs)
embeddings=ChromaEmbeddings()

documents=[
    Document("LangChain invokes LLMs"),
    Document("LangGraph runs agents")
]

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
)
vector_store.add_documents(documents)
retriever=vector_store.as_retriever(search_kwargs={
    "k":1
})
res_langchain=retriever.invoke("What is LangChain?")[0]
res_langgraph=retriever.invoke("What is LangGraph?")[0]
print(f"Expects LangChain document, got {res_langchain}")
print(f"Expects LangGraph document, got {res_langgraph}")
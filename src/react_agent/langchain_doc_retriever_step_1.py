from langchain_chroma import Chroma
from chromadb.utils import embedding_functions
from langchain_core.documents import Document

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
res=retriever.invoke("What is LangChain?")
print(res)
res=retriever.invoke("What is LangGraph?")
print(res)
# Import this file in your agent and check the logs when running the app
# https://docs.trychroma.com/docs/overview/getting-started
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.sitemap import SitemapLoader

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="test_collection")

# NOTE: sitemap loaders can be targeted by SSRF attacks
# ideally the agent should NOT have access to "localhost"
# so it cannot be forced to request internal services
ls_docs_sitemap_loader = SitemapLoader(web_path="https://docs.smith.langchain.com/sitemap.xml", continue_on_failure=True)
ls_docs = ls_docs_sitemap_loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(ls_docs)

collection.add(
     documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    ids=["id1", "id2"]
)
results = collection.query(
    query_texts=["This is a query document about hawaii"], # Chroma will embed this for you
    n_results=2 # how many results to return
)
print(results)

from chromadb.utils import embedding_functions
embd = embedding_functions.DefaultEmbeddingFunction()

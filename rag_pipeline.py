from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

class RAGPipeline:
    def __init__(self, persist_directory="chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None

    def load_and_split(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return splitter.split_documents(docs)

    def build_vectorstore(self, chunks):
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vectorstore.persist()

    def get_relevant_context(self, query, k=3):
        if not self.vectorstore:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )

        docs = self.vectorstore.similarity_search(query, k=k)

        if not docs:
            return "No relevant information found."

        return "\n\n".join([doc.page_content for doc in docs])
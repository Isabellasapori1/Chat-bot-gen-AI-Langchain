from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Diretório onde o VectorStore foi salvo
persist_directory = "./chroma_db"

def load_vectorstore():
    """
    Carrega o VectorStore existente.
    """
    try:
        vectorstore = Chroma(
            embedding_function=OllamaEmbeddings(model="llama3"),
            persist_directory=persist_directory,
        )
        print("VectorStore carregado com sucesso!")
        return vectorstore
    except Exception as e:
        print(f"Erro ao carregar o VectorStore: {e}")
        return None

if __name__ == "__main__":
    vectorstore = load_vectorstore()
    if vectorstore:
        print("VectorStore está pronto para uso!")
    else:
        print("Falha ao carregar o VectorStore.")

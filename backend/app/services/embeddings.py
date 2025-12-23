from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import settings
from typing import List


def get_embedding_model():
    """Get embedding model based on configuration"""
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key required for OpenAI embeddings")
        return OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model="text-embedding-ada-002"
        )
    else:  # huggingface
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model
        )


async def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Create embeddings for a list of texts"""
    embedding_model = get_embedding_model()
    embeddings = embedding_model.embed_documents(texts)
    return embeddings



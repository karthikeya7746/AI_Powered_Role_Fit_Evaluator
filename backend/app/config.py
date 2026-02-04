from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # MongoDB
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "resume_matcher"
    
    # Pinecone
    pinecone_api_key: str
    pinecone_environment: str = "us-east-1"
    pinecone_index_name: str = "resume-jd-index"
    
    # OpenAI
    openai_api_key: Optional[str] = None
    
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2"
    
    # LangSmith
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "resume-matcher"
    
    # LLM Provider
    llm_provider: str = "ollama"  # openai or ollama
    
    # Embedding Provider
    embedding_provider: str = "huggingface"  # openai or huggingface
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # CORS (for Vercel frontend; comma-separated URLs, e.g. https://myapp.vercel.app)
    cors_origins: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()



from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone.vectorstores import Pinecone as PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import settings
from app.services.embeddings import get_embedding_model
from typing import List, Dict, Any
import uuid


def get_pinecone_index():
    """Initialize and return Pinecone index"""
    pc = Pinecone(api_key=settings.pinecone_api_key)
    
    # Check if index exists, create if not
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    # Determine dimension based on embedding provider
    dimension = 1536 if settings.embedding_provider == "openai" else 384
    
    if settings.pinecone_index_name not in existing_indexes:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=settings.pinecone_environment
            )
        )
    
    return pc.Index(settings.pinecone_index_name)


def get_vector_store():
    """Get LangChain Pinecone vector store"""
    index = get_pinecone_index()
    embeddings = get_embedding_model()
    
    return PineconeVectorStore(
        index=index,
        embedding=embeddings
    )


async def store_resume_chunks(resume_text: str, user_id: str, resume_id: str):
    """Split resume into chunks and store in Pinecone"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(resume_text)
    
    vector_store = get_vector_store()
    
    # Add metadata to each chunk
    metadatas = [
        {
            "type": "resume",
            "user_id": user_id,
            "resume_id": resume_id,
            "chunk_index": i,
            "text": chunk
        }
        for i, chunk in enumerate(chunks)
    ]
    
    vector_store.add_texts(
        texts=chunks,
        metadatas=metadatas,
        ids=[f"{resume_id}_chunk_{i}" for i in range(len(chunks))]
    )
    
    return len(chunks)


async def store_job_description(jd_text: str, user_id: str, jd_id: str):
    """Store job description in Pinecone"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(jd_text)
    
    vector_store = get_vector_store()
    
    metadatas = [
        {
            "type": "job_description",
            "user_id": user_id,
            "jd_id": jd_id,
            "chunk_index": i,
            "text": chunk
        }
        for i, chunk in enumerate(chunks)
    ]
    
    vector_store.add_texts(
        texts=chunks,
        metadatas=metadatas,
        ids=[f"{jd_id}_chunk_{i}" for i in range(len(chunks))]
    )
    
    return len(chunks)


async def retrieve_relevant_chunks(query: str, top_k: int = 5, filter_type: str = "resume") -> List[Dict[str, Any]]:
    """Retrieve relevant chunks from vector store"""
    vector_store = get_vector_store()
    
    # Retrieve similar chunks
    try:
        results = vector_store.similarity_search_with_score(
            query,
            k=top_k,
            filter={"type": filter_type}
        )
    except Exception:
        # If filter doesn't work, try without filter
        results = vector_store.similarity_search_with_score(query, k=top_k)
    
    return [
        {
            "text": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score) if score else 0.0
        }
        for doc, score in results
    ]


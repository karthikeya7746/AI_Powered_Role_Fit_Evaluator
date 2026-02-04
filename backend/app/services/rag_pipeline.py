from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.services.vector_store import retrieve_relevant_chunks
from app.services.llm_service import analyze_fit_score
from app.models import FitScoreResponse
from typing import List, Dict, Any
from langsmith import traceable


def _chunk_text(text: str, max_chunks: int = 10) -> List[Dict[str, Any]]:
    """Split text into chunks in memory (no Pinecone). Used for one-shot evaluate."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    # Limit so we don't overflow context
    chunks = chunks[:max_chunks]
    return [{"text": c, "score": 1.0} for c in chunks]


@traceable
async def run_rag_pipeline(
    resume_text: str,
    job_description: str,
    user_id: str = None
) -> FitScoreResponse:
    """Run the complete RAG pipeline for resume-job matching (uses Pinecone)."""
    
    # Step 1: Retrieve relevant resume chunks based on job description
    resume_chunks = await retrieve_relevant_chunks(
        query=job_description,
        top_k=5,
        filter_type="resume"
    )
    
    # Step 2: Also retrieve job description chunks to understand requirements
    jd_chunks = await retrieve_relevant_chunks(
        query=resume_text,
        top_k=3,
        filter_type="job_description"
    )
    
    # Step 3: Combine chunks for context
    all_chunks = resume_chunks + jd_chunks
    
    # Step 4: Analyze with LLM
    result = await analyze_fit_score(
        resume_text=resume_text,
        job_description=job_description,
        relevant_chunks=all_chunks
    )
    
    return result


@traceable
async def run_evaluate_in_memory(
    resume_text: str,
    job_description: str,
) -> FitScoreResponse:
    """
    Evaluate resume vs job description using only the provided texts (no Pinecone).
    Use this for one-shot /evaluate so the LLM always sees the actual resume and JD.
    """
    resume_chunks = _chunk_text(resume_text, max_chunks=12)
    jd_chunks = _chunk_text(job_description, max_chunks=5)
    all_chunks = resume_chunks + jd_chunks
    if not all_chunks:
        all_chunks = [{"text": resume_text[:8000] or "(no resume text)", "score": 1.0}]
    return await analyze_fit_score(
        resume_text=resume_text,
        job_description=job_description,
        relevant_chunks=all_chunks,
    )



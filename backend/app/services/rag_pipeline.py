from app.services.vector_store import retrieve_relevant_chunks
from app.services.llm_service import analyze_fit_score
from app.models import FitScoreResponse
from typing import List, Dict, Any
from langsmith import traceable


@traceable
async def run_rag_pipeline(
    resume_text: str,
    job_description: str,
    user_id: str = None
) -> FitScoreResponse:
    """Run the complete RAG pipeline for resume-job matching"""
    
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



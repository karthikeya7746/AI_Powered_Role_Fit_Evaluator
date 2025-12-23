from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.database import connect_to_mongo, close_mongo_connection, get_database
from app.models import ResumeUpload, JobDescription, FitScoreResponse, AnalysisResult
from app.services.pdf_parser import parse_pdf, parse_resume_file
from app.services.rag_pipeline import run_rag_pipeline
from app.services.vector_store import store_resume_chunks, store_job_description
from app.config import settings
from typing import Optional, List
import uuid
from datetime import datetime
from bson import ObjectId
import os

# Initialize LangSmith if API key is provided
if settings.langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

app = FastAPI(
    title="Resume-Job Matcher API",
    description="AI-powered resume and job description matching system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()


@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()


@app.get("/")
async def root():
    return {"message": "Resume-Job Matcher API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/upload-resume", response_model=dict)
async def upload_resume(file: UploadFile = File(...), user_id: Optional[str] = None):
    """Upload and parse resume (PDF, TXT, DOCX, or DOC)"""
    try:
        filename_lower = file.filename.lower()
        if not (filename_lower.endswith('.pdf') or 
                filename_lower.endswith('.txt') or 
                filename_lower.endswith('.docx') or 
                filename_lower.endswith('.doc')):
            raise HTTPException(
                status_code=400, 
                detail="Only PDF, TXT, DOCX, and DOC files are supported"
            )
        
        # Read file content
        content = await file.read()
        
        # Parse file based on type
        resume_text = await parse_resume_file(content, file.filename)
        
        if not resume_text:
            raise HTTPException(status_code=400, detail="Could not extract text from file")
        
        # Generate resume ID
        resume_id = str(uuid.uuid4())
        
        # Store in vector database
        await store_resume_chunks(resume_text, user_id or "anonymous", resume_id)
        
        # Store in MongoDB
        db = get_database()
        resume_doc = {
            "resume_id": resume_id,
            "user_id": user_id or "anonymous",
            "filename": file.filename,
            "text": resume_text,
            "created_at": datetime.utcnow()
        }
        await db.resumes.insert_one(resume_doc)
        
        return {
            "resume_id": resume_id,
            "filename": file.filename,
            "text_length": len(resume_text),
            "message": "Resume uploaded and processed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze", response_model=FitScoreResponse)
async def analyze_resume_job_match(
    resume_id: str,
    job_description: JobDescription
):
    """Analyze resume-job match using RAG pipeline"""
    try:
        # Get resume from database
        db = get_database()
        resume_doc = await db.resumes.find_one({"resume_id": resume_id})
        
        if not resume_doc:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        resume_text = resume_doc["text"]
        user_id = resume_doc.get("user_id", "anonymous")
        
        # Store job description in vector store
        jd_id = str(uuid.uuid4())
        await store_job_description(job_description.text, user_id, jd_id)
        
        # Store JD in MongoDB
        jd_doc = {
            "jd_id": jd_id,
            "user_id": user_id,
            "text": job_description.text,
            "created_at": datetime.utcnow()
        }
        await db.job_descriptions.insert_one(jd_doc)
        
        # Run RAG pipeline
        result = await run_rag_pipeline(
            resume_text=resume_text,
            job_description=job_description.text,
            user_id=user_id
        )
        
        # Store analysis result
        analysis_doc = AnalysisResult(
            user_id=user_id,
            resume_text=resume_text,
            job_description=job_description.text,
            fit_score=result.fit_score,
            gaps=result.gaps,
            strengths=result.strengths,
            evidence=result.evidence,
            tailored_bullets=result.tailored_bullets,
            cover_letter_snippet=result.cover_letter_snippet
        )
        
        await db.analysis_results.insert_one(analysis_doc.model_dump(by_alias=True))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results/{result_id}", response_model=AnalysisResult)
async def get_analysis_result(result_id: str):
    """Get stored analysis result"""
    try:
        db = get_database()
        result = await db.analysis_results.find_one({"_id": ObjectId(result_id)})
        
        if not result:
            raise HTTPException(status_code=404, detail="Result not found")
        
        result["_id"] = str(result["_id"])
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/user/{user_id}/results")
async def get_user_results(user_id: str, limit: int = 10):
    """Get all analysis results for a user"""
    try:
        db = get_database()
        cursor = db.analysis_results.find(
            {"user_id": user_id}
        ).sort("created_at", -1).limit(limit)
        
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        
        return {"results": results, "count": len(results)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
async def evaluate_role_fit(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    """
    Evaluate role fit - accepts resume (PDF/TXT) and job description.
    Returns simplified fit score analysis.
    """
    try:
        # Validate file type
        filename_lower = resume.filename.lower()
        if not (filename_lower.endswith('.pdf') or 
                filename_lower.endswith('.txt') or 
                filename_lower.endswith('.docx') or 
                filename_lower.endswith('.doc')):
            raise HTTPException(
                status_code=400,
                detail="Only PDF, TXT, DOCX, and DOC files are supported"
            )
        
        # Read and parse resume file
        content = await resume.read()
        resume_text = await parse_resume_file(content, resume.filename)
        
        if not resume_text:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from file"
            )
        
        if not job_description.strip():
            raise HTTPException(
                status_code=400,
                detail="Job description cannot be empty"
            )
        
        user_id = "anonymous"
        
        # Run RAG pipeline
        result = await run_rag_pipeline(
            resume_text=resume_text,
            job_description=job_description,
            user_id=user_id
        )
        
        # Generate full cover letter
        from app.services.llm_service import generate_tailored_content
        cover_letter = await generate_tailored_content(
            resume_text=resume_text,
            job_description=job_description,
            content_type="cover_letter"
        )
        
        # Transform to simplified format
        # Extract strengths as simple strings
        strengths_list = result.strengths if isinstance(result.strengths, list) else []
        
        # Extract gaps as simple strings (from requirement field)
        gaps_list = []
        if isinstance(result.gaps, list):
            for gap in result.gaps:
                if isinstance(gap, dict):
                    gaps_list.append(gap.get("requirement", str(gap)))
                else:
                    gaps_list.append(str(gap))
        
        # Extract evidence as simple strings (from resume_evidence field)
        evidence_list = []
        if isinstance(result.evidence, list):
            for ev in result.evidence:
                if isinstance(ev, dict):
                    evidence_list.append(ev.get("resume_evidence", str(ev)))
                else:
                    evidence_list.append(str(ev))
        
        return {
            "fit_score": result.fit_score,
            "strengths": strengths_list,
            "gaps": gaps_list,
            "evidence": evidence_list,
            "cover_letter": cover_letter
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



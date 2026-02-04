from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from app.config import settings
from app.models import FitScoreResponse
from langsmith import traceable
from typing import List, Dict, Any
import json


def get_llm():
    """Get LLM based on configuration"""
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key required")
        return ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model_name="gpt-4",
            temperature=0.3
        )
    else:  # ollama
        return Ollama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.3
        )


@traceable
async def analyze_fit_score(
    resume_text: str,
    job_description: str,
    relevant_chunks: List[Dict[str, Any]]
) -> FitScoreResponse:
    """Analyze resume-job fit using LLM with RAG context"""
    
    llm = get_llm()
    
    # Combine relevant chunks as context
    context = "\n\n".join([
        f"Chunk {i+1}:\n{chunk['text']}\n(Relevance: {chunk['score']:.3f})"
        for i, chunk in enumerate(relevant_chunks)
    ])
    
    # Create prompt template
    prompt_template = """You are an expert resume analyzer. Analyze how well a resume matches a job description.

Job Description:
{job_description}

Resume Context (Relevant Chunks):
{context}

Based on the job description and the resume information provided, analyze the fit and provide:
1. A fit score (0-100) - how well the resume matches the job requirements
2. Gaps - missing or partially met requirements
3. Strengths - what the candidate does well
4. Evidence - specific examples from the resume that support your analysis
5. Tailored bullets - suggested resume bullet points tailored to this job
6. Cover letter snippet - a brief paragraph highlighting the best match

Return your analysis as a single valid JSON object with exactly these keys: fit_score (number 0-100), gaps (array of objects with requirement, status, evidence), strengths (array of strings), evidence (array of objects with requirement, resume_evidence, relevance_score), tailored_bullets (array of strings), cover_letter_snippet (string).

Example shape:
{{"fit_score": 75, "gaps": [{{"requirement": "...", "status": "partial", "evidence": "..."}}], "strengths": ["..."], "evidence": [{{"requirement": "...", "resume_evidence": "...", "relevance_score": 0.9}}], "tailored_bullets": ["..."], "cover_letter_snippet": "..."}}

You must respond with ONLY the JSON object, no markdown, no code fences, no explanation before or after."""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["job_description", "context"]
    )
    
    # Format prompt
    formatted_prompt = prompt.format(
        job_description=job_description,
        context=context
    )
    
    # Get LLM response
    if settings.llm_provider == "openai":
        response = llm.invoke(formatted_prompt)
        content = getattr(response, "content", None) or str(response)
    else:  # ollama
        response = llm(formatted_prompt)
        content = response if isinstance(response, str) else str(response)
    
    if not content or not content.strip():
        raise ValueError(
            "LLM returned an empty response. Check your API key, rate limits, and that the model is available."
        )
    
    content = content.strip()
    
    # Parse JSON from response
    try:
        # Extract JSON from markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        data = json.loads(content)
        
        # Create response model
        return FitScoreResponse(**data)
    except json.JSONDecodeError as e:
        # Fallback: try to extract JSON object
        import re
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return FitScoreResponse(**data)
            except (json.JSONDecodeError, TypeError):
                pass
        raise ValueError(
            f"Failed to parse LLM response as JSON: {e}. "
            f"Response preview: {content[:500]!r}"
        )


@traceable
async def generate_tailored_content(
    resume_text: str,
    job_description: str,
    content_type: str = "bullets"  # "bullets" or "cover_letter"
) -> str:
    """Generate tailored resume bullets or cover letter"""
    
    llm = get_llm()
    
    if content_type == "bullets":
        prompt = f"""Based on this resume and job description, generate 3-5 tailored resume bullet points that highlight the best matches.

Resume:
{resume_text[:2000]}

Job Description:
{job_description[:2000]}

Generate bullet points in the format:
- [Action verb] [achievement/metric] [relevant to job requirement]

Return only the bullet points, one per line."""
    else:  # cover_letter
        prompt = f"""Write a professional, compelling cover letter that connects the resume to the job description. The cover letter should be well-structured with:
1. An engaging opening paragraph that expresses interest in the position
2. A body paragraph (or two) that highlights the most relevant skills and experiences from the resume that match the job requirements
3. A closing paragraph that expresses enthusiasm and includes a call to action

Resume:
{resume_text[:3000]}

Job Description:
{job_description[:3000]}

Write a complete cover letter (3-4 paragraphs, approximately 200-300 words) that:
- Addresses the specific requirements mentioned in the job description
- Highlights the candidate's most relevant experiences and achievements
- Uses professional, confident language
- Shows genuine interest in the role
- Is tailored specifically to this job opportunity

Format the cover letter as a proper business letter with appropriate structure."""
    
    if settings.llm_provider == "openai":
        response = llm.invoke(prompt)
        return response.content
    else:
        return llm(prompt)


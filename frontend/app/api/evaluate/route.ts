import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';

// Lazy-load Node-only libs (pdf-parse is CJS and uses fs)
async function extractTextFromFile(file: File): Promise<string> {
  const name = (file.name || '').toLowerCase();
  const buffer = Buffer.from(await file.arrayBuffer());

  if (name.endsWith('.pdf')) {
    const pdfParse = (await import('pdf-parse')).default;
    const data = await pdfParse(buffer);
    return (data?.text || '').trim();
  }
  if (name.endsWith('.docx') || name.endsWith('.doc')) {
    const mammoth = await import('mammoth');
    const result = await mammoth.extractRawText({ buffer });
    return (result?.value || '').trim();
  }
  if (name.endsWith('.txt')) {
    return buffer.toString('utf-8').trim();
  }
  throw new Error('Unsupported file type. Use PDF, TXT, or DOCX.');
}

const ANALYSIS_PROMPT = `You are an expert resume analyzer. Analyze how well a resume matches a job description.

Job Description:
{{job_description}}

Resume Context:
{{resume_context}}

Return your analysis as a single valid JSON object with exactly these keys: fit_score (number 0-100), gaps (array of objects with requirement, status, evidence), strengths (array of strings), evidence (array of objects with requirement, resume_evidence, relevance_score), tailored_bullets (array of strings), cover_letter_snippet (string).

Example shape:
{"fit_score": 75, "gaps": [{"requirement": "...", "status": "partial", "evidence": "..."}], "strengths": ["..."], "evidence": [{"requirement": "...", "resume_evidence": "...", "relevance_score": 0.9}], "tailored_bullets": ["..."], "cover_letter_snippet": "..."}

You must respond with ONLY the JSON object, no markdown, no code fences, no explanation.`;

const COVER_LETTER_PROMPT = `Write a professional, compelling cover letter (3-4 paragraphs, 200-300 words) that connects this resume to the job description. Use a proper business letter structure. Be specific and highlight the best matches.

Resume:
{{resume}}

Job Description:
{{job_description}}

Return only the cover letter text, no headings or labels.`;

export async function POST(request: NextRequest) {
  try {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      return NextResponse.json(
        { detail: 'OPENAI_API_KEY is not set' },
        { status: 500 }
      );
    }

    const formData = await request.formData();
    const resumeFile = formData.get('resume') as File | null;
    const jobDescription = (formData.get('job_description') as string) || '';

    if (!resumeFile || resumeFile.size === 0) {
      return NextResponse.json(
        { detail: 'Resume file is required' },
        { status: 400 }
      );
    }
    if (!jobDescription.trim()) {
      return NextResponse.json(
        { detail: 'Job description cannot be empty' },
        { status: 400 }
      );
    }

    const resumeText = await extractTextFromFile(resumeFile);
    if (!resumeText) {
      return NextResponse.json(
        { detail: 'Could not extract text from file' },
        { status: 400 }
      );
    }

    // Limit context size for API
    const resumeContext = resumeText.slice(0, 12000);
    const jobContext = jobDescription.slice(0, 4000);

    const openai = new OpenAI({ apiKey });

    const analysisPrompt = ANALYSIS_PROMPT
      .replace('{{job_description}}', jobContext)
      .replace('{{resume_context}}', resumeContext);

    const analysisRes = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: analysisPrompt }],
      temperature: 0.3,
    });
    let content = analysisRes.choices[0]?.message?.content?.trim() || '';
    if (!content) {
      return NextResponse.json(
        { detail: 'OpenAI returned an empty response' },
        { status: 500 }
      );
    }

    // Parse JSON from response
    if (content.includes('```json')) content = content.split('```json')[1].split('```')[0].trim();
    else if (content.includes('```')) content = content.split('```')[1].split('```')[0].trim();
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    const data = jsonMatch ? JSON.parse(jsonMatch[0]) : JSON.parse(content);

    const coverLetterPrompt = COVER_LETTER_PROMPT
      .replace('{{resume}}', resumeText.slice(0, 3000))
      .replace('{{job_description}}', jobContext);

    const coverRes = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: coverLetterPrompt }],
      temperature: 0.5,
    });
    const cover_letter = coverRes.choices[0]?.message?.content?.trim() || '';

    const strengths = Array.isArray(data.strengths) ? data.strengths : [];
    const gaps = (Array.isArray(data.gaps) ? data.gaps : []).map(
      (g: { requirement?: string } & object) =>
        typeof g === 'object' && g !== null && 'requirement' in g ? g.requirement : String(g)
    );
    const evidence = (Array.isArray(data.evidence) ? data.evidence : []).map(
      (e: { resume_evidence?: string } & object) =>
        typeof e === 'object' && e !== null && 'resume_evidence' in e ? e.resume_evidence : String(e)
    );

    return NextResponse.json({
      fit_score: typeof data.fit_score === 'number' ? data.fit_score : Number(data.fit_score) || 0,
      strengths,
      gaps,
      evidence,
      cover_letter,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Evaluation failed';
    return NextResponse.json({ detail: message }, { status: 500 });
  }
}

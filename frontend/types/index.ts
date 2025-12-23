export interface GapItem {
  requirement: string;
  status: 'missing' | 'partial';
  evidence?: string;
}

export interface EvidenceItem {
  requirement: string;
  resume_evidence: string;
  relevance_score: number;
}

export interface AnalysisResult {
  fit_score: number;
  gaps: GapItem[];
  strengths: string[];
  evidence: EvidenceItem[];
  tailored_bullets: string[];
  cover_letter_snippet?: string;
}

export interface ResumeUploadResponse {
  resume_id: string;
  filename: string;
  text_length: number;
  message: string;
}



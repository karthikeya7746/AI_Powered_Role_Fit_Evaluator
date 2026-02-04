# Resume-Job Matcher

AI-powered resume and job description matching system built with Next.js 14, FastAPI, MongoDB, Pinecone, and LangChain.

## 🏗️ Architecture

```
[ Next.js Frontend ]
   ↓  (Resume PDF + Job Description)
[ FastAPI Backend ]
   ↓  Parse PDFs (pypdf)
   ↓  Generate Embeddings (HuggingFace / OpenAI)
   ↓  Store Vectors (Pinecone)
   ↓  Retrieval via LangChain (RAG)
   ↓  LLM Response (Ollama / OpenAI)
   ↓  JSON Output → Fit Score + Gaps + Evidence
[ MongoDB ]
   ↳ Store results, user profiles, and history
[ LangSmith ]
   ↳ Trace & monitor the whole chain
```

## 🚀 Features

- **Resume Upload & Parsing**: Upload PDF resumes and extract text
- **Job Description Analysis**: Input job descriptions for matching
- **RAG Pipeline**: Retrieval-Augmented Generation for accurate matching
- **Fit Score**: AI-generated compatibility score (0-100)
- **Gap Analysis**: Identify missing or partial requirements
- **Evidence-Based Matching**: Show specific resume sections that match requirements
- **Tailored Content**: Generate customized resume bullets and cover letter snippets
- **Export**: Download analysis results as Markdown
- **LangSmith Integration**: Monitor latency, tokens, and costs

## 🛠️ Tech Stack

### Frontend
- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- React

### Backend
- FastAPI
- Python 3.11
- LangChain
- Pinecone (Vector DB)
- MongoDB
- Ollama / OpenAI (LLM)
- LangSmith (Monitoring)

## 📦 Prerequisites

- Docker & Docker Compose
- Node.js 20+
- Python 3.11+
- Pinecone API key
- (Optional) OpenAI API key
- (Optional) LangSmith API key

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd resume-matcher
```

### 2. Configure Environment Variables

Create `.env` file in the root directory:

```env
# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=resume-jd-index

# OpenAI (optional)
OPENAI_API_KEY=your_openai_api_key

# LangSmith (optional)
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=resume-matcher

# LLM Provider (openai or ollama)
LLM_PROVIDER=ollama

# Embedding Provider (openai or huggingface)
EMBEDDING_PROVIDER=huggingface
```

### 3. Start with Docker Compose

```bash
docker-compose up -d
```

This will start:
- MongoDB on port 27017
- FastAPI backend on port 8000
- Ollama on port 11434
- Next.js frontend on port 3000

### 4. Pull Ollama Model (if using Ollama)

```bash
docker exec resume-matcher-ollama ollama pull llama2
```

### 5. Access the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 🧪 Development

### Backend Development

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Development

```bash
cd frontend
npm install
npm run dev
```

## 📝 API Endpoints

### POST `/api/upload-resume`
Upload a resume PDF file.

**Request**: Multipart form data with `file` field

**Response**:
```json
{
  "resume_id": "uuid",
  "filename": "resume.pdf",
  "text_length": 1234,
  "message": "Resume uploaded and processed successfully"
}
```

### POST `/api/analyze`
Analyze resume-job match.

**Request**:
```json
{
  "resume_id": "uuid",
  "text": "Job description text..."
}
```

**Response**:
```json
{
  "fit_score": 85,
  "gaps": [
    {
      "requirement": "5 years experience",
      "status": "partial",
      "evidence": "Candidate has 3 years"
    }
  ],
  "strengths": ["Strong Python skills", "ML experience"],
  "evidence": [...],
  "tailored_bullets": [...],
  "cover_letter_snippet": "..."
}
```

### GET `/api/results/{result_id}`
Get stored analysis result.

### GET `/api/user/{user_id}/results`
Get all results for a user.

## 🐳 Docker

### Build Images

```bash
docker-compose build
```

### Run Services

```bash
docker-compose up -d
```

### View Logs

```bash
docker-compose logs -f
```

### Stop Services

```bash
docker-compose down
```

## 🔄 CI/CD

GitHub Actions workflow is configured in `.github/workflows/ci-cd.yml`:

- **On Push/Pull Request**: Runs tests for backend and frontend
- **On Push to main/release**: Builds and pushes Docker images
- **On Push to main/release**: Deploys to staging/production

## 📊 Monitoring

LangSmith integration provides:
- Request tracing
- Latency metrics
- Token usage tracking
- Cost monitoring

Access LangSmith dashboard at https://smith.langchain.com

## 🔧 Configuration

### LLM Providers

**Ollama (Local)**:
- Set `LLM_PROVIDER=ollama`
- Ensure Ollama is running and model is pulled

**OpenAI (Cloud)**:
- Set `LLM_PROVIDER=openai`
- Provide `OPENAI_API_KEY`

### Embedding Providers

**HuggingFace (Local)**:
- Set `EMBEDDING_PROVIDER=huggingface`
- Model: `sentence-transformers/all-MiniLM-L6-v2` (default)

**OpenAI (Cloud)**:
- Set `EMBEDDING_PROVIDER=openai`
- Provide `OPENAI_API_KEY`

## 🚀 Deploy to Vercel

The **frontend** can be deployed on [Vercel](https://vercel.com). The **backend** (FastAPI) must be hosted separately (e.g. [Railway](https://railway.app), [Render](https://render.com), or another service that runs Python/Docker).

### 1. Deploy the backend first

- Deploy the FastAPI app (e.g. Railway: connect repo, set **Root Directory** to `backend`, add env vars from `.env`, deploy).
- Note the backend URL (e.g. `https://your-api.railway.app`).
- Add **CORS**: set `CORS_ORIGINS` to your Vercel frontend URL (e.g. `https://your-app.vercel.app`). You can add this after the frontend is deployed.

### 2. Deploy the frontend on Vercel

1. Go to [vercel.com](https://vercel.com) → **Add New** → **Project**.
2. **Import** your Git repository (e.g. `karthikeya7746/AI_Powered_Role_Fit_Evaluator`).
3. **Root Directory**: click **Edit** and set to **`frontend`**.
4. **Environment Variables**: add  
   - `NEXT_PUBLIC_API_URL` = your backend URL (e.g. `https://your-api.railway.app`).
5. Click **Deploy**. Vercel will build and host the Next.js app.

### 3. Allow the frontend in backend CORS

On your backend host, set:

- `CORS_ORIGINS` = your Vercel URL (e.g. `https://ai-powered-role-fit-evaluator.vercel.app`).

Use a comma-separated list if you have multiple (e.g. preview + production).

After this, the Vercel app will call your backend and the full flow will work.

## 📄 License

MIT

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.



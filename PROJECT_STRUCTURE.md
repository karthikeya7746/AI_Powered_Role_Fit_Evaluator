# Project Structure

```
resume-matcher/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration & settings
│   │   ├── database.py            # MongoDB connection
│   │   ├── main.py                # FastAPI application
│   │   ├── models.py              # Pydantic models
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── embeddings.py     # Embedding generation
│   │       ├── llm_service.py     # LLM integration (Ollama/OpenAI)
│   │       ├── pdf_parser.py     # PDF text extraction
│   │       ├── rag_pipeline.py   # RAG orchestration
│   │       └── vector_store.py   # Pinecone integration
│   ├── Dockerfile
│   ├── env.example               # Environment variables template
│   └── requirements.txt          # Python dependencies
│
├── frontend/
│   ├── app/
│   │   ├── layout.tsx            # Root layout
│   │   ├── page.tsx              # Main page
│   │   └── globals.css           # Global styles
│   ├── components/
│   │   ├── AnalysisResults.tsx   # Results display component
│   │   ├── JobDescriptionInput.tsx  # JD input component
│   │   └── ResumeUpload.tsx      # Resume upload component
│   ├── types/
│   │   └── index.ts              # TypeScript type definitions
│   ├── Dockerfile
│   ├── next.config.mjs           # Next.js configuration
│   ├── package.json
│   └── tailwind.config.ts        # Tailwind CSS config
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml             # GitHub Actions CI/CD
│
├── docker-compose.yml            # Docker Compose configuration
├── .dockerignore
├── .gitignore
├── README.md                     # Main documentation
├── SETUP.md                      # Setup instructions
├── start.sh                      # Quick start script
└── PROJECT_STRUCTURE.md          # This file
```

## Key Components

### Backend (FastAPI)

**Main Application** (`app/main.py`)
- FastAPI app with CORS middleware
- API endpoints for upload, analysis, and results
- MongoDB integration
- LangSmith initialization

**Services**
- `pdf_parser.py`: Extracts text from PDF files using pypdf
- `embeddings.py`: Creates embeddings using HuggingFace or OpenAI
- `vector_store.py`: Manages Pinecone vector database operations
- `llm_service.py`: Handles LLM calls (Ollama/OpenAI) with LangSmith tracing
- `rag_pipeline.py`: Orchestrates the RAG workflow

**Models** (`app/models.py`)
- Pydantic models for request/response validation
- Database document models

### Frontend (Next.js 14)

**Pages**
- `app/page.tsx`: Main application page with state management

**Components**
- `ResumeUpload.tsx`: File upload with drag-and-drop
- `JobDescriptionInput.tsx`: Text area for job description
- `AnalysisResults.tsx`: Tabbed results display with export

**Types** (`types/index.ts`)
- TypeScript interfaces for API responses

### Infrastructure

**Docker**
- `docker-compose.yml`: Orchestrates MongoDB, Backend, Ollama, Frontend
- Backend Dockerfile: Python 3.11 with FastAPI
- Frontend Dockerfile: Node.js 20 with Next.js standalone build

**CI/CD**
- GitHub Actions workflow for testing, building, and deployment

## Data Flow

1. **Upload Resume**: PDF → Parse → Chunk → Embed → Store in Pinecone → Store metadata in MongoDB
2. **Analyze Match**: JD text → Chunk → Embed → Store in Pinecone → Retrieve relevant resume chunks → LLM analysis → Store results in MongoDB
3. **Display Results**: Fetch from MongoDB → Display in React components → Export as Markdown

## Environment Variables

See `backend/env.example` for all required and optional variables.

## API Endpoints

- `POST /api/upload-resume`: Upload resume PDF
- `POST /api/analyze`: Analyze resume-job match
- `GET /api/results/{result_id}`: Get analysis result
- `GET /api/user/{user_id}/results`: Get user's results
- `GET /health`: Health check
- `GET /docs`: API documentation (Swagger UI)



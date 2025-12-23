# Setup Guide

## Prerequisites

1. **Docker & Docker Compose** - [Install Docker](https://docs.docker.com/get-docker/)
2. **Pinecone Account** - [Sign up](https://www.pinecone.io/) and get API key
3. **(Optional) OpenAI API Key** - For cloud LLM
4. **(Optional) LangSmith API Key** - For monitoring

## Quick Start

### 1. Clone and Navigate

```bash
cd resume-matcher
```

### 2. Configure Environment

Create a `.env` file in the root directory:

```bash
cp backend/env.example .env
```

Edit `.env` and add your API keys:

```env
PINECONE_API_KEY=your_actual_pinecone_api_key
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=resume-jd-index

# Optional: For OpenAI LLM
OPENAI_API_KEY=your_openai_key
LLM_PROVIDER=openai

# Optional: For LangSmith monitoring
LANGSMITH_API_KEY=your_langsmith_key
```

### 3. Start Services

**Option A: Using the start script**
```bash
./start.sh
```

**Option B: Manual Docker Compose**
```bash
docker-compose up -d
```

### 4. Pull Ollama Model (if using Ollama)

If you're using Ollama as your LLM provider:

```bash
docker exec resume-matcher-ollama ollama pull llama2
```

Or use a different model:
```bash
docker exec resume-matcher-ollama ollama pull mistral
# Then update .env: OLLAMA_MODEL=mistral
```

### 5. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Development Setup

### Backend Development

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set environment variables
export PINECONE_API_KEY=your_key
# ... other vars

# Run server
uvicorn app.main:app --reload
```

### Frontend Development

```bash
cd frontend
npm install

# Create .env.local
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Run dev server
npm run dev
```

## Configuration Options

### LLM Providers

**Ollama (Local - Recommended for Development)**
- Free and runs locally
- Set `LLM_PROVIDER=ollama`
- Models: llama2, mistral, codellama, etc.

**OpenAI (Cloud)**
- Requires API key
- Set `LLM_PROVIDER=openai`
- Models: gpt-4, gpt-3.5-turbo

### Embedding Providers

**HuggingFace (Local - Default)**
- Free, runs locally
- Set `EMBEDDING_PROVIDER=huggingface`
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)

**OpenAI (Cloud)**
- Requires API key
- Set `EMBEDDING_PROVIDER=openai`
- Model: `text-embedding-ada-002` (1536 dimensions)

## Troubleshooting

### MongoDB Connection Issues
```bash
# Check if MongoDB is running
docker ps | grep mongodb

# View MongoDB logs
docker-compose logs mongodb
```

### Pinecone Index Creation
- The index is created automatically on first run
- Ensure your Pinecone API key has permission to create indexes
- Check Pinecone dashboard for index status

### Ollama Model Not Found
```bash
# List available models
docker exec resume-matcher-ollama ollama list

# Pull a model
docker exec resume-matcher-ollama ollama pull llama2
```

### Frontend Can't Connect to Backend
- Ensure backend is running on port 8000
- Check `NEXT_PUBLIC_API_URL` in frontend `.env.local`
- Check CORS settings in backend `app/main.py`

## Production Deployment

### Using Docker Compose

1. Update environment variables in `.env`
2. Build images:
   ```bash
   docker-compose build
   ```
3. Run in production mode:
   ```bash
   docker-compose up -d
   ```

### Using Cloud Platforms

**Render**
- Connect GitHub repository
- Set environment variables in Render dashboard
- Deploy backend and frontend as separate services

**AWS ECS/Fargate**
- Build and push Docker images to ECR
- Create ECS task definitions
- Set up load balancer and auto-scaling

## Monitoring

### LangSmith Integration

1. Sign up at [LangSmith](https://smith.langchain.com)
2. Get your API key
3. Add to `.env`: `LANGSMITH_API_KEY=your_key`
4. View traces in LangSmith dashboard

### Health Checks

- Backend: `GET http://localhost:8000/health`
- Frontend: Check browser console for errors

## Next Steps

1. Upload a resume PDF
2. Paste a job description
3. Click "Analyze Match"
4. Review the fit score, gaps, and tailored content
5. Export results as Markdown



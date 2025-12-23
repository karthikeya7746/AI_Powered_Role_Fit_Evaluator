# Next Steps - Getting Started

## Step 1: Set Up Environment Variables

Create a `.env` file in the root directory with your API keys:

```bash
cd /Users/karthikeyareddy/Downloads/resume-matcher
cp backend/env.example .env
```

Then edit `.env` and add your **Pinecone API key** (required):

```env
PINECONE_API_KEY=your_actual_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=resume-jd-index
```

**Get a Pinecone API key:**
1. Go to https://www.pinecone.io/
2. Sign up for a free account
3. Create a new project
4. Copy your API key from the dashboard

## Step 2: Start the Services

### Option A: Using the start script (Recommended)
```bash
./start.sh
```

### Option B: Using Docker Compose directly
```bash
docker-compose up -d
```

This will start:
- ✅ MongoDB (port 27017)
- ✅ FastAPI Backend (port 8000)
- ✅ Ollama (port 11434)
- ✅ Next.js Frontend (port 3000)

## Step 3: Pull Ollama Model (if using Ollama)

If you're using Ollama as your LLM provider (default), pull a model:

```bash
docker exec resume-matcher-ollama ollama pull llama2
```

This may take a few minutes depending on your internet speed.

**Alternative models you can use:**
```bash
docker exec resume-matcher-ollama ollama pull mistral
docker exec resume-matcher-ollama ollama pull codellama
```

Then update `.env` if using a different model:
```env
OLLAMA_MODEL=mistral
```

## Step 4: Verify Services Are Running

Check that all containers are up:
```bash
docker-compose ps
```

You should see 4 services running:
- resume-matcher-mongodb
- resume-matcher-backend
- resume-matcher-ollama
- resume-matcher-frontend

## Step 5: Test the Application

### Access the Frontend
Open your browser and go to:
```
http://localhost:3000
```

### Test the Backend API
Visit the API documentation:
```
http://localhost:8000/docs
```

Or test the health endpoint:
```bash
curl http://localhost:8000/health
```

## Step 6: Use the Application

1. **Upload a Resume**
   - Click "Upload Resume" on the frontend
   - Select a PDF file
   - Wait for upload confirmation

2. **Enter Job Description**
   - Paste a job description in the text area
   - Click "Analyze Match"

3. **View Results**
   - See the fit score (0-100)
   - Review gaps and strengths
   - Check evidence and tailored content
   - Export results as Markdown

## Troubleshooting

### If services won't start:
```bash
# Check logs
docker-compose logs

# Restart services
docker-compose restart

# Rebuild if needed
docker-compose up -d --build
```

### If Pinecone connection fails:
- Verify your API key is correct
- Check your Pinecone dashboard for index status
- Ensure you have internet connection

### If Ollama model not found:
```bash
# List available models
docker exec resume-matcher-ollama ollama list

# Pull a model
docker exec resume-matcher-ollama ollama pull llama2
```

### If frontend can't connect to backend:
- Check that backend is running: `curl http://localhost:8000/health`
- Verify `NEXT_PUBLIC_API_URL` in frontend (should be `http://localhost:8000`)

## Optional: Configure Additional Services

### Use OpenAI Instead of Ollama

Edit `.env`:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
```

### Enable LangSmith Monitoring

Edit `.env`:
```env
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=resume-matcher
```

Get your key at: https://smith.langchain.com

### Use OpenAI Embeddings

Edit `.env`:
```env
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
```

## Development Mode

### Run Backend Locally (without Docker)
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set environment variables
export PINECONE_API_KEY=your_key
# ... other vars

uvicorn app.main:app --reload
```

### Run Frontend Locally (without Docker)
```bash
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev
```

## View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f mongodb
```

## Stop Services

```bash
docker-compose down
```

## Next Steps After Setup

1. ✅ Upload your first resume
2. ✅ Test with a real job description
3. ✅ Review the analysis results
4. ✅ Export and use the tailored content
5. 🔄 Iterate and improve your resume based on gaps

## Need Help?

- Check `README.md` for full documentation
- Check `SETUP.md` for detailed setup instructions
- Check `PROJECT_STRUCTURE.md` for code organization
- View API docs at http://localhost:8000/docs



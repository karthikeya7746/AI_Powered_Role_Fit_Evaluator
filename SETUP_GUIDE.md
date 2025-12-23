# Step-by-Step Setup Guide

## ✅ Step 1: Environment File Created

I've created your `.env` file. Now you need to add your API keys.

## 🔑 Step 2: Get Your Pinecone API Key (REQUIRED)

1. **Go to Pinecone**: https://www.pinecone.io/
2. **Sign up** for a free account (or log in if you have one)
3. **Create a new project** (or use existing)
4. **Get your API key**:
   - Go to your project dashboard
   - Click on "API Keys" in the sidebar
   - Copy your API key (starts with something like `pc-...`)

## ✏️ Step 3: Edit Your .env File

Open the `.env` file in the project root and replace the placeholder:

```bash
# Open in your editor
open /Users/karthikeyareddy/Downloads/resume-matcher/.env
```

Or edit it directly:
```bash
nano /Users/karthikeyareddy/Downloads/resume-matcher/.env
```

**Replace this line:**
```
PINECONE_API_KEY=your_pinecone_api_key
```

**With your actual key:**
```
PINECONE_API_KEY=pc-xxxxxxxxxxxxxxxxxxxxx
```

**Minimum required configuration:**
- ✅ `PINECONE_API_KEY` - Your actual Pinecone API key
- ✅ `PINECONE_ENVIRONMENT` - Usually `us-east-1` (already set)
- ✅ `PINECONE_INDEX_NAME` - `resume-jd-index` (already set)

**Optional (you can add later):**
- `OPENAI_API_KEY` - If you want to use OpenAI instead of Ollama
- `LANGSMITH_API_KEY` - For monitoring (get from https://smith.langchain.com)

## 🐳 Step 4: Start Docker Services

Once you've added your Pinecone API key, run:

```bash
cd /Users/karthikeyareddy/Downloads/resume-matcher
docker-compose up -d
```

This will start:
- MongoDB (database)
- FastAPI Backend (API server)
- Ollama (local LLM)
- Next.js Frontend (web app)

**Wait about 30 seconds** for all services to start.

## 📦 Step 5: Pull Ollama Model

Ollama needs a model to run. Pull one:

```bash
docker exec resume-matcher-ollama ollama pull llama2
```

This downloads the model (about 3.8GB). It may take 5-10 minutes depending on your internet.

**Alternative models (smaller/faster):**
```bash
# Mistral (smaller, faster)
docker exec resume-matcher-ollama ollama pull mistral

# Then update .env: OLLAMA_MODEL=mistral
```

## ✅ Step 6: Verify Everything is Running

Check all services:
```bash
docker-compose ps
```

You should see 4 services with "Up" status:
- resume-matcher-mongodb
- resume-matcher-backend
- resume-matcher-ollama
- resume-matcher-frontend

Test the backend:
```bash
curl http://localhost:8000/health
```

Should return: `{"status":"healthy"}`

## 🌐 Step 7: Access the Application

Open your browser and go to:
- **Frontend**: http://localhost:3000
- **Backend API Docs**: http://localhost:8000/docs

## 🧪 Step 8: Test the Application

1. **Upload a Resume**:
   - Click "Upload Resume"
   - Select a PDF file
   - Wait for "✓ filename.pdf" confirmation

2. **Enter Job Description**:
   - Paste a job description in the text area
   - Click "Analyze Match"

3. **View Results**:
   - See fit score (0-100)
   - Review gaps, strengths, evidence
   - Export as Markdown

## 🔧 Troubleshooting

### Services won't start?
```bash
# Check logs
docker-compose logs

# Restart
docker-compose restart

# Rebuild
docker-compose up -d --build
```

### Backend errors?
```bash
# Check backend logs
docker-compose logs backend

# Common issue: Missing Pinecone API key
# Make sure .env has PINECONE_API_KEY set
```

### Ollama model not found?
```bash
# List available models
docker exec resume-matcher-ollama ollama list

# Pull a model
docker exec resume-matcher-ollama ollama pull llama2
```

### Frontend can't connect?
- Make sure backend is running: `curl http://localhost:8000/health`
- Check browser console for errors
- Verify ports 3000 and 8000 are not in use

## 📝 Quick Command Reference

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Restart a service
docker-compose restart backend

# Check service status
docker-compose ps

# Pull Ollama model
docker exec resume-matcher-ollama ollama pull llama2
```

## 🎯 Current Status

- ✅ Project structure created
- ✅ .env file created (needs your Pinecone API key)
- ✅ Docker installed and ready
- ⏳ Waiting for: Pinecone API key
- ⏳ Next: Start services

## Need Help?

If you get stuck:
1. Check the logs: `docker-compose logs`
2. Verify your `.env` file has the correct API key
3. Make sure Docker is running: `docker ps`
4. Check `README.md` for more details



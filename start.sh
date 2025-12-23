#!/bin/bash

echo "🚀 Starting Resume-Job Matcher..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp backend/env.example .env
    echo "📝 Please edit .env file with your API keys before continuing."
    echo "   Required: PINECONE_API_KEY"
    echo "   Optional: OPENAI_API_KEY, LANGSMITH_API_KEY"
    read -p "Press enter to continue after updating .env..."
fi

# Start Docker Compose
echo "🐳 Starting Docker containers..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check if Ollama model is needed
if grep -q "LLM_PROVIDER=ollama" .env 2>/dev/null; then
    echo "📦 Checking Ollama model..."
    if ! docker exec resume-matcher-ollama ollama list | grep -q llama2; then
        echo "📥 Pulling llama2 model (this may take a while)..."
        docker exec resume-matcher-ollama ollama pull llama2
    fi
fi

echo "✅ Services started!"
echo ""
echo "📍 Access points:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "📊 View logs: docker-compose logs -f"
echo "🛑 Stop services: docker-compose down"



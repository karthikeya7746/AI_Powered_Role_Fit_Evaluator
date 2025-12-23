#!/bin/bash

echo "🚀 Resume-Job Matcher - Quick Start"
echo "===================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp backend/env.example .env
    echo "✅ Created .env file"
    echo ""
    echo "📝 IMPORTANT: Edit .env and add your PINECONE_API_KEY"
    echo "   File location: $(pwd)/.env"
    echo ""
    read -p "Press Enter after you've added your Pinecone API key..."
fi

# Check if Pinecone API key is set
if grep -q "PINECONE_API_KEY=your_pinecone_api_key" .env 2>/dev/null || grep -q "PINECONE_API_KEY=$" .env 2>/dev/null; then
    echo "⚠️  WARNING: PINECONE_API_KEY not set in .env"
    echo "   Please edit .env and add your Pinecone API key"
    echo "   Get one at: https://www.pinecone.io/"
    echo ""
    read -p "Press Enter after adding your API key, or Ctrl+C to exit..."
fi

echo "🐳 Starting Docker services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to start (30 seconds)..."
sleep 30

echo ""
echo "🔍 Checking service status..."
docker-compose ps

echo ""
echo "📦 Checking Ollama models..."
if docker exec resume-matcher-ollama ollama list 2>/dev/null | grep -q llama2; then
    echo "✅ Ollama model 'llama2' is available"
else
    echo "⚠️  Ollama model not found. Pulling llama2 (this may take 5-10 minutes)..."
    echo "   You can cancel and use a different model if needed"
    docker exec resume-matcher-ollama ollama pull llama2
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📍 Access points:"
echo "   Frontend:  http://localhost:3000"
echo "   Backend:   http://localhost:8000"
echo "   API Docs:  http://localhost:8000/docs"
echo ""
echo "📊 View logs: docker-compose logs -f"
echo "🛑 Stop:     docker-compose down"



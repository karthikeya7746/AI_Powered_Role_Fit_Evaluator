# Quick Reference: AI Components Cheat Sheet

## 🚀 HuggingFace Embeddings - Quick Start

```python
# Install
pip install sentence-transformers

# Basic Usage
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
texts = ["Python developer", "FastAPI expert"]
embeddings = model.encode(texts)

# Calculate similarity
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print(f"Similarity: {similarity:.3f}")  # 0.0 to 1.0
```

## 🔗 LangChain - Quick Start

```python
# Install
pip install langchain langchain-openai langchain-community

# LLM
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
response = llm.invoke("Hello!")
print(response.content)

# Prompts
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(
    template="Write about {topic}",
    input_variables=["topic"]
)
formatted = prompt.format(topic="AI")
response = llm.invoke(formatted)

# Text Splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_text(long_text)

# Vector Store (Pinecone)
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()
vectorstore = PineconeVectorStore.from_existing_index(
    index_name="my-index",
    embedding=embeddings
)

# Add texts
vectorstore.add_texts(["Python developer"], metadatas=[{"type": "skill"}])

# Search
results = vectorstore.similarity_search("backend developer", k=3)
```

## 🎯 RAG Pipeline - Quick Start

```python
# Complete RAG Example
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# 1. Setup
llm = ChatOpenAI()
embeddings = HuggingFaceEmbeddings()
vectorstore = PineconeVectorStore.from_existing_index(
    index_name="resume-index",
    embedding=embeddings
)

# 2. Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)

# 3. Query
answer = qa_chain.run("What skills does this resume highlight?")
print(answer)
```

## 📊 Your Project's Flow - Visualized

```
┌─────────────────┐
│  Resume Upload  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Parse PDF/DOCX │  ← pypdf, python-docx
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Chunking  │  ← LangChain RecursiveCharacterTextSplitter
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Generate       │  ← HuggingFaceEmbeddings
│  Embeddings     │     (384-dim vectors)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Store Vectors  │  ← Pinecone (via LangChain)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  User Query     │
│  (Job Desc)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Similarity     │  ← Cosine similarity search
│  Search         │     (Find top 5 chunks)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Format Context │  ← Combine chunks with metadata
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Analysis   │  ← OpenAI GPT-4 (via LangChain)
│  with Context   │     PromptTemplate + ChatOpenAI
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Parse JSON     │  ← Extract structured response
│  Response       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Return Result  │  ← Fit score, strengths, gaps
└─────────────────┘
```

## 🔑 Key Concepts

### Embeddings
- **What**: Numerical representation of text
- **Why**: Enables semantic search (meaning, not keywords)
- **Dimension**: 384 for MiniLM, 768 for MPNet
- **Similarity**: Cosine similarity (0.0 = different, 1.0 = identical)

### Text Chunking
- **Why**: LLMs have token limits, better retrieval
- **Size**: 1000 chars (balance between context and precision)
- **Overlap**: 200 chars (maintains context between chunks)

### Vector Search
- **How**: Cosine similarity between query and document vectors
- **Top-K**: Retrieve top K most similar chunks
- **Filtering**: Use metadata to filter results

### RAG
- **Retrieve**: Find relevant context from vector store
- **Augment**: Add context to prompt
- **Generate**: LLM generates answer with context

## 📝 Common Patterns

### Pattern 1: Simple Embedding Search
```python
model = SentenceTransformer('all-MiniLM-L6-v2')
query_emb = model.encode("Python developer")
doc_embs = model.encode(documents)
similarities = cosine_similarity([query_emb], doc_embs)
```

### Pattern 2: LangChain RAG
```python
retriever = vectorstore.as_retriever()
chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
answer = chain.run(question)
```

### Pattern 3: Custom RAG (Your Project)
```python
chunks = retrieve_relevant_chunks(query, top_k=5)
context = format_chunks(chunks)
prompt = create_prompt(context, query)
response = llm.invoke(prompt)
result = parse_json(response)
```

## 🐛 Debugging Tips

### Embeddings Not Working?
```python
# Check embedding shape
print(embeddings.shape)  # Should be (num_texts, 384)

# Check similarity range
print(f"Min: {similarities.min()}, Max: {similarities.max()}")
# Should be between -1 and 1 (cosine similarity)
```

### RAG Not Finding Relevant Chunks?
```python
# Test retrieval directly
results = vectorstore.similarity_search_with_score(query, k=5)
for doc, score in results:
    print(f"Score: {score:.3f} - {doc.page_content[:100]}")
```

### LLM Not Following Format?
```python
# Add explicit instructions
prompt = """You MUST return valid JSON:
{
    "fit_score": <number>,
    "strengths": ["..."],
    ...
}
"""
```

## 📚 Model Comparison

| Model | Dimensions | Size | Speed | Quality |
|-------|-----------|------|-------|---------|
| all-MiniLM-L6-v2 | 384 | 22MB | Fast | Good |
| all-mpnet-base-v2 | 768 | 420MB | Medium | Better |
| text-embedding-ada-002 | 1536 | API | Fast | Best |

## 🎯 Performance Tips

1. **Batch Processing**: Process multiple texts at once
2. **Caching**: Cache embeddings for repeated queries
3. **Chunk Size**: Experiment with 500-2000 chars
4. **Top-K**: Start with 5, increase if needed
5. **Normalize**: Always normalize embeddings for cosine similarity




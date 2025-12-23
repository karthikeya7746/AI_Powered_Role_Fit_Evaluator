# Complete Project Explanation: Resume Matcher (A-Z Guide for Beginners)

## 📚 Table of Contents
1. [What This Project Does](#what-this-project-does)
2. [Project Structure Overview](#project-structure-overview)
3. [Backend Files Explained](#backend-files-explained)
4. [Frontend Files Explained](#frontend-files-explained)
5. [Configuration Files](#configuration-files)
6. [How Everything Works Together](#how-everything-works-together)
7. [Key Concepts for Beginners](#key-concepts-for-beginners)

---

## What This Project Does

**Resume Matcher** is an AI-powered web application that:
- Takes your resume (PDF, DOCX, or TXT file) and a job description
- Uses AI to analyze how well your resume matches the job requirements
- Gives you a **fit score** (0-100%), shows your **strengths**, identifies **gaps**, and provides **evidence** for why you're a good match

Think of it like a smart assistant that reads both documents and tells you: "You're 85% qualified! Here's what you're great at, and here's what you might want to improve."

---

## Project Structure Overview

The project is split into two main parts:

```
resume-matcher/
├── backend/          # The "brain" - Python code that does the AI analysis
├── frontend/         # The "face" - Web interface users interact with
├── docker-compose.yml # Instructions for running everything together
└── Various .md files # Documentation
```

**Think of it like a restaurant:**
- **Frontend** = The dining room (what customers see)
- **Backend** = The kitchen (where the work happens)
- **Docker** = The building that houses everything

---

## Backend Files Explained

The backend is written in **Python** and uses **FastAPI** (a web framework). Let's go through each file:

### 📁 `backend/app/main.py` - The Main Controller

**What it does:** This is the "traffic director" of your application. It receives requests from the frontend and routes them to the right place.

**Key parts:**
- **`@app.post("/api/upload-resume")`** - When you upload a resume, this function handles it
- **`@app.post("/api/analyze")`** - When you click "Analyze", this function runs the matching process
- **`@app.post("/evaluate")`** - A simpler endpoint that does everything in one step
- **CORS middleware** - Allows the frontend (running on port 3000) to talk to the backend (port 8000)

**In simple terms:** This file is like a receptionist who takes your request and sends it to the right department.

---

### 📁 `backend/app/config.py` - Settings Manager

**What it does:** Stores all the configuration settings for your app (like API keys, database URLs, etc.)

**Key settings:**
- `mongodb_uri` - Where to find the database
- `pinecone_api_key` - Your Pinecone API key (for storing vectors)
- `openai_api_key` - Optional OpenAI key
- `ollama_base_url` - Where Ollama (local AI) is running
- `llm_provider` - Which AI to use: "ollama" (local) or "openai" (cloud)

**In simple terms:** This is like a settings file on your phone - it stores all your preferences and passwords.

---

### 📁 `backend/app/models.py` - Data Structures

**What it does:** Defines the "shapes" of data that flow through your application.

**Key models:**
- **`ResumeUpload`** - What data comes in when uploading a resume
- **`JobDescription`** - What data comes in with a job description
- **`FitScoreResponse`** - What data goes out after analysis (fit score, gaps, strengths, etc.)
- **`AnalysisResult`** - The complete analysis stored in the database

**In simple terms:** These are like forms - they define what information you need to fill in and what format it should be in.

---

### 📁 `backend/app/database.py` - Database Connection

**What it does:** Handles connecting to and disconnecting from MongoDB (the database that stores resumes, job descriptions, and results).

**Key functions:**
- `connect_to_mongo()` - Opens connection to database when app starts
- `close_mongo_connection()` - Closes connection when app shuts down
- `get_database()` - Returns the database object so other files can use it

**In simple terms:** This is like a phone operator who connects you to the database "phone line" and hangs up when you're done.

---

### 📁 `backend/app/services/pdf_parser.py` - File Reader

**What it does:** Extracts text from different file formats (PDF, DOCX, TXT).

**Key functions:**
- `parse_pdf()` - Reads text from PDF files using `pypdf` library
- `parse_docx()` - Reads text from Word documents using `python-docx` library
- `parse_resume_file()` - Smart function that detects file type and calls the right parser

**In simple terms:** This is like a translator that can read different languages (file formats) and convert them all to plain text.

**Example:**
```python
# If you upload "resume.pdf", it:
# 1. Opens the PDF
# 2. Reads all the text from every page
# 3. Returns: "John Doe\nSoftware Engineer\n5 years experience..."
```

---

### 📁 `backend/app/services/embeddings.py` - Text to Numbers Converter

**What it does:** Converts text into numbers (called "embeddings") that computers can understand and compare.

**Why this matters:** Computers can't read text like humans. They need numbers. Embeddings convert "Python programming" into something like `[0.23, -0.45, 0.67, ...]` (a list of 384 or 1536 numbers).

**Key functions:**
- `get_embedding_model()` - Chooses which embedding service to use (HuggingFace or OpenAI)
- `create_embeddings()` - Converts a list of text strings into embeddings

**In simple terms:** This is like converting words into a secret code (numbers) that represents the meaning. Similar words get similar codes.

**Example:**
- "Python programming" → `[0.23, -0.45, 0.67, ...]`
- "Software development" → `[0.25, -0.43, 0.65, ...]` (similar numbers = similar meaning!)

---

### 📁 `backend/app/services/vector_store.py` - Smart Storage

**What it does:** Stores text chunks as embeddings in Pinecone (a vector database) and retrieves similar chunks.

**Key concepts:**
- **Chunking:** Splits long text into smaller pieces (1000 characters each)
- **Vector Store:** Pinecone database that stores embeddings
- **Similarity Search:** Finds text chunks that are similar to a query

**Key functions:**
- `store_resume_chunks()` - Splits resume into chunks, converts to embeddings, stores in Pinecone
- `store_job_description()` - Same for job descriptions
- `retrieve_relevant_chunks()` - Searches for chunks similar to a query

**In simple terms:** 
- Imagine you have a library with thousands of books
- Instead of searching by title, you search by "meaning"
- If you search for "Python developer", it finds all resumes that mention Python, coding, software, etc.

**Example flow:**
```
Resume text: "I'm a Python developer with 5 years experience..."
↓
Split into chunks: ["I'm a Python developer", "with 5 years experience", ...]
↓
Convert to embeddings: [[0.23, -0.45, ...], [0.34, -0.12, ...], ...]
↓
Store in Pinecone with metadata: {type: "resume", user_id: "123", ...}
```

---

### 📁 `backend/app/services/llm_service.py` - The AI Brain

**What it does:** Uses Large Language Models (LLMs) like Ollama or OpenAI to analyze resumes and generate insights.

**Key functions:**
- `get_llm()` - Chooses which AI to use (Ollama for local, OpenAI for cloud)
- `analyze_fit_score()` - The main function that analyzes resume vs job description
- `generate_tailored_content()` - Creates custom resume bullets or cover letter snippets

**How `analyze_fit_score()` works:**
1. Takes the resume text, job description, and relevant chunks from vector store
2. Creates a detailed prompt (instructions) for the AI
3. Sends everything to the LLM
4. LLM returns JSON with fit score, gaps, strengths, evidence, etc.
5. Parses the JSON and returns a structured response

**In simple terms:** This is like having a very smart assistant read both documents and write a detailed analysis report.

**Example prompt sent to AI:**
```
You are an expert resume analyzer. Analyze how well a resume matches a job description.

Job Description:
[The job description text]

Resume Context:
[Relevant chunks from the resume]

Based on this, provide:
1. Fit score (0-100)
2. Gaps (missing requirements)
3. Strengths
4. Evidence
...
```

---

### 📁 `backend/app/services/rag_pipeline.py` - The Orchestrator

**What it does:** Coordinates the entire RAG (Retrieval-Augmented Generation) process.

**RAG explained:** Instead of just sending the full resume to the AI (which might be too long), RAG:
1. **Retrieves** the most relevant parts of the resume based on the job description
2. **Augments** the AI's knowledge with these relevant chunks
3. **Generates** a response using both the job description and relevant resume parts

**Key function:**
- `run_rag_pipeline()` - The main function that:
  1. Retrieves relevant resume chunks (using job description as query)
  2. Retrieves relevant job description chunks (using resume as query)
  3. Combines them
  4. Sends everything to the LLM for analysis
  5. Returns the result

**In simple terms:** This is like a research assistant who:
- Reads the job description
- Finds the most relevant parts of your resume
- Highlights them
- Gives everything to the AI analyst
- Returns the final report

**Flow:**
```
Job Description: "Looking for Python developer with 5 years experience"
↓
Search Pinecone for resume chunks similar to "Python developer 5 years"
↓
Find: "I'm a Python developer with 5 years experience in web development"
↓
Send to AI: "Here's the job description and the most relevant parts of the resume"
↓
AI analyzes and returns: {fit_score: 85, strengths: [...], gaps: [...]}
```

---

## Frontend Files Explained

The frontend is written in **TypeScript/React** using **Next.js 14**. Let's go through each file:

### 📁 `frontend/app/page.tsx` - The Main Page

**What it does:** This is the main page users see when they visit your website. It contains the entire user interface.

**Key parts:**
- **State management:** Uses React hooks (`useState`) to track:
  - `resumeFile` - The uploaded resume
  - `jobDescription` - The job description text
  - `loading` - Whether analysis is in progress
  - `result` - The analysis results
  - `error` - Any error messages

- **`handleFileSelect()`** - Called when user uploads a file
- **`handleEvaluate()`** - Called when user clicks "Evaluate Resume"
  - Creates a FormData object
  - Sends POST request to `/evaluate` endpoint
  - Updates state with results
- **`handleDownloadReport()`** - Creates a text file with the results and downloads it

**In simple terms:** This is like the main control panel of your app - it shows everything, handles user clicks, and displays results.

**Visual structure:**
```
┌─────────────────────────────────┐
│   Role Fit Evaluator (Header)   │
├─────────────────────────────────┤
│  [Upload Resume Button]         │
│  [Job Description Text Area]     │
│  [Evaluate Resume Button]       │
├─────────────────────────────────┤
│  [Fit Score: 85%]               │
│  [Strengths] [Gaps]              │
│  [Evidence Panel]                │
│  [Download Report Button]       │
└─────────────────────────────────┘
```

---

### 📁 `frontend/components/FileUpload.tsx` - File Upload Component

**What it does:** A reusable component that handles file uploads with drag-and-drop functionality.

**Key features:**
- **Drag and drop:** Users can drag files onto the component
- **Click to upload:** Traditional file picker
- **File validation:** Checks if file type is allowed (.pdf, .txt, .docx, .doc)
- **Visual feedback:** Shows different states (dragging, selected, disabled)
- **File removal:** Users can remove selected file

**Key functions:**
- `handleDragEnter/Leave/Over/Drop()` - Handles drag-and-drop events
- `handleFileInput()` - Handles traditional file selection
- `isValidFileType()` - Checks if file extension is allowed

**In simple terms:** This is like a fancy file upload box that looks nice and is easy to use.

**Visual states:**
- **Empty:** Shows upload icon, "Click to upload or drag and drop"
- **Dragging:** Blue border, highlighted background
- **File selected:** Green border, shows filename and size, shows X button to remove

---

### 📁 `frontend/components/CircularGauge.tsx` - Score Display

**What it does:** Displays the fit score as an animated circular progress indicator.

**Key features:**
- **Animated:** Score animates from 0 to actual value
- **Color-coded:** 
  - Green (≥80%) - Great match
  - Yellow (60-79%) - Good match
  - Red (<60%) - Needs improvement
- **Smooth animation:** Uses Framer Motion for smooth transitions

**How it works:**
1. Calculates circle circumference
2. Draws background circle (gray)
3. Draws progress circle (colored) that fills based on score
4. Animates the number from 0 to actual score
5. Displays score in center: "85 / 100"

**In simple terms:** This is like a speedometer in a car, but instead of speed, it shows how well you match the job.

---

### 📁 `frontend/app/layout.tsx` - Page Layout

**What it does:** Defines the overall structure and metadata for all pages.

**Key parts:**
- Sets page title, description, and metadata
- Includes global CSS
- Wraps all pages with consistent layout

**In simple terms:** This is like the frame of a picture - it defines how all pages look and what information they have.

---

### 📁 `frontend/app/globals.css` - Global Styles

**What it does:** Contains all the CSS styles used throughout the application.

**In simple terms:** This is like the style guide for your app - it defines colors, fonts, spacing, etc.

---

### 📁 `frontend/types/index.ts` - Type Definitions

**What it does:** Defines TypeScript types/interfaces for data structures.

**Why it matters:** TypeScript helps catch errors before code runs. If you try to use `result.fit_score` but `result` doesn't have that property, TypeScript will warn you.

**In simple terms:** This is like a contract that says "this data must have these properties" - it prevents mistakes.

---

## Configuration Files

### 📁 `docker-compose.yml` - Container Orchestrator

**What it does:** Defines all the services (containers) that need to run and how they connect.

**Services defined:**
1. **mongodb** - Database server (port 27017)
2. **backend** - FastAPI application (port 8000)
3. **ollama** - Local LLM server (port 11434)
4. **frontend** - Next.js application (port 3000)

**Key concepts:**
- **Networks:** All services are on the same network so they can talk to each other
- **Volumes:** Persistent storage for MongoDB and Ollama data
- **Environment variables:** Passes settings from `.env` file to containers
- **Depends on:** Ensures services start in the right order (backend waits for MongoDB)

**In simple terms:** This is like a blueprint that says "run these 4 programs, connect them together, and make sure they can talk to each other."

**Example:**
```yaml
backend:
  depends_on:
    - mongodb  # Wait for MongoDB to start first
    - ollama   # Wait for Ollama to start first
```

---

### 📁 `backend/Dockerfile` - Backend Container Definition

**What it does:** Instructions for building a Docker image for the backend.

**Steps:**
1. Start with Python 3.11 base image
2. Install system dependencies (gcc for compiling Python packages)
3. Copy requirements.txt and install Python packages
4. Copy application code
5. Expose port 8000
6. Run the FastAPI server

**In simple terms:** This is like a recipe for creating a container that runs your Python backend.

---

### 📁 `frontend/Dockerfile` - Frontend Container Definition

**What it does:** Instructions for building a Docker image for the frontend.

**Steps (multi-stage build):**
1. **deps stage:** Install npm packages
2. **builder stage:** Build the Next.js application
3. **runner stage:** Create production image with only necessary files

**Why multi-stage?** The final image is smaller because it doesn't include build tools, only the compiled application.

**In simple terms:** This is like building a house - you use big tools during construction, but the final house only has what's needed to live in it.

---

### 📁 `backend/requirements.txt` - Python Dependencies

**What it does:** Lists all Python packages needed for the backend.

**Key packages:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server (runs FastAPI)
- `langchain` - AI/LLM framework
- `pinecone-client` - Pinecone database client
- `motor` - Async MongoDB driver
- `pypdf` - PDF parser
- `sentence-transformers` - For HuggingFace embeddings

**In simple terms:** This is like a shopping list of libraries your Python code needs to work.

---

### 📁 `frontend/package.json` - Node.js Dependencies

**What it does:** Lists all npm packages needed for the frontend.

**Key packages:**
- `next` - Next.js framework
- `react` - React library
- `framer-motion` - Animation library
- `axios` - HTTP client (for API calls)
- `lucide-react` - Icon library

**In simple terms:** This is like a shopping list of libraries your JavaScript/TypeScript code needs to work.

---

## How Everything Works Together

### Complete Flow (Step by Step)

**1. User visits website (http://localhost:3000)**
- Frontend (`page.tsx`) loads
- User sees upload form

**2. User uploads resume**
- `FileUpload.tsx` component handles the file
- File is stored in React state

**3. User enters job description**
- Text is stored in React state

**4. User clicks "Evaluate Resume"**
- `handleEvaluate()` function runs
- Creates FormData with resume file and job description
- Sends POST request to `http://localhost:8000/evaluate`

**5. Backend receives request**
- `main.py` `/evaluate` endpoint receives the request
- Calls `parse_resume_file()` to extract text from PDF/DOCX
- Validates that both resume and job description exist

**6. RAG Pipeline runs**
- `rag_pipeline.py` `run_rag_pipeline()` is called
- **Step 6a:** Resume text is split into chunks and stored in Pinecone (via `vector_store.py`)
- **Step 6b:** Job description is used as a query to find relevant resume chunks
- **Step 6c:** Relevant chunks are retrieved from Pinecone
- **Step 6d:** Everything is sent to LLM (`llm_service.py`)
- **Step 6e:** LLM analyzes and returns JSON with fit score, gaps, strengths, evidence

**7. Backend returns response**
- JSON response is sent back to frontend
- Example: `{fit_score: 85, strengths: [...], gaps: [...], evidence: [...]}`

**8. Frontend displays results**
- `page.tsx` receives the response
- Updates state with results
- `CircularGauge.tsx` displays the score
- Strengths, gaps, and evidence are displayed in cards
- User can download report

---

### Data Flow Diagram

```
User Browser
    ↓
[Frontend: page.tsx]
    ↓ (HTTP POST)
[Backend: main.py /evaluate]
    ↓
[pdf_parser.py] → Extract text from PDF
    ↓
[rag_pipeline.py]
    ↓
[vector_store.py] → Store resume chunks in Pinecone
    ↓
[vector_store.py] → Retrieve relevant chunks
    ↓
[llm_service.py] → Send to AI (Ollama/OpenAI)
    ↓
[LLM] → Analyze and return JSON
    ↓
[main.py] → Return JSON response
    ↓ (HTTP Response)
[Frontend: page.tsx] → Display results
    ↓
User sees fit score, strengths, gaps, evidence
```

---

## Key Concepts for Beginners

### 1. **API (Application Programming Interface)**
- A way for different programs to talk to each other
- Like a waiter taking your order (request) and bringing food (response)
- In this project: Frontend sends requests to Backend API

### 2. **REST API**
- A standard way to structure API requests
- Uses HTTP methods: GET (read), POST (create), PUT (update), DELETE (remove)
- In this project: `POST /evaluate` means "create an evaluation"

### 3. **Vector Database (Pinecone)**
- Stores data as "vectors" (lists of numbers)
- Can find similar items quickly using math
- Like a library where books are organized by meaning, not title

### 4. **Embeddings**
- Converting text to numbers that represent meaning
- Similar texts get similar number patterns
- Example: "Python" and "programming" have similar embeddings

### 5. **RAG (Retrieval-Augmented Generation)**
- Instead of sending everything to AI, first find the most relevant parts
- Then send only those parts + the question
- Makes AI responses more accurate and faster

### 6. **LLM (Large Language Model)**
- AI that understands and generates text
- Examples: GPT-4 (OpenAI), Llama 2 (Ollama)
- In this project: Analyzes resume vs job description

### 7. **Docker & Docker Compose**
- **Docker:** Packages applications into containers (like shipping containers)
- **Docker Compose:** Runs multiple containers together
- Makes it easy to run the entire project with one command

### 8. **State Management (React)**
- React uses "state" to track data that changes
- When state changes, React automatically updates the UI
- Example: When `result` state changes, the results appear on screen

### 9. **Async/Await**
- JavaScript/Python way to handle operations that take time
- Like waiting for a pizza delivery - you don't block everything else
- In this project: Waiting for API responses, database queries, AI analysis

### 10. **TypeScript**
- JavaScript with type checking
- Catches errors before code runs
- Makes code more reliable and easier to understand

---

## File Summary Table

| File | Purpose | Key Function |
|------|---------|--------------|
| `backend/app/main.py` | API endpoints, request handling | Routes requests to services |
| `backend/app/config.py` | Settings and configuration | Stores API keys, URLs, etc. |
| `backend/app/models.py` | Data structures | Defines request/response shapes |
| `backend/app/database.py` | MongoDB connection | Connects to database |
| `backend/app/services/pdf_parser.py` | File parsing | Extracts text from PDF/DOCX |
| `backend/app/services/embeddings.py` | Text to numbers | Converts text to embeddings |
| `backend/app/services/vector_store.py` | Vector database | Stores/retrieves from Pinecone |
| `backend/app/services/llm_service.py` | AI analysis | Calls LLM for analysis |
| `backend/app/services/rag_pipeline.py` | RAG orchestration | Coordinates the entire process |
| `frontend/app/page.tsx` | Main UI page | User interface and interactions |
| `frontend/components/FileUpload.tsx` | File upload UI | Handles file selection |
| `frontend/components/CircularGauge.tsx` | Score display | Shows fit score visually |
| `docker-compose.yml` | Container orchestration | Runs all services together |

---

## Common Questions

**Q: Why split into frontend and backend?**
A: Separation of concerns. Frontend handles UI, backend handles logic. This makes code easier to maintain and allows multiple frontends (web, mobile) to use the same backend.

**Q: Why use Pinecone instead of regular database?**
A: Regular databases search by exact matches. Vector databases search by meaning. "Python developer" will find resumes mentioning "software engineer" or "coding" if they're semantically similar.

**Q: Why chunk text instead of sending full resume?**
A: LLMs have token limits. Also, chunking allows finding the most relevant parts. If a job requires "Python", we only send resume chunks that mention Python, not the entire resume.

**Q: What's the difference between Ollama and OpenAI?**
A: Ollama runs locally (free, private, slower). OpenAI runs in the cloud (paid, faster, requires internet). You can switch between them in config.

**Q: Why use Docker?**
A: Docker ensures the app runs the same way on any computer. No "it works on my machine" problems. One command (`docker-compose up`) runs everything.

---

## Next Steps for Learning

1. **Try modifying the UI:** Change colors, add new sections
2. **Add new API endpoints:** Create endpoints for saving results, viewing history
3. **Experiment with prompts:** Modify the LLM prompt in `llm_service.py` to get different analysis styles
4. **Add new file formats:** Extend `pdf_parser.py` to support more file types
5. **Improve chunking:** Adjust chunk size and overlap in `vector_store.py`
6. **Add authentication:** Require users to log in before using the app
7. **Add database features:** Store user profiles, analysis history, favorites

---

## Conclusion

This project demonstrates:
- **Full-stack development:** Frontend + Backend
- **AI/ML integration:** LLMs, embeddings, vector search
- **Modern web technologies:** Next.js, FastAPI, TypeScript
- **DevOps:** Docker, containerization
- **Database design:** MongoDB for structured data, Pinecone for vectors

Each file has a specific purpose, and they work together to create a complete AI-powered application. Start by understanding one file at a time, then see how they connect!

---

*Happy coding! 🚀*



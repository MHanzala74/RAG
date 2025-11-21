import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from advanced_rag_ebl import UnityRAGEBLSystem

from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG-Based Adaptive Quiz System",
    description="FastAPI backend for Unity WebGL integration",
    version="1.0.0"
)

# Initialize system
system = UnityRAGEBLSystem(api_key=os.getenv("OPENAI_API_KEY"))

# Enable CORS for Unity WebGL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Models
class ProcessBookRequest(BaseModel):
    pdf_path: str = Field(..., description="Path to the PDF file")
    collection_name: str = Field(..., min_length=1, description="Name for the vector collection")

class StudentPerformance(BaseModel):
    time_per_question: List[float] = Field(..., description="Time taken for each question")
    hints_used: int = Field(ge=0, description="Number of hints used")
    correct_answers: int = Field(ge=0, description="Number of correct answers")
    total_questions: int = Field(gt=0, description="Total questions attempted")
    current_difficulty: str = Field(..., description="Current difficulty level")

class GenerateQuizRequest(BaseModel):
    collection_name: str = Field(..., min_length=1, description="Collection name must not be empty")
    topic: str = Field(..., min_length=2, max_length=100, description="Topic for quiz generation")
    student_id: str = Field(..., pattern="^[a-zA-Z0-9_-]+$", description="Student identifier")  # Changed regex to pattern
    num_questions: int = Field(default=10, gt=0, le=20, description="Number of questions between 1-20")
    student_performance: Optional[Dict[str, Any]] = Field(default=None, description="Student performance data")
    
    @validator('topic')
    def topic_must_be_valid(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Topic must be at least 2 characters long')
        return v.strip()

# Response Models
class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str
    version: str = "1.0"

class ProcessBookResponse(BaseModel):
    book_title: str
    grade: str
    total_pages: int
    key_concepts: List[str]
    processing_time: float
    status: str = "success"

class QuizResponse(BaseModel):
    topic: str
    main_heading: str
    difficulty: str
    total_questions: int
    questions: List[Dict[str, Any]]
    generated_at: str

class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: Optional[str] = None
    timestamp: str


# Custom Exception Classes
class BookProcessingError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Book processing failed: {detail}"
        )

class QuizGenerationError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Quiz generation failed: {detail}"
        )

class CollectionNotFoundError(HTTPException):
    def __init__(self, collection_name: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_name}' not found"
        )

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logging.error(f"Global error handler: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            message="Something went wrong. Please try again later.",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_quiz_api")



@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint 1: Server health check"""
    logger.info("Health check requested")
    return HealthResponse(
        status="active",
        message="RAG Quiz System is running successfully",
        timestamp=datetime.now().isoformat()
    )

@app.post("/process-book", response_model=ProcessBookResponse)
async def process_book(request: ProcessBookRequest):
    """Endpoint 2: Process PDF and create vector database"""
    logger.info(f"Processing book: {request.pdf_path} for collection: {request.collection_name}")
    
    try:
        # Check if PDF file exists
        if not os.path.exists(request.pdf_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "FILE_NOT_FOUND", "message": "PDF file not found"}
            )
        
        # Process book and create vector DB
        book_structure = system.process_book_to_vector_db(
            pdf_path=request.pdf_path,
            collection_name=request.collection_name
        )
        
        logger.info(f"Book processed successfully: {book_structure.book_title}")
        
        return ProcessBookResponse(
            book_title=book_structure.book_title,
            grade=book_structure.grade,
            total_pages=book_structure.total_pages,
            key_concepts=book_structure.key_concepts,
            processing_time=book_structure.metadata.get("processing_time", 0)
        )
        
    except Exception as e:
        logger.error(f"Book processing error: {str(e)}")
        raise BookProcessingError(str(e))

@app.post("/generate-quiz")
async def generate_quiz(request: GenerateQuizRequest):
    """Endpoint 3: Main endpoint - Generate adaptive quiz"""
    logger.info(f"Generating quiz for student: {request.student_id}, topic: {request.topic}")
    
    try:
        # Check if collection exists
        collections = system.curriculum_agent.chroma_client.list_collections()
        collection_names = [col.name for col in collections]
        
        if request.collection_name not in collection_names:
            raise CollectionNotFoundError(request.collection_name)
        
        # Extract topic and generate quiz
        topic_structure, quiz = system.extract_topic_and_generate_quiz(
            collection_name=request.collection_name,
            topic=request.topic,
            student_id=request.student_id,
            student_performance=request.student_performance,
            num_questions=request.num_questions
        )
        
        logger.info(f"Quiz generated successfully: {len(quiz)} questions")
        
        # Convert MCQ objects to dictionaries properly
        quiz_dicts = []
        for q in quiz:
            if hasattr(q, '__dict__'):
                quiz_dicts.append(q.__dict__)
            else:
                # Fallback for different object types
                quiz_dicts.append({
                    'question': getattr(q, 'question', ''),
                    'options': getattr(q, 'options', []),
                    'correct': getattr(q, 'correct', ''),
                    'hint': getattr(q, 'hint', ''),
                    'explanation': getattr(q, 'explanation', ''),
                    'difficulty': getattr(q, 'difficulty', 'medium'),
                    'concepts_covered': getattr(q, 'concepts_covered', []),
                    'page_reference': getattr(q, 'page_reference', '')
                })
        
        return {
            "topic_structure": topic_structure.__dict__ if hasattr(topic_structure, '__dict__') else topic_structure,
            "quiz": quiz_dicts,
            "total_questions": len(quiz),
            "difficulty_adjusted": getattr(quiz[0], 'difficulty', 'medium') if quiz else 'medium'
        }
        
    except Exception as e:
        logger.error(f"Quiz generation error: {str(e)}")
        raise QuizGenerationError(str(e))

@app.get("/get-student-metrics/{student_id}")
async def get_student_metrics(student_id: str):
    """Endpoint 4: Get student performance data"""
    logger.info(f"Fetching metrics for student: {student_id}")
    
    behavioral_agent = system.behavioral_agent
    if student_id in behavioral_agent.student_data:
        student_data = behavioral_agent.student_data[student_id]
        return {
            "student_id": student_data.student_id,
            "performance_score": student_data.performance_score,
            "avg_time_per_question": student_data.avg_time_per_question,
            "hints_used_count": student_data.hints_used_count,
            "total_questions_attempted": student_data.total_questions_attempted,
            "correct_answers": student_data.correct_answers,
            "current_difficulty": student_data.current_difficulty,
            "last_updated": student_data.last_updated
        }
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "STUDENT_NOT_FOUND", "message": f"Student {student_id} not found"}
    )

@app.get("/list-collections")
async def list_collections():
    """Endpoint 5: Get available textbooks list"""
    logger.info("Listing all collections")
    
    try:
        collections = system.curriculum_agent.chroma_client.list_collections()
        return {
            "collections": [
                {
                    "name": col.name,
                    "metadata": col.metadata
                } for col in collections
            ],
            "total_collections": len(collections)
        }
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "COLLECTION_LIST_ERROR", "message": "Failed to list collections"}
        )



if __name__ == "__main__":
    import uvicorn
    
    # Environment variables for production
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    environment = os.getenv("ENVIRONMENT", "development")
    
    uvicorn.run(
        "main:app",  # Changed from "structured_fastapi:app" to "main:app"
        host=host,
        port=port,
        reload=environment == "development",  # Auto-reload only in development
        workers=1 if environment == "development" else 4,  # Multiple workers in production
        access_log=True
    )
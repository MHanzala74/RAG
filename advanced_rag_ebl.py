import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import PyPDF2
from openai import OpenAI
from dataclasses import dataclass, asdict
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


@dataclass
class BookStructure:
    """Structured representation of parsed book content"""
    grade: str
    book_title: str
    total_pages: int
    key_concepts: List[str]
    metadata: Dict[str, Any]


@dataclass
class TopicStructure:
    """Hierarchical structure for a specific topic"""
    topic: str
    main_heading: str
    subheadings: List[Dict[str, Any]]
    key_concepts: List[str]
    page_references: List[int]
    total_content_length: int


@dataclass
class BehavioralMetrics:
    """Student behavioral data"""
    student_id: str
    avg_time_per_question: float
    hints_used_count: int
    total_questions_attempted: int
    correct_answers: int
    current_difficulty: str
    performance_score: float
    last_updated: str


@dataclass
class MCQ:
    """Multiple Choice Question structure"""
    question: str
    options: List[str]
    correct: str
    hint: str
    explanation: str
    difficulty: str
    concepts_covered: List[str]
    page_reference: str


class CurriculumAgent:
    """
    Agent 1: Curriculum Understanding Agent with RAG
    - Parses ENTIRE PDF book
    - Creates vector embeddings for semantic search
    - Stores in ChromaDB for fast retrieval
    - Extracts topic-specific content based on teacher input
    """
    
    def __init__(self, api_key: str, chroma_persist_dir: str = "./chroma_db"):
        self.client = OpenAI(api_key=api_key)
        self.logs = []
        self.chroma_persist_dir = chroma_persist_dir
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=chroma_persist_dir,
            anonymized_telemetry=False
        ))
        
        # OpenAI embedding function
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )
    
    def _log(self, message: str, status: str = "info"):
        """Internal logging"""
        log_entry = {
            "agent": "Curriculum Agent",
            "message": message,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        self.logs.append(log_entry)
        print(f"[{status.upper()}] Curriculum Agent: {message}")
    
    def parse_entire_pdf(self, pdf_path: str) -> tuple[str, int]:
        """Extract ALL text from PDF - no chunk limits"""
        self._log(f"Starting FULL PDF parsing: {pdf_path}", "processing")
        
        try:
            text_content = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                self._log(f"PDF has {total_pages} pages - processing ALL", "info")
                
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    text_content += f"\n\n--- PAGE {page_num + 1} ---\n\n{page_text}"
                    
                    if (page_num + 1) % 50 == 0:
                        self._log(f"Progress: {page_num + 1}/{total_pages} pages processed", "info")
            
            self._log(f"COMPLETE: Extracted {len(text_content)} characters from {total_pages} pages", "success")
            return text_content, total_pages
        
        except Exception as e:
            self._log(f"PDF parsing error: {str(e)}", "error")
            raise
    
    def _chunk_text_with_overlap(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Create overlapping chunks with metadata"""
        self._log("Creating overlapping text chunks for embeddings...", "processing")
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # Extract page number from chunk
            page_match = chunk_text.rfind("--- PAGE ")
            page_num = 1
            if page_match != -1:
                try:
                    page_num = int(chunk_text[page_match + 9:page_match + 15].split()[0])
                except:
                    page_num = chunk_id // 3  # Approximate
            
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": chunk_text.strip(),
                "page": page_num,
                "start_pos": start,
                "end_pos": end
            })
            
            chunk_id += 1
            start += (chunk_size - overlap)
        
        self._log(f"Created {len(chunks)} overlapping chunks", "success")
        return chunks
    
    def create_vector_db(self, pdf_path: str, collection_name: str) -> BookStructure:
        """Process entire book and create vector database"""
        start_time = time.time()
        self._log("=== Starting Vector DB Creation ===", "info")
        
        # Step 1: Parse entire PDF
        book_content, total_pages = self.parse_entire_pdf(pdf_path)
        
        # Step 2: Detect grade and title
        grade = self._detect_grade(book_content[:5000])
        book_title = self._detect_book_title(book_content[:2000])
        
        # Step 3: Create chunks with overlap
        chunks = self._chunk_text_with_overlap(book_content)
        
        # Step 4: Create or get ChromaDB collection
        try:
            self.chroma_client.delete_collection(collection_name)
            self._log(f"Deleted existing collection: {collection_name}", "info")
        except:
            pass
        
        collection = self.chroma_client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"grade": grade, "book_title": book_title}
        )
        
        # Step 5: Add chunks to vector DB in batches
        self._log(f"Adding {len(chunks)} chunks to ChromaDB...", "processing")
        batch_size = 100
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            collection.add(
                ids=[chunk["id"] for chunk in batch],
                documents=[chunk["text"] for chunk in batch],
                metadatas=[{"page": chunk["page"], "start_pos": chunk["start_pos"]} for chunk in batch]
            )
            
            self._log(f"Progress: {min(i + batch_size, len(chunks))}/{len(chunks)} chunks added", "info")
            time.sleep(0.5)  # Rate limiting
        
        # Step 6: Extract global key concepts
        key_concepts = self._extract_global_concepts(book_content[:15000], grade)
        
        book_structure = BookStructure(
            grade=grade,
            book_title=book_title,
            total_pages=total_pages,
            key_concepts=key_concepts,
            metadata={
                "processing_time": round(time.time() - start_time, 2),
                "total_chunks": len(chunks),
                "collection_name": collection_name,
                "processed_at": datetime.now().isoformat()
            }
        )
        
        duration = round(time.time() - start_time, 2)
        self._log(f"=== Vector DB Creation Complete in {duration}s ===", "success")
        
        return book_structure
    
    def _detect_grade(self, content: str) -> str:
        """Detect grade level"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Analyze text and return ONLY the grade level (e.g., '5th Grade', '8th Grade')."},
                    {"role": "user", "content": f"Grade level of:\n\n{content}"}
                ],
                max_tokens=20,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self._log(f"Grade detection error: {str(e)}", "warning")
            return "Unknown Grade"
    
    def _detect_book_title(self, content: str) -> str:
        """Extract book title"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Extract the book title from the text. Return ONLY the title."},
                    {"role": "user", "content": content}
                ],
                max_tokens=50,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except:
            return "Untitled Textbook"
    
    def _extract_global_concepts(self, sample: str, grade: str) -> List[str]:
        """Extract key concepts from book sample"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"Extract 20 key concepts from this {grade} textbook. Return as JSON array."},
                    {"role": "user", "content": sample}
                ],
                max_tokens=500,
                temperature=0.4
            )
            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            self._log(f"Concept extraction error: {str(e)}", "warning")
            return []
    
    def extract_topic_content(
        self,
        collection_name: str,
        topic: str,
        num_results: int = 20
    ) -> TopicStructure:
        """RAG: Retrieve topic-specific content and generate hierarchical structure"""
        self._log(f"Extracting content for topic: '{topic}'", "processing")
        start_time = time.time()
        
        try:
            # Get collection
            collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            
            # Perform semantic search
            self._log(f"Performing semantic search (top {num_results} results)...", "processing")
            results = collection.query(
                query_texts=[topic],
                n_results=num_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Extract retrieved content
            retrieved_chunks = results['documents'][0]
            page_refs = [meta['page'] for meta in results['metadatas'][0]]
            
            combined_content = "\n\n".join(retrieved_chunks)
            self._log(f"Retrieved {len(retrieved_chunks)} relevant chunks", "success")
            
            # Generate hierarchical structure using LLM
            structure = self._generate_topic_hierarchy(topic, combined_content)
            
            duration = round(time.time() - start_time, 2)
            self._log(f"Topic extraction complete in {duration}s", "success")
            
            return TopicStructure(
                topic=topic,
                main_heading=structure["main_heading"],
                subheadings=structure["subheadings"],
                key_concepts=structure["key_concepts"],
                page_references=sorted(list(set(page_refs))),
                total_content_length=len(combined_content)
            )
        
        except Exception as e:
            self._log(f"Topic extraction error: {str(e)}", "error")
            raise
    
    def _generate_topic_hierarchy(self, topic: str, content: str) -> Dict[str, Any]:
        """Generate hierarchical structure from retrieved content"""
        self._log("Generating topic hierarchy using LLM...", "processing")
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an educational content analyzer. Generate a hierarchical structure from the provided content.

Return ONLY valid JSON in this format:
{
  "main_heading": "Main Topic Title",
  "subheadings": [
    {
      "title": "Subheading 1",
      "description": "Detailed description of this subtopic",
      "key_points": ["Point 1", "Point 2", "Point 3"]
    }
  ],
  "key_concepts": ["Concept 1", "Concept 2", "..."]
}"""
                    },
                    {
                        "role": "user",
                        "content": f"Topic: {topic}\n\nContent:\n{content[:8000]}"
                    }
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        
        except Exception as e:
            self._log(f"Hierarchy generation error: {str(e)}", "error")
            return {
                "main_heading": topic,
                "subheadings": [],
                "key_concepts": []
            }


class BehavioralAgent:
    """Agent 2: Behavioral Agent - tracks student performance"""
    
    def __init__(self):
        self.logs = []
        self.student_data: Dict[str, BehavioralMetrics] = {}
    
    def _log(self, message: str, status: str = "info"):
        log_entry = {
            "agent": "Behavioral Agent",
            "message": message,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        self.logs.append(log_entry)
        print(f"[{status.upper()}] Behavioral Agent: {message}")
    
    def track_student_performance(
        self,
        student_id: str,
        time_per_question: List[float],
        hints_used: int,
        correct_answers: int,
        total_questions: int,
        current_difficulty: str
    ) -> BehavioralMetrics:
        """Track and analyze student performance"""
        self._log(f"Analyzing performance for student: {student_id}", "processing")
        
        avg_time = sum(time_per_question) / len(time_per_question) if time_per_question else 0
        performance_score = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        
        metrics = BehavioralMetrics(
            student_id=student_id,
            avg_time_per_question=round(avg_time, 2),
            hints_used_count=hints_used,
            total_questions_attempted=total_questions,
            correct_answers=correct_answers,
            current_difficulty=current_difficulty,
            performance_score=round(performance_score, 2),
            last_updated=datetime.now().isoformat()
        )
        
        self.student_data[student_id] = metrics
        self._log(f"Performance tracked: {performance_score}% accuracy", "success")
        
        return metrics
    
    def adjust_difficulty(self, student_metrics: BehavioralMetrics, grade: str) -> str:
        """Dynamically adjust difficulty"""
        performance = student_metrics.performance_score
        avg_time = student_metrics.avg_time_per_question
        hints_used = student_metrics.hints_used_count
        
        new_difficulty = student_metrics.current_difficulty
        
        if performance >= 80 and avg_time < 30 and hints_used < 2:
            new_difficulty = "hard"
        elif performance < 50 and (avg_time > 60 or hints_used >= 5):
            new_difficulty = "easy"
        elif 50 <= performance < 80:
            new_difficulty = "medium"
        
        if "5th" in grade.lower() or "5" in grade:
            if new_difficulty == "hard":
                new_difficulty = "medium"
        
        self._log(f"Difficulty adjusted to: {new_difficulty.upper()}", "success")
        return new_difficulty


class QuizGeneratorAgent:
    """Agent 3: RAG-Enhanced Quiz Generator"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.logs = []
    
    def _log(self, message: str, status: str = "info"):
        log_entry = {
            "agent": "Quiz Generator Agent",
            "message": message,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        self.logs.append(log_entry)
        print(f"[{status.upper()}] Quiz Generator: {message}")
    
    def generate_quiz_from_topic(
        self,
        topic_structure: TopicStructure,
        difficulty: str,
        num_questions: int = 10
    ) -> List[MCQ]:
        """Generate quiz from RAG-retrieved topic content"""
        self._log(f"Generating {num_questions} questions at {difficulty.upper()} difficulty", "processing")
        start_time = time.time()
        
        try:
            # Prepare context from topic structure
            context = self._prepare_context(topic_structure)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an expert quiz generator. Create {num_questions} high-quality MCQs at {difficulty} difficulty.

STRICT REQUIREMENTS:
- Base questions ONLY on the provided context
- Each question has 4 options (A, B, C, D)
- Include hints and explanations
- Adjust complexity: easy=recall, medium=application, hard=analysis

Return ONLY valid JSON array:
[
  {{
    "question": "Question text?",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "correct": "A",
    "hint": "Helpful hint",
    "explanation": "Why this is correct",
    "difficulty": "{difficulty}",
    "concepts_covered": ["concept1"],
    "page_reference": "Page X"
  }}
]"""
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nGenerate {num_questions} MCQs."
                    }
                ],
                max_tokens=3000,
                temperature=0.7
            )
            
            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            mcqs_data = json.loads(text)
            
            mcqs = [MCQ(**mcq) for mcq in mcqs_data]
            
            duration = round(time.time() - start_time, 2)
            self._log(f"Generated {len(mcqs)} questions in {duration}s", "success")
            
            return mcqs
        
        except Exception as e:
            self._log(f"Quiz generation error: {str(e)}", "error")
            raise
    
    def _prepare_context(self, topic_structure: TopicStructure) -> str:
        """Prepare context from topic structure"""
        context = f"Topic: {topic_structure.topic}\n\n"
        context += f"Main Heading: {topic_structure.main_heading}\n\n"
        
        for sub in topic_structure.subheadings:
            context += f"## {sub['title']}\n"
            context += f"{sub['description']}\n"
            if 'key_points' in sub:
                context += "Key Points:\n"
                for point in sub['key_points']:
                    context += f"- {point}\n"
            context += "\n"
        
        context += f"\nKey Concepts: {', '.join(topic_structure.key_concepts)}\n"
        context += f"Pages: {', '.join(map(str, topic_structure.page_references[:5]))}"
        
        return context


class ValidatorAgent:
    """Agent 4: Validator - ensures questions are grounded in source content"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.logs = []
    
    def _log(self, message: str, status: str = "info"):
        log_entry = {
            "agent": "Validator Agent",
            "message": message,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        self.logs.append(log_entry)
        print(f"[{status.upper()}] Validator: {message}")
    
    def validate_quiz(
        self,
        mcqs: List[MCQ],
        topic_structure: TopicStructure
    ) -> List[MCQ]:
        """Validate MCQs against source content"""
        self._log(f"Validating {len(mcqs)} questions", "processing")
        
        try:
            context = self._prepare_validation_context(topic_structure)
            questions_text = "\n".join([f"Q{i+1}: {mcq.question}" for i, mcq in enumerate(mcqs)])
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """Validate if questions are based on provided context. Return JSON:
[{"index": 0, "valid": true, "issue": "", "confidence": 0.95}]"""
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestions:\n{questions_text}"
                    }
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            validation_results = json.loads(text)
            
            validated_mcqs = []
            for idx, mcq in enumerate(mcqs):
                validation = next((v for v in validation_results if v["index"] == idx), None)
                if validation and validation.get("valid", True):
                    validated_mcqs.append(mcq)
                else:
                    self._log(f"Q{idx+1} rejected: {validation.get('issue', 'Invalid')}", "warning")
            
            self._log(f"Validation complete: {len(validated_mcqs)}/{len(mcqs)} approved", "success")
            return validated_mcqs
        
        except Exception as e:
            self._log(f"Validation error: {str(e)}", "warning")
            return mcqs
    
    def _prepare_validation_context(self, topic_structure: TopicStructure) -> str:
        """Prepare context for validation"""
        context = f"Topic: {topic_structure.topic}\n"
        context += f"Key Concepts: {', '.join(topic_structure.key_concepts[:10])}\n"
        return context


# ============= UNITY INTEGRATION CONTROLLER =============

class UnityRAGEBLSystem:
    """Main controller for Unity with RAG pipeline"""
    
    def __init__(self, api_key: str, chroma_dir: str = "./chroma_db"):
        self.curriculum_agent = CurriculumAgent(api_key, chroma_dir)
        self.behavioral_agent = BehavioralAgent()
        self.quiz_agent = QuizGeneratorAgent(api_key)
        self.validator_agent = ValidatorAgent(api_key)
        self.all_logs = []
    
    def process_book_to_vector_db(
        self,
        pdf_path: str,
        collection_name: str
    ) -> BookStructure:
        """Step 1: Process entire book and create vector database"""
        print("\n" + "="*60)
        print("STEP 1: CREATING VECTOR DATABASE FROM ENTIRE BOOK")
        print("="*60)
        
        book_structure = self.curriculum_agent.create_vector_db(pdf_path, collection_name)
        self.all_logs.extend(self.curriculum_agent.logs)
        
        return book_structure
    
    def extract_topic_and_generate_quiz(
        self,
        collection_name: str,
        topic: str,
        student_id: str,
        student_performance: Optional[Dict] = None,
        num_questions: int = 10
    ) -> tuple[TopicStructure, List[MCQ]]:
        """Step 2-5: Extract topic content and generate adaptive quiz"""
        
        # Extract topic content using RAG
        print("\n" + "="*60)
        print(f"STEP 2: EXTRACTING CONTENT FOR TOPIC: '{topic}'")
        print("="*60)
        
        topic_structure = self.curriculum_agent.extract_topic_content(
            collection_name,
            topic,
            num_results=20
        )
        self.all_logs.extend(self.curriculum_agent.logs)
        
        # Behavioral analysis
        print("\n" + "="*60)
        print("STEP 3: BEHAVIORAL ANALYSIS")
        print("="*60)
        
        if student_performance:
            metrics = self.behavioral_agent.track_student_performance(
                student_id=student_id,
                **student_performance
            )
            difficulty = self.behavioral_agent.adjust_difficulty(
                metrics,
                self.curriculum_agent.chroma_client.get_collection(collection_name).metadata.get('grade', 'Unknown')
            )
        else:
            difficulty = "medium"
        
        self.all_logs.extend(self.behavioral_agent.logs)
        
        # Generate quiz
        print("\n" + "="*60)
        print("STEP 4: GENERATING QUIZ FROM EXTRACTED CONTENT")
        print("="*60)
        
        mcqs = self.quiz_agent.generate_quiz_from_topic(
            topic_structure,
            difficulty,
            num_questions
        )
        self.all_logs.extend(self.quiz_agent.logs)
        
        # Validate
        print("\n" + "="*60)
        print("STEP 5: VALIDATING QUESTIONS")
        print("="*60)
        
        validated_mcqs = self.validator_agent.validate_quiz(mcqs, topic_structure)
        self.all_logs.extend(self.validator_agent.logs)
        
        return topic_structure, validated_mcqs
    
    def save_outputs(self, topic_structure: TopicStructure, mcqs: List[MCQ], prefix: str = "output"):
        """Save all outputs to JSON files"""
        # Save topic structure
        with open(f"{prefix}_topic_structure.json", 'w', encoding='utf-8') as f:
            json.dump(asdict(topic_structure), f, indent=2, ensure_ascii=False)
        
        # Save quiz
        quiz_data = {
            "generated_at": datetime.now().isoformat(),
            "topic": topic_structure.topic,
            "total_questions": len(mcqs),
            "questions": [asdict(mcq) for mcq in mcqs]
        }
        with open(f"{prefix}_quiz.json", 'w', encoding='utf-8') as f:
            json.dump(quiz_data, f, indent=2, ensure_ascii=False)
        
        # Save logs
        with open(f"{prefix}_logs.json", 'w') as f:
            json.dump(self.all_logs, f, indent=2)
        
        print(f"\n✅ Outputs saved: {prefix}_*.json")


# ============= EXAMPLE USAGE =============

def main():
    """Example workflow"""
    
    API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    system = UnityRAGEBLSystem(API_KEY)
    
    # ONE-TIME: Process book and create vector DB
    print("\n" + "="*60)
    print("PROCESSING TEXTBOOK TO VECTOR DATABASE")
    print("="*60)
    
    book_structure = system.process_book_to_vector_db(
        pdf_path="textbook.pdf",
        collection_name="biology_grade8"
    )
    
    print(f"\n📚 Book Processed:")
    print(f"   Title: {book_structure.book_title}")
    print(f"   Grade: {book_structure.grade}")
    print(f"   Pages: {book_structure.total_pages}")
    print(f"   Chunks: {book_structure.metadata['total_chunks']}")
    
    # TEACHER WORKFLOW: Extract topic and generate quiz
    topic_structure, quiz = system.extract_topic_and_generate_quiz(
        collection_name="biology_grade8",
        topic="Photosynthesis in Plants",
        student_id="student_001",
        num_questions=10
    )
    
    print(f"\n📋 Topic Structure:")
    print(f"   Main Heading: {topic_structure.main_heading}")
    print(f"   Subheadings: {len(topic_structure.subheadings)}")
    print(f"   Key Concepts: {len(topic_structure.key_concepts)}")
    print(f"   Pages: {topic_structure.page_references[:5]}")
    
    print(f"\n✅ Generated {len(quiz)} validated questions")
    
    # Save outputs
    system.save_outputs(topic_structure, quiz, "unity_output")
    
    print("\n" + "="*60)
    print("✅ COMPLETE - Ready for Unity Integration")
    print("="*60)


if __name__ == "__main__":
    main()

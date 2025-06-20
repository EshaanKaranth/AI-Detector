import os
import re
import logging
import json
import subprocess
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Set, Dict, Optional
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain.text_splitter import MarkdownTextSplitter
import pytesseract
from pdf2image import convert_from_path
from unstructured.partition.auto import partition

load_dotenv()

tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 3),
                max_features=1000,
                lowercase=True
            )

qdrant = None
model = None
splitter = None
logger = None
text_cache: Dict[str, str] = {}  # Cache for extracted text
processed_files_cache: Dict[str, float] = {}  # Cache for processed file modification times
COLLECTION_NAME = "resume_database"
SUPPORTED_EXTENSIONS = {'.doc', '.docx', '.pdf', '.txt'}

def get_logger(log_file: str) -> logging.Logger:
    """Initialize or retrieve singleton logger."""
    global logger
    if logger is None:
        logger = logging.getLogger('resume_analyzer')
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def validate_folder(path: str, folder_type: str = "input") -> Path:
    """Validate folder path for input or output."""
    path = Path(path.strip())
    try:
        if folder_type == "input":
            if not path.exists() or not path.is_dir():
                raise ValueError(f"Input folder does not exist or is not a directory: {path}")
        elif folder_type == "output":
            path.mkdir(parents=True, exist_ok=True)
            test_file = path / ".test_write"
            test_file.touch()
            test_file.unlink()
        else:
            raise ValueError(f"Invalid folder_type: {folder_type}. Use 'input' or 'output'.")
        return path
    except (PermissionError, OSError) as e:
        raise ValueError(f"Folder validation failed for {path}: {e}")

def validate_environment():
    """Validate required environment variables."""
    load_dotenv()
    required_vars = ["QDRANT_URL", "QDRANT_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        if logger:
            logger.error(f"Missing environment variables: {missing}")
        raise ValueError(f"Missing environment variables: {missing}")

def init(qdrant_url: str, qdrant_api_key: str, collection_name: str):
    """Initialize Qdrant client, model, splitter, and caches."""
    global qdrant, model, splitter, tfidf_vectorizer, logger
    
    try:
        # Initialize logger first if not already done
        if logger is None:
            get_logger("resume_analyzer.log")
            
        logger.info(f"Initializing Qdrant with URL: {qdrant_url}, Collection: {collection_name}")
        
        # Initialize Qdrant client
        if qdrant is None:
            qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            # Test connection by fetching collections
            collections = qdrant.get_collections()
            logger.info(f"Connected to Qdrant. Available collections: {[c.name for c in collections.collections]}")
        
        # Initialize SentenceTransformer model
        if model is None:
            device = os.getenv("TRANSFORMER_DEVICE", "cpu")
            logger.info(f"Loading SentenceTransformer model on device: {device}")
            model = SentenceTransformer('intfloat/multilingual-e5-large-instruct', device=device)
            logger.info("SentenceTransformer model loaded successfully")
        
        # Initialize text splitter with error handling
        if splitter is None:
            try:
                splitter = MarkdownTextSplitter(chunk_size=300, chunk_overlap=75)
                logger.info("MarkdownTextSplitter initialized successfully")
            except Exception as splitter_error:
                logger.warning(f"Failed to initialize MarkdownTextSplitter: {splitter_error}")
                logger.info("Falling back to RecursiveCharacterTextSplitter")
                try:
                    from langchain_text_splitters import RecursiveCharacterTextSplitter
                    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=75)
                    logger.info("RecursiveCharacterTextSplitter initialized successfully")
                except ImportError:
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=75)
                    logger.info("RecursiveCharacterTextSplitter initialized successfully (legacy import)")
        
        # Ensure collection exists
        ensure_collection_exists(collection_name)
        logger.info("Initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        # Reset global variables on failure
        qdrant = None
        model = None
        splitter = None
        tfidf_vectorizer = None
        raise ValueError(f"Failed to initialize components: {e}")

def is_initialized() -> bool:
    """Check if all required components are initialized."""
    return all([qdrant is not None, model is not None, splitter is not None, tfidf_vectorizer is not None])

def ensure_collection_exists(collection_name: str):
    """Ensure Qdrant collection exists."""
    if qdrant is None:
        raise ValueError("Qdrant client not initialized")
    
    try:
        existing_collections = [c.name for c in qdrant.get_collections().collections]
        if collection_name not in existing_collections:
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE)
            )
            logger.info(f"Created Qdrant collection: {collection_name}")
        else:
            logger.info(f"Collection {collection_name} already exists")
    except Exception as e:
        logger.error(f"Failed to ensure collection {collection_name}: {e}")
        raise

def load_json_set(path: Path) -> Set[str]:
    """Load a set from a JSON file."""
    path = Path(path)
    try:
        if path.exists():
            with path.open("r", encoding='utf-8') as f:
                return set(json.load(f))
        return set()
    except Exception as e:
        if logger:
            logger.error(f"Failed to load {path}: {e}")
        return set()

def save_json_set(data: Set[str], path: Path):
    """Save a set to a JSON file."""
    path = Path(path)
    try:
        with path.open("w", encoding='utf-8') as f:
            json.dump(sorted(list(data)), f, indent=2)
        if logger:
            logger.info(f"Saved {len(data)} items to {path}")
    except Exception as e:
        if logger:
            logger.error(f"Failed to save {path}: {e}")

def safe_upsert_with_retry(client: QdrantClient, collection_name: str, points: list, max_retries: int = 3) -> bool:
    """Upsert points to Qdrant with retries."""
    for attempt in range(max_retries):
        try:
            client.upsert(collection_name=collection_name, points=points)
            return True
        except Exception as e:
            wait = 2 ** attempt
            if logger:
                logger.warning(f"Qdrant upsert failed (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
            import time
            time.sleep(wait)
    if logger:
        logger.error("Qdrant upsert failed after all retries.")
    return False

def convert_doc_to_pdf(file_path: Path) -> Optional[Path]:
    """Convert .doc/.docx to PDF using LibreOffice."""
    file_path = Path(file_path)
    output_dir = file_path.parent
    pdf_path = output_dir / f"{file_path.stem}.pdf"
    try:
        subprocess.run(
            [
                "soffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(output_dir),
                str(file_path)
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if pdf_path.exists():
            if logger:
                logger.info(f"Converted {file_path} to PDF: {pdf_path}")
            return pdf_path
        if logger:
            logger.error(f"Conversion failed: {pdf_path} not created")
        return None
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"LibreOffice conversion failed for {file_path}: {e}")
        return None

def is_text_pdf(file_path: Path) -> bool:
    """Check if a PDF contains extractable text."""
    file_path = Path(file_path)
    try:
        elements = partition(filename=str(file_path))
        return any(element.text.strip() for element in elements)
    except Exception as e:
        if logger:
            logger.error(f"Error checking text in PDF {file_path}: {e}")
        return False

def ocr_pdf(file_path: Path) -> str:
    """Perform OCR on a PDF file."""
    file_path = Path(file_path)
    try:
        images = convert_from_path(str(file_path))
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image) + " "
        text = re.sub(r'\s+', ' ', text.strip())
        if logger:
            logger.info(f"OCR completed for {file_path}")
        return text
    except Exception as e:
        if logger:
            logger.error(f"OCR failed for {file_path}: {e}")
        return ""

def unstructured_to_markdown(file_path: Path) -> str:
    """Extract text from a file using unstructured and convert to markdown."""
    file_path = Path(file_path)
    try:
        elements = partition(filename=str(file_path))
        text = "\n".join(element.text for element in elements if element.text.strip())
        text = re.sub(r'\s+', ' ', text.strip())
        if logger:
            logger.info(f"Extracted markdown from {file_path}")
        return text
    except Exception as e:
        if logger:
            logger.error(f"Text extraction failed for {file_path}: {e}")
        return ""

def safe_create_documents(text: str, metadata: dict = None) -> list:
    """Safely create documents using the text splitter with proper error handling."""
    if not is_initialized():
        raise ValueError("Components not properly initialized. Call init() first.")
    
    if splitter is None:
        raise
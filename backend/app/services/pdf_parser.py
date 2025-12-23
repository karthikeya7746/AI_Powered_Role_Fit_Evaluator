import pypdf
from docx import Document
from typing import Optional
import io


async def parse_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error parsing PDF: {str(e)}")


async def parse_docx(file_content: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        docx_file = io.BytesIO(file_content)
        doc = Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error parsing DOCX: {str(e)}")


async def parse_resume_file(file_content: bytes, filename: str) -> str:
    """Extract text from PDF, TXT, or DOCX file"""
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.pdf'):
        return await parse_pdf(file_content)
    elif filename_lower.endswith('.txt'):
        try:
            return file_content.decode('utf-8').strip()
        except UnicodeDecodeError:
            # Try with different encoding
            return file_content.decode('latin-1').strip()
    elif filename_lower.endswith('.docx'):
        return await parse_docx(file_content)
    elif filename_lower.endswith('.doc'):
        # .doc files (older format) - try to parse as .docx first, if that fails, raise error
        # Note: python-docx doesn't support .doc format directly
        # For .doc files, we'd need additional libraries like python-docx2txt or textract
        # For now, we'll attempt to parse and provide a helpful error message
        try:
            # Try parsing as .docx (sometimes works if file is actually .docx with wrong extension)
            return await parse_docx(file_content)
        except:
            raise ValueError(
                f"Unsupported file type: {filename}. "
                "Please convert .doc files to .docx format. "
                "Supported formats: PDF, TXT, DOCX"
            )
    else:
        raise ValueError(
            f"Unsupported file type: {filename}. "
            "Supported formats: PDF, TXT, DOCX"
        )



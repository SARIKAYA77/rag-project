import requests
import PyPDF2
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')


def download_pdf(url: str) -> str:
    response = requests.get(url)
    pdf_file_path = "document.pdf"
    with open(pdf_file_path, "wb") as file:
        file.write(response.content)
    return pdf_file_path


def extract_text_from_pdf(pdf_file_path: str) -> str:
    with open(pdf_file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


def split_text_into_chunks(text: str, chunk_size: int = 100) -> list:
    words = word_tokenize(text)
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

import re
import requests
import PyPDF2
import nltk

nltk.download('punkt')


def clean_text(text: str) -> str:
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\sğüşıöçĞÜŞİÖÇ]', '', text)
    return text


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
            page_text = page.extract_text()
            text += clean_text(page_text)
    return text


def split_text_into_chunks(text: str, chunk_size: int = 100) -> list:
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


def find_relevant_chunks(chunks: list, query: str) -> list:
    relevant_chunks = []
    query_words = set(query.lower().split())

    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        if query_words & chunk_words:
            relevant_chunks.append(chunk)

    return relevant_chunks

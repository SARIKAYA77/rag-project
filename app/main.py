from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.pdf_processing import download_pdf, extract_text_from_pdf, split_text_into_chunks
from app.embedding import embed_text
from app.database import index_embeddings, search_embeddings
import numpy as np

app = FastAPI()


class AskRequest(BaseModel):
    url: str
    query: str


@app.get("/")
def read_root():
    return {"message": "Welcome to your first FastAPI project!"}


@app.post("/ask")
def ask_question(request: AskRequest):
    try:
        # PDF'i indir
        pdf_file = download_pdf(request.url)
        # PDF'den metin çıkar
        text = extract_text_from_pdf(pdf_file)
        # Metni parçalara ayır
        chunks = split_text_into_chunks(text)
        # Parçalardan embedding oluştur
        embeddings = embed_text(chunks)
        # Embedding'leri Redis'e indeksle
        index_embeddings(chunks, embeddings)

        # Sorgu için embedding oluştur
        query_embedding = embed_text([request.query])[0]

        # Benzerliği ölçmek için embedding'leri sorgu ile karşılaştırın
        results = search_embeddings(chunks, embeddings, query_embedding)

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


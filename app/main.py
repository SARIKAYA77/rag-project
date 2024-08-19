from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.pdf_processing import download_pdf, extract_text_from_pdf, split_text_into_chunks, find_relevant_chunks, clean_text
from app.embedding import embed_text
from app.database import index_embeddings, search_embeddings

app = FastAPI()

# Global variable to store the last results
last_results = None


class AskRequest(BaseModel):
    url: str
    query: str


@app.post("/ask")
def ask_question(request: AskRequest):
    global last_results  # Global değişkeni kullanacağınızı belirtiyoruz
    try:
        pdf_file = download_pdf(request.url)
        text = extract_text_from_pdf(pdf_file)
        chunks = split_text_into_chunks(text)

        if not chunks:
            raise HTTPException(status_code=404, detail="No relevant chunks found for the query")

        relevant_chunks = find_relevant_chunks(chunks, request.query)
        if not relevant_chunks:
            raise HTTPException(status_code=404, detail="No relevant chunks found for the query")

        embeddings = embed_text(relevant_chunks)
        index_embeddings(relevant_chunks, embeddings)

        query_embedding = embed_text([request.query])[0]
        results = search_embeddings(relevant_chunks, embeddings, query_embedding)

        cleaned_results = [{"chunk": clean_text(result['chunk']), "distance": round(result['distance'], 3)} for result in results]

        last_results = cleaned_results

        return {"results": cleaned_results}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/results")
def get_last_results():
    global last_results
    if last_results is None:
        raise HTTPException(status_code=404, detail="No results available")
    return {"results": last_results}

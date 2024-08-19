from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.pdf_processing import download_pdf, extract_text_from_pdf, split_text_into_chunks, find_relevant_chunks, \
    clean_text
from app.embedding import embed_text
from app.database import index_embeddings, search_embeddings

app = FastAPI()

last_results = None


class AskRequest(BaseModel):
    url: str
    query: str


@app.post("/ask")
def ask_question(request: AskRequest):
    global last_results
    try:
        pdf_file = download_pdf(request.url)
        text = extract_text_from_pdf(pdf_file)
        chunks = split_text_into_chunks(text)

        if not chunks:
            raise HTTPException(status_code=404, detail="No relevant chunks found in the document")

        relevant_chunks = find_relevant_chunks(chunks, request.query)

        if not relevant_chunks:
            raise HTTPException(status_code=404, detail="No relevant chunks found for the query")
        if relevant_chunks:
            embeddings = embed_text(relevant_chunks)
            index_embeddings(relevant_chunks, embeddings)

            query_embedding = embed_text([request.query])[0]
            results = search_embeddings(relevant_chunks, embeddings, query_embedding)

            if not results:
                raise HTTPException(status_code=404, detail="No relevant results found for the query")

            cleaned_results = [{"chunk": clean_text(result['chunk']), "distance": result['distance']} for result in
                               results]
        else:
            cleaned_results = []

        last_results = cleaned_results

        return {"results": cleaned_results}
    except HTTPException as e:
        raise e
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(error)}")


@app.get("/results")
def get_last_results():
    global last_results
    if last_results is None:
        raise HTTPException(status_code=404, detail="No results available")
    return {"results": last_results}

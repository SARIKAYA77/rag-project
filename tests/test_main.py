import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from app.pdf_processing import split_text_into_chunks, find_relevant_chunks

client = TestClient(app)

mocked_embedding = [0.1, 0.2, 0.3, 0.4]


@pytest.fixture
def mock_pdf_text():
    return """
    Yapay Zeka Nedir? Yapay zeka (AI), makinelerin insanlar gibi düşünme, öğrenme, problem çözme, algılama ve dil anlama
     gibi görevleri yerine getirme yeteneğine sahip olduğu bir teknoloji alanıdır.
    Yapay Zeka Türleri: Dar Yapay Zeka (ANI) ve Genel Yapay Zeka (AGI) gibi türleri bulunmaktadır.
    """


@pytest.fixture
def mock_chunks(mock_pdf_text):
    return split_text_into_chunks(mock_pdf_text)


@pytest.fixture
def mock_relevant_chunks(mock_chunks):
    return find_relevant_chunks(mock_chunks, "Yapay zeka nedir")


@pytest.fixture
def mock_embeddings():
    return [mocked_embedding for _ in range(5)]


@patch("app.pdf_processing.download_pdf")
@patch("app.embedding.embed_text")
@patch("app.database.index_embeddings")
@patch("app.database.search_embeddings")
def test_ask_question(mock_search_embeddings, mock_embed_text, mock_download_pdf,
                      mock_relevant_chunks, mock_embeddings):
    mock_download_pdf.return_value = "document.pdf"
    mock_embed_text.side_effect = lambda chunks: mock_embeddings
    mock_search_embeddings.return_value = [{"chunk": chunk, "distance": 0.1} for chunk in mock_relevant_chunks]
    response = client.post("/ask", json={
        "url": "https://drive.google.com/uc?export=download&id=1p6PqK07fzTwupJmDXYoczn1hWpGnEUEj",
        "query": "Yapay zeka nedir?"})

    if response.status_code != 200:
        print("test_ask_question error:", response.json())

    # Sonuçları doğrula
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 5  # En fazla 5 sonuç dönmeli
    assert all("chunk" in result and "distance" in result for result in data["results"])


@patch("app.pdf_processing.download_pdf")
def test_get_last_results(mock_download_pdf):
    mock_download_pdf.return_value = "document.pdf"
    client.post("/ask", json={"url": "https://drive.google.com/uc?export=download&id=1p6PqK07fzTwupJmDXYoczn1hWpGnEUEj",
                              "query": "Yapay zeka nedir?"})

    response = client.get("/results")

    if response.status_code != 200:
        print("test_get_last_results error:", response.json())

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) > 0

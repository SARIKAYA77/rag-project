# PDF Question Answering API Project

This project provides a RESTFULL API that extracts text from a PDF document, allows users to ask questions about the content, and returns the most relevant answers. The API is built with FastAPI and uses Redis for embedding database management.

## Features

- Extract text from a PDF file
- Split the text into meaningful chunks
- Find the most relevant chunks related to a query
- Return the top 5 most relevant answers to a query

## Installation

### Requirements

To run this project, you need the following dependencies:

- Python 3.8+
- FastAPI
- PyTorch
- Transformers (Hugging Face)
- Redis
- PyPDF2
- NLTK
- Requests
- Numpy

### Docker Installation

To set up the project using Docker, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone git@github.com:SARIKAYA77/rag-project.git
   cd repo-name
   
2. **Build and Run the Docker Container**:
```bash
    docker build -t pdf-question-api .
    docker run -d -p 8000:8000 pdf-question-api
```
**Install the Dependencies**:
```
    pip install -r requirements.txt
```

**Start the Application:**:
```
    uvicorn app.main:app --reload
```

**Usage:**
    POST /ask: Takes a URL to a PDF and a query string, and returns the most relevant answers from the PDF content.
    GET /results: Returns the last results from the previous query

**Example:**

_1.POST /ask:_
```
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{"url": "http://example.com/document.pdf", "query": "Yapay zeka nedir?"}'
```
_2.GET /results:_
```
curl -X GET "http://localhost:8000/results"
```

    


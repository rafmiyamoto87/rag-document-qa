# Document Q&A with RAG

Ask questions about your documents using AI and semantic search.

## What it does

Upload a PDF or text file and ask questions. The system finds relevant sections using embeddings and provides accurate answers based only on the document content.

## Tech stack

- Python
- OpenAI API (GPT-4o-mini, embeddings)
- tiktoken
- PyPDF2
- NumPy

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Create `.env`:
```
OPENAI_API_KEY=your-key-here
```

Run:
```bash
python rag_qa.py
```

## Example
```
Document path: handbook.pdf
Loaded 5420 chars
Created 5 chunks
Creating 5 embeddings...
  5/5
Ready!

Q: What's the vacation policy?

Relevant chunks:
  Chunk 2: 0.856
  Chunk 1: 0.723
  Chunk 0: 0.634

A: Employees receive 15 days PTO per year.

Q: quit
Save? (y/n): y
Filename: session.txt
Saved to session.txt
```

## How it works

1. Splits document into 1000-token chunks with 200-token overlap
2. Converts each chunk to a vector using OpenAI embeddings
3. For questions, finds most similar chunks using cosine similarity
4. Sends only relevant chunks to GPT for answers

## Author

Rafael Miyamoto
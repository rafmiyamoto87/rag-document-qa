# Document Q&A System with RAG

Ask questions about your documents using AI and semantic search.

## What It Does

- Upload PDF or TXT documents
- Ask questions in plain English
- Get accurate answers based only on the document content
- Uses RAG (Retrieval Augmented Generation) to find relevant sections

## Technologies

- Python
- OpenAI API (GPT-4o-mini, embeddings)
- tiktoken (token counting)
- PyPDF2 (PDF reading)
- NumPy (similarity calculations)

## Setup

1. Install dependencies:
```bash
pip install openai python-dotenv PyPDF2 tiktoken numpy
```

2. Create `.env` file:
```
OPENAI_API_KEY=your-key-here
```

3. Run:
```bash
python rag_qa.py
```

## Example
```
Enter document path: handbook.pdf
✅ Loaded 5,000 characters
📊 Created 5 chunks
🧠 Creating embeddings...
✅ Ready!

Question: What's the vacation policy?
Answer: Employees get 15 days PTO per year...

Question: quit
```

## What I Learned

- Building RAG systems from scratch
- Working with OpenAI embeddings API
- Semantic search with vector similarity
- Token-based text chunking
```
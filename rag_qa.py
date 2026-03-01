from openai import OpenAI
from dotenv import load_dotenv
import os
import PyPDF2
import tiktoken
import numpy as np

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def read_pdf(filepath):
    """Extract text from PDF file"""
    text_parts = []
    with open(filepath, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text_parts.append(page.extract_text() or '')
    return "\n\n".join(text_parts)


def read_txt(filepath):
    """Read text file"""
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()


def chunk_text(text: str, max_tokens: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping token-based chunks"""
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    
    print(f"Total tokens: {len(tokens)}")
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(encoder.decode(chunk_tokens))
        start = end - overlap
    
    return chunks


def get_embedding(text: str) -> list[float]:
    """Convert text to embedding vector"""
    text = text.replace("\n", " ").strip()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def create_chunk_embeddings(chunks: list[str]) -> list[list[float]]:
    """Generate embeddings for all chunks"""
    print(f"Creating embeddings for {len(chunks)} chunks...")
    embeddings = []
    
    for i, chunk in enumerate(chunks):
        embeddings.append(get_embedding(chunk))
        
        if (i + 1) % 5 == 0 or (i + 1) == len(chunks):
            print(f"Progress: {i + 1}/{len(chunks)}")
    
    return embeddings


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    return dot_product / (norm1 * norm2)


def find_relevant_chunks(
    question: str,
    chunks: list[str],
    chunk_embeddings: list[list[float]],
    top_n: int = 3
) -> list[str]:
    """Find most relevant chunks using semantic search"""
    question_embedding = get_embedding(question)
    
    similarities = []
    for i, chunk_emb in enumerate(chunk_embeddings):
        score = cosine_similarity(question_embedding, chunk_emb)
        similarities.append((i, score))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nMost relevant chunks:")
    for i, score in similarities[:top_n]:
        print(f"  Chunk {i}: {score:.3f}")
    
    return [chunks[i] for i, _ in similarities[:top_n]]


def show_stats(content: str, chunks: list, embeddings: list, qa_count: int):
    """Display document statistics"""
    print(f"\nDocument Statistics:")
    print(f"  Characters: {len(content):,}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Embeddings: {len(embeddings)}")
    print(f"  Questions asked: {qa_count}\n")


def save_session(qa_history: list, filepath: str):
    """Save Q&A session to file"""
    if not qa_history:
        print("No questions to save")
        return
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("Document Q&A Session\n")
        f.write("=" * 50 + "\n\n")
        
        for i, (question, answer) in enumerate(qa_history, 1):
            f.write(f"Q{i}: {question}\n")
            f.write(f"A{i}: {answer}\n\n")
    
    print(f"Saved to {filepath}")


def main():
    """Main program loop"""
    filepath = input("Enter document path: ")
    
    if not filepath.strip():
        print("No path entered")
        return
    
    # Load document
    try:
        if filepath.endswith('.pdf'):
            content = read_pdf(filepath)
        elif filepath.endswith('.txt'):
            content = read_txt(filepath)
        else:
            print("Only .txt or .pdf files supported")
            return
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return
    except Exception as e:
        print(f"Error: {e}")
        return
    
    if not content or not content.strip():
        print("Empty file")
        return
    
    print(f"Loaded {len(content):,} characters\n")
    
    # Process document
    chunks = chunk_text(content, max_tokens=1000, overlap=200)
    print(f"Created {len(chunks)} chunks\n")
    
    chunk_embeddings = create_chunk_embeddings(chunks)
    
    qa_history = []
    
    print(f"\nReady! Commands: ask questions, 'stats', 'quit'\n")
    
    # Q&A loop
    while True:
        question = input("Question: ")
        
        if question.lower() == 'quit':
            break
        
        if question.lower() == 'stats':
            show_stats(content, chunks, chunk_embeddings, len(qa_history))
            continue
        
        if not question.strip():
            continue
        
        relevant_chunks = find_relevant_chunks(question, chunks, chunk_embeddings, top_n=3)
        context = "\n\n---\n\n".join(relevant_chunks)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Answer based ONLY on the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ],
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        qa_history.append((question, answer))
        
        print(f"\nAnswer: {answer}\n")
    
    # Save option
    if qa_history:
        save_choice = input("\nSave session? (yes/no): ")
        if save_choice.lower() == 'yes':
            filename = input("Filename: ")
            save_session(qa_history, filename)


if __name__ == "__main__":
    main()
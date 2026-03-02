from openai import OpenAI
from dotenv import load_dotenv
import os
import PyPDF2
import tiktoken
import numpy as np

class DocumentQA:
    """RAG-based document Q&A system"""

    def __init__(self, api_key=None):
        """Initialize the Q&A system"""
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)

        # Document data
        self.filepath = None
        self.content = None
        self.chunks = []
        self.embeddings = []

        # Session data
        self.history = []

    def load(self, filepath):
        """Load and process a document"""
        self.filepath = filepath
        
        # Read file
        try:
            if filepath.endswith(".pdf"):
                self.content = self._read_pdf(filepath)
            elif filepath.endswith(".txt"):
                self.content = self._read_txt(filepath)
            else:
                raise ValueError("Only .pdf or .txt files")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if not self.content.strip():
            raise ValueError("Empty file")
        
        print(f"Loaded {len(self.content)} chars")

        # Process
        self._chunk()
        self._create_embeddings()
        
        print("Ready!\n")
        
    def ask(self, question):
        """Ask a question about the document"""
        if not self.chunks:
            return "No document loaded"

        # Find relevant chunks
        relevant = self._find_relevant(question)
        context = "\n\n".join(relevant)
        
        # Get answer
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Answer based only on context"
                },
                
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQ: {question}"
                }
            ],
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        self.history.append((question, answer))

        return answer
    
    def save_session(self, filename):
        """Save Q&A history"""
        if not self.history:
            print("Np history")
            return
        
        with open(filename, 'w') as file:
            file.write("Q&A Session\n")
            file.write("=" * 40 + "\n\n")
            for i, (question, anwser) in enumerate(self.history, 1):
                file.write(f"Q{i}: {question}\n")
                file.write(f"A{i}: {anwser}\n\n")
        
        print(f"Saved to {filename}")
    
    def stats(self):
        """Show statistics"""
        print("\nStats:")
        print(f"    File: {self.filepath}")
        print(f"    Characters: {len(self.content) if self.content else 0:,}")
        print(f"Question: {len(self.history)}\n")

    # Private methods
    
    def _read_pdf(self, filepath):
        """Read PDF file"""
        parts = []
        with open(filepath, "rb") as file:
            reader = PyPDF2.PdfFileReader(file)
            for page in reader.pages:
                parts.append(page.extract_text or '')
        return "\n\n".join(parts)

    def _read_txt(self, filepath):
        """Read text file"""
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
        
    def _chunk(self, max_tokens=1000, overlap=200):
        """Split text into chunks"""
        encoder = tiktoken.get_encoding("cl100k_base")
        tokens = encoder.encode(self.content)
        print(f"Tokens: {len(tokens)}")

        self.chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            self.chunks.append(encoder.decode(chunk_tokens))

            start = end - overlap
        
        print(f"Chunks: {len(self.chunks)}")
        
    def _get_embedding(self, text):
        """Get embedding for text"""
        text = text.replace("\n", " ").strip()
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def _create_embeddings(self):
        """Create embeddings for all chunks"""
        print("Creating embeddings...")
        self.embeddings = []
        
        for i, chunk in enumerate(self.chunks):
            emb = self._get_embedding(chunk)
            self.embeddings.append(emb)
            if ( i + 1) % 5 == 0:
                print(f"    {i + 1}/{len(self.chunks)}")

    def _find_relevant(self, question, n=3):
        """Find most relevant chunks"""
        question_emb = self._get_embedding(question)

        scores = []
        for i, emb in enumerate(self.embeddings):
            score = self._similarity(question_emb, emb)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        print("\nRelevant chunks:")
        for i, score in scores[:n]:
            print(f"    Chunk {i}: {score:.3f}")

        return [self.chunks[i] for i, _ in scores[:n]]
    
    def _similarity(self, v1, v2):
        """Calculate cosine similarity"""
        v1 = np.array(v1)
        v2 = np.array(v2)

        return np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
def main():
    """Main program"""
    qa = DocumentQA()
    
    # Load document
    path = input("Document: ")

    try:
        qa.load(path)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Q&A loop
    while True:
        question = input("\nQ: ")

        if question.lower() == 'quit':
            break
        
        if question.lower() == 'stats':
            qa.stats()
            continue
        
        if not question.strip():
            continue
        
        answer = qa.ask(question)
        print(f"\nA: {answer}")

    # Save
    if qa.history:
        save = input("\nSave? (y/n): ")
        if save.lower() == 'y':
            filename = input("Filename: ")
            qa.save_session(filename)
            
if __name__ == '__main__':
    main()
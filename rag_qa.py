# ============================================
# IMPORTS - Libraries we need
# ============================================
from openai import OpenAI          # To call OpenAI API for embeddings and chat
from dotenv import load_dotenv     # To load .env file with API key
import os                          # To get environment variables
import PyPDF2                      # To read PDF files
import tiktoken                    # To count tokens (OpenAI's text units)
import numpy as np                 # For math operations (cosine similarity)

# ============================================
# SETUP - Load API key and create client
# ============================================
load_dotenv()  # Loads .env file - makes OPENAI_API_KEY available
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Creates client object we'll use to call API


# ============================================
# FUNCTION 1: Read PDF Files
# ============================================
def read_pdf(filepath):
    """
    Read a PDF file and extract all text
    
    Args:
        filepath: Path to PDF file (e.g., "document.pdf")
        
    Returns:
        str: All text from PDF joined together
        
    Called by: Main program when user provides .pdf file
    """
    text_parts = []  # Empty list to collect text from each page
    
    with open(filepath, "rb") as file:  # "rb" = read binary (PDFs are binary files)
        reader = PyPDF2.PdfReader(file)  # Create PDF reader object
        
        # Loop through each page
        for page in reader.pages:  # reader.pages = list of all pages
            # Extract text from this page, or use '' if extraction fails
            text_parts.append(page.extract_text() or '')
            # text_parts now looks like: ["page 1 text", "page 2 text", ...]
    
    # Join all pages with double newlines between them
    return "\n\n".join(text_parts)  # Returns one big string with all text


# ============================================
# FUNCTION 2: Read Text Files
# ============================================
def read_txt(filepath):
    """
    Read a text file
    
    Args:
        filepath: Path to text file (e.g., "document.txt")
        
    Returns:
        str: Contents of the file
        
    Called by: Main program when user provides .txt file
    """
    with open(filepath, "r", encoding="utf-8") as file:  # "r" = read mode
        return file.read()  # Returns entire file as one string


# ============================================
# FUNCTION 3: Chunk Text by Tokens
# ============================================
def chunk_text(text: str, max_tokens: int = 1000, overlap: int = 200) -> list[str]:
    """
    Split text into smaller chunks based on tokens
    
    Args:
        text: Full document text to split
        max_tokens: Maximum tokens per chunk (default 1000)
        overlap: How many tokens to overlap between chunks (default 200)
        
    Returns:
        list of strings: Each string is one chunk
        
    Called by: Main program after loading document
    
    Why overlap? So context isn't lost at chunk boundaries
    Example: "The CEO announced... [chunk1 ends] ...a new product [chunk2 starts]"
             With overlap, both chunks include some shared context
    
    Prints: Total token count
    """
    encoder = tiktoken.get_encoding("cl100k_base")  # Get tokenizer (converts text to tokens)
    
    tokens = encoder.encode(text)  # Convert entire text to list of token IDs
    # Example: "hello world" → [15339, 1917] (numbers representing the words)
    
    print(f"Total tokens: {len(tokens)}")  # Shows how long the document is in tokens
    
    chunks = []  # Empty list to store chunks
    start = 0    # Starting position in token list
    
    # Loop until we've processed all tokens
    while start < len(tokens):
        end = start + max_tokens  # End position for this chunk
        
        chunk_tokens = tokens[start:end]  # Get slice of tokens for this chunk
        # Example: If start=0, end=1000, gets tokens[0:1000]
        
        chunk_text = encoder.decode(chunk_tokens)  # Convert tokens back to text
        chunks.append(chunk_text)  # Add this chunk to our list
        # chunks now looks like: ["chunk 1 text", "chunk 2 text", ...]
        
        start = end - overlap  # Move start forward (with overlap)
        # If end=1000, overlap=200, next start=800
        # This means tokens 800-1000 appear in BOTH chunks
    
    return chunks  # Returns list of chunk strings


# ============================================
# FUNCTION 4: Get Embedding for Text
# ============================================
def get_embedding(text: str) -> list[float]:
    """
    Convert text into a vector (list of numbers representing meaning)
    
    Args:
        text: Text to convert
        
    Returns:
        list of floats: Vector of 1536 numbers
        
    Called by: create_chunk_embeddings() and find_relevant_chunks()
    
    What it does: Turns text like "dog" into [0.23, -0.45, 0.67, ..., 0.12]
                  Similar meanings get similar vectors
    """
    text = text.replace("\n", " ").strip()  # Clean text: remove newlines, trim spaces
    
    # Call OpenAI embeddings API
    response = client.embeddings.create(
        model="text-embedding-3-small",  # Cheaper embedding model
        input=text
    )
    
    # Extract the vector from response
    # response.data[0].embedding is a list of 1536 floats
    return response.data[0].embedding


# ============================================
# FUNCTION 5: Create Embeddings for All Chunks
# ============================================
def create_chunk_embeddings(chunks: list[str]) -> list[list[float]]:
    """
    Create embedding vectors for all chunks
    
    Args:
        chunks: List of text chunks
        
    Returns:
        list of vectors: One vector per chunk
        
    Called by: Main program after chunking
    
    Prints: Progress every 5 chunks
    
    Example output:
        🧠 Creating embeddings for 12 chunks...
           Progress: 5/12 (42%)
           Progress: 10/12 (83%)
           Progress: 12/12 (100%)
        ✅ Created 12 embeddings
    """
    print(f"🧠 Creating embeddings for {len(chunks)} chunks...")
    embeddings = []  # Empty list to store all embeddings
    
    # Loop through each chunk with index
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)  # Get vector for this chunk
        embeddings.append(emb)       # Add to list
        # embeddings now looks like: [[vec1], [vec2], [vec3], ...]
        
        # Show progress every 5 chunks or at the end
        if (i + 1) % 5 == 0 or (i + 1) == len(chunks):
            percentage = ((i + 1) / len(chunks)) * 100
            print(f"   Progress: {i + 1}/{len(chunks)} ({percentage:.0f}%)")
    
    print(f"✅ Created {len(embeddings)} embeddings")
    return embeddings  # Returns list of vectors


# ============================================
# FUNCTION 6: Calculate Cosine Similarity
# ============================================
def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate how similar two vectors are
    
    Args:
        vec1: First vector (list of numbers)
        vec2: Second vector (list of numbers)
        
    Returns:
        float: Similarity score (0 to 1, higher = more similar)
        
    Called by: find_relevant_chunks()
    
    Example:
        vec1 = embedding for "dog"
        vec2 = embedding for "puppy"
        similarity = 0.92 (very similar!)
        
        vec1 = embedding for "dog"
        vec2 = embedding for "car"
        similarity = 0.23 (not similar)
    """
    vec1 = np.array(vec1)  # Convert to numpy array for math
    vec2 = np.array(vec2)
    
    # Cosine similarity formula:
    # similarity = (vec1 · vec2) / (||vec1|| * ||vec2||)
    
    dot_product = np.dot(vec1, vec2)      # vec1 · vec2 (dot product)
    norm1 = np.linalg.norm(vec1)          # ||vec1|| (magnitude/length)
    norm2 = np.linalg.norm(vec2)          # ||vec2|| (magnitude/length)
    
    return dot_product / (norm1 * norm2)  # Returns similarity score


# ============================================
# FUNCTION 7: Find Most Relevant Chunks
# ============================================
def find_relevant_chunks(
    question: str, 
    chunks: list[str], 
    chunk_embeddings: list[list[float]], 
    top_n: int = 3
) -> list[str]:
    """
    Find the chunks most relevant to a question
    
    Args:
        question: User's question
        chunks: All document chunks (text)
        chunk_embeddings: Embeddings for all chunks (vectors)
        top_n: How many chunks to return (default 3)
        
    Returns:
        list of strings: The most relevant chunks
        
    Called by: Main program for each question
    
    Prints: Similarity scores for top chunks
    
    Example output:
        🔍 Most relevant chunks:
           Chunk 2: 0.847
           Chunk 5: 0.821
           Chunk 1: 0.723
    """
    # Step 1: Convert question to vector
    question_embedding = get_embedding(question)
    
    # Step 2: Calculate similarity between question and each chunk
    similarities = []  # Will store (chunk_index, score) tuples
    
    for i, chunk_emb in enumerate(chunk_embeddings):
        score = cosine_similarity(question_embedding, chunk_emb)
        similarities.append((i, score))
        # similarities looks like: [(0, 0.45), (1, 0.89), (2, 0.67), ...]
        # Meaning: chunk 0 has score 0.45, chunk 1 has score 0.89, etc.
    
    # Step 3: Sort by score (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    # After sort: [(1, 0.89), (2, 0.67), (0, 0.45), ...]
    # Now chunk 1 (highest score) is first
    
    # Step 4: Get top N chunks
    top_chunks = []  # Will store the actual chunk texts
    
    print(f"\n🔍 Most relevant chunks:")
    for i, score in similarities[:top_n]:  # Loop through top N
        print(f"   Chunk {i}: {score:.3f}")  # Print which chunk and its score
        top_chunks.append(chunks[i])         # Add the actual text of that chunk
        # top_chunks looks like: ["text of chunk 1", "text of chunk 2", ...]
    
    return top_chunks  # Returns list of most relevant chunk texts


# ============================================
# FUNCTION 8: Show Statistics
# ============================================
def show_stats(content: str, chunks: list, embeddings: list, qa_count: int):
    """
    Display document statistics
    
    Args:
        content: Original document text
        chunks: List of chunks
        embeddings: List of embeddings
        qa_count: Number of questions asked
        
    Called by: Main program when user types 'stats'
    
    Prints:
        📊 Document Statistics:
           Characters: 8,420
           Chunks: 12
           Embeddings: 12
           Questions asked: 5
    """
    print("\n📊 Document Statistics:")
    print(f"   Characters: {len(content):,}")      # :, adds thousand separators
    print(f"   Chunks: {len(chunks)}")
    print(f"   Embeddings: {len(embeddings)}")
    print(f"   Questions asked: {qa_count}\n")


# ============================================
# FUNCTION 9: Save Q&A Session
# ============================================
def save_session(qa_history: list, filepath: str):
    """
    Save question and answer history to a file
    
    Args:
        qa_history: List of (question, answer) tuples
        filepath: Where to save (e.g., "session.txt")
        
    Called by: Main program when user chooses to save after quitting
    
    Creates file like:
        Document Q&A Session
        ==================================================
        
        Q1: What is the return policy?
        A1: The return policy allows returns within 30 days...
        
        Q2: How many employees?
        A2: The company has 150 employees...
    """
    if not qa_history:  # If list is empty
        print("No questions to save")
        return
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("Document Q&A Session\n")
        f.write("=" * 50 + "\n\n")
        
        # Loop through Q&A pairs with numbering
        for i, (question, answer) in enumerate(qa_history, 1):
            # qa_history looks like: [("Q1 text", "A1 text"), ("Q2 text", "A2 text")]
            # enumerate adds index: (1, ("Q1 text", "A1 text")), (2, ("Q2 text", "A2 text"))
            f.write(f"Q{i}: {question}\n")
            f.write(f"A{i}: {answer}\n\n")
    
    print(f"✅ Saved to {filepath}")


# ============================================
# MAIN PROGRAM STARTS HERE
# ============================================

# Step 1: Get file path from user
filepath = input("Enter document path: ")

if not filepath.strip():  # If user just pressed Enter
    print("No path entered")
    exit()

# Step 2: Load the document
try:
    if filepath.endswith('.pdf'):
        content = read_pdf(filepath)  # Calls read_pdf function
    elif filepath.endswith('.txt'):
        content = read_txt(filepath)  # Calls read_txt function
    else:
        print("Only .txt or .pdf")
        exit()
        
except FileNotFoundError:
    print(f"File not found: {filepath}")
    exit()
except Exception as e:
    print(f"Error: {e}")
    exit()

# Step 3: Validate content isn't empty
if not content or not content.strip():
    print("Empty file")
    exit()

print(f"✅ Loaded {len(content):,} characters\n")
# Example output: "✅ Loaded 8,420 characters"

# ============================================
# PROCESS DOCUMENT (BEFORE LOOP - DO ONCE)
# ============================================

# Step 4: Chunk the document
chunks = chunk_text(content, max_tokens=1000, overlap=200)
# chunks is now a list like: ["chunk 1 text...", "chunk 2 text...", ...]
print(f"📊 Created {len(chunks)} chunks\n")
# Example output: "📊 Created 5 chunks"

# Step 5: Create embeddings for all chunks
chunk_embeddings = create_chunk_embeddings(chunks)
# chunk_embeddings is now: [[vec1], [vec2], [vec3], ...]
# Same length as chunks - one vector per chunk

# Step 6: Initialize Q&A history tracker
qa_history = []  # Empty list - will store (question, answer) tuples

print(f"\n✅ Ready! Commands: ask questions, 'stats', 'quit'\n")

# ============================================
# QUESTION LOOP (RUNS MANY TIMES)
# ============================================

while True:  # Infinite loop until user types 'quit'
    question = input("Question: ")
    
    # Check for quit command
    if question.lower() == 'quit':
        print("Goodbye!")
        break  # Exit the loop
    
    # Check for stats command
    if question.lower() == 'stats':
        show_stats(content, chunks, chunk_embeddings, len(qa_history))
        continue  # Skip to next loop iteration
    
    # Skip empty questions
    if not question.strip():
        continue  # Skip to next loop iteration
    
    # Step 7: Find relevant chunks for THIS question
    relevant_chunks = find_relevant_chunks(question, chunks, chunk_embeddings, top_n=3)
    # relevant_chunks is now: ["most relevant chunk", "2nd most relevant", "3rd most relevant"]
    # These are the actual text chunks, not the vectors
    
    # Step 8: Combine relevant chunks into one context string
    context = "\n\n---\n\n".join(relevant_chunks)
    # context is now one string with all 3 chunks separated by "---"
    
    # Step 9: Ask OpenAI with ONLY the relevant chunks
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Answer based ONLY on the provided context. If not in context, say 'I don't know based on this document.'"
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        temperature=0.3  # Lower temperature = more focused answers
    )
    
    # Step 10: Extract answer from response
    answer = response.choices[0].message.content
    # answer is a string with AI's response
    
    # Step 11: Save to history
    qa_history.append((question, answer))
    # qa_history now looks like: [("Q1", "A1"), ("Q2", "A2"), ...]
    
    # Step 12: Print the answer
    print(f"\nAnswer: {answer}\n")

# ============================================
# AFTER LOOP ENDS (USER TYPED 'quit')
# ============================================

if qa_history:  # If user asked at least one question
    save_choice = input("\n💾 Save this session? (yes/no): ")
    
    if save_choice.lower() == 'yes':
        filename = input("Filename: ")
        save_session(qa_history, filename)
        # Creates a text file with all Q&A pairs
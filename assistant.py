import os
import json
import re
from typing import List, Dict, Optional
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize clients
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index(os.getenv('PINECONE_INDEX', 'spiritual'))

# Load prompt templates
def load_prompts() -> Dict:
    with open('prompts.json', 'r') as f:
        return json.load(f)

PROMPTS = load_prompts()

# Sacred mode filter
def is_sacred_input(text: str) -> bool:
    # List of words to check for (can be expanded)
    profanity = ['profanity1', 'profanity2']  # Add actual words to check
    sarcasm_indicators = ['yeah right', 'sure thing', 'whatever', 'duh']
    
    # Check for profanity
    if any(word in text.lower() for word in profanity):
        return False
    
    # Check for sarcasm indicators
    if any(indicator in text.lower() for indicator in sarcasm_indicators):
        return False
    
    # Check for excessive punctuation (indicating sarcasm)
    if re.search(r'[!?]{2,}', text):
        return False
    
    return True

def get_relevant_chunks(query: str, top_k: int = 5) -> List[Dict]:
    """Retrieve relevant chunks from Pinecone."""
    # Get query embedding
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    query_embedding = response.data[0].embedding
    
    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    return results.matches

def format_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into context for GPT."""
    context = []
    for chunk in chunks:
        metadata = chunk.metadata
        context.append(f"Source: {metadata['source']}, Page: {metadata['page']}\n{metadata['text']}\n")
    return "\n".join(context)

def generate_response(query: str, tone: str = "teaching", context: str = "") -> str:
    """Generate response using GPT-4 with the selected tone."""
    if tone not in PROMPTS:
        tone = "teaching"  # Default to teaching tone
    
    system_prompt = PROMPTS[tone]["system"]
    
    messages = [
        {"role": "system", "content": f"{system_prompt}\n\nContext:\n{context}"},
        {"role": "user", "content": query}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content

def process_query(query: str, tone: str = "teaching") -> str:
    """Process a user query and return a response."""
    # Check sacred mode
    if not is_sacred_input(query):
        return "This sacred assistant is reserved for spiritual refinement. Please reframe your question with sincerity."
    
    # Get relevant chunks
    chunks = get_relevant_chunks(query)
    if not chunks:
        return "I apologize, but I couldn't find relevant information to answer your question."
    
    # Format context
    context = format_context(chunks)
    
    # Generate response
    response = generate_response(query, tone, context)
    
    return response

def main():
    print("Spiritual Assistant (Type 'exit' to quit)")
    print("Available tones:", ", ".join(PROMPTS.keys()))
    
    while True:
        # Get tone
        tone = input("\nSelect tone (default: teaching): ").lower()
        if not tone or tone not in PROMPTS:
            tone = "teaching"
        
        # Get query
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            break
        
        # Process and display response
        response = process_query(query, tone)
        print("\nResponse:", response)

if __name__ == "__main__":
    main() 
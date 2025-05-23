import os
import json
import re
import sys
import platform
from typing import List, Dict

# Handle Windows-specific readline issue
if platform.system() == 'Windows':
    try:
        import readline
    except ImportError:
        # Create a dummy readline module for Windows
        class DummyReadline:
            def __init__(self):
                pass
            def get_completer(self):
                return None
            def set_completer(self, *args):
                pass
        sys.modules['readline'] = DummyReadline()

import streamlit as st
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
        # Get text from the chunk's values or metadata
        text = metadata.get('text', '')  # Use get() to avoid KeyError
        if not text:  # If text is not in metadata, try to get it from the chunk
            text = chunk.get('text', '')
        
        context.append(f"Source: {metadata.get('source', 'Unknown')}, Page: {metadata.get('page', 'Unknown')}\n{text}\n")
    return "\n".join(context)

def generate_response(query: str, tone: str = "scriptural", context: str = "") -> str:
    """Generate response using GPT-4 with the selected tone."""
    if tone not in PROMPTS:
        tone = "scriptural"  # Default to scriptural tone
    
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

def process_query(query: str, tone: str = "scriptural") -> str:
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

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Spiritual Assistant",
        page_icon="✨",
        layout="centered"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stTextInput>div>div>input {
            font-size: 18px;
        }
        .stButton>button {
            width: 100%;
            font-size: 18px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("✨ Spiritual Assistant")
    st.markdown("---")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for tone selection
    with st.sidebar:
        st.header("Settings")
        tone = st.selectbox(
            "Select Tone",
            options=list(PROMPTS.keys()),
            index=0  # Default to "scriptural"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This assistant is built on The Laws of Creation, 
        a multi-dimensional spiritual framework.
        
        Choose your preferred tone of interaction:
        - **Scriptural**: Reverent and doctrinal
        - **Teaching**: Clear and instructive
        - **Conversational**: Warm and friendly
        - **Philosophical**: Deep and reflective
        """)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = process_query(prompt, tone)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 
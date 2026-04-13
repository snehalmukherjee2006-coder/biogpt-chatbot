import streamlit as st
from transformers import pipeline

# 1. Page Configuration
st.set_page_config(
    page_title="BioGPT Medical Assistant",
    page_icon="🧬",
    layout="centered"
)

# 2. Sidebar with Logo and Info
with st.sidebar:
    try:
        st.image("logo.png", width=200)
    except:
        st.warning("Logo 'logo.png' not found. Please check the file path.")
    
    st.title("Navigation & Info")
    st.markdown("---")
    st.header("How to use")
    st.write("""
    1. Enter your medical query in the chat box.
    2. Review the AI-generated insights.
    3. *Note: For educational purposes only.*
    """)
    st.markdown("---")
    st.caption("Created by: **Snehal Mukherjee, Renisha Gracelin, Bhumiga, Lavanaya**")

# 3. Main Title
st.title("🧬 BioGPT Medical Assistant")
st.markdown("---")

# 4. Load the Model (Cached for performance)
@st.cache_resource
def load_model():
    # We use 'text-generation' but will format the input like a question
    return pipeline('text-generation', model='google/flan-t5-large')

with st.spinner("Loading BioGPT model... please wait."):
    chatbot = load_model()

# 5. Initialize Chat History
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# 6. User Input Logic
user_input = st.chat_input('Type your health or biology question here...')

if user_input:
    # Add user message to history
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    # Generate AI Response
    with st.chat_message('assistant'):
        with st.spinner('BioGPT is thinking...'):
            # STRATEGY: Use a structured prompt to force a direct answer
            structured_prompt = f"Question: {user_input}\nAnswer:"
            
            # Request generation with better parameters
            raw_output = chatbot(
                structured_prompt, 
                max_new_tokens=150, 
                do_sample=True,      # Allows for more natural language
                temperature=0.6,     # Lower temperature = more factual/focused
                top_k=50,
                num_return_sequences=1
            )
            
            # CLEANING: Extract only the generated answer text
            full_text = raw_output[0]['generated_text']
            
            # This line removes the "Question: ... Answer:" part from the display
            clean_answer = full_text.split("Answer:")[-1].strip()
            
            # If the model gives an empty or broken response, provide a fallback
            if not clean_answer:
                clean_answer = "I'm sorry, I couldn't generate a clear answer. Could you please rephrase your question?"

            st.write(clean_answer)
            
            # Save assistant response to history
            st.session_state.messages.append({'role': 'assistant', 'content': clean_answer})

# Footer Disclaimer
st.markdown("---")
st.caption("⚠️ **Disclaimer:** This AI tool provides general information and is not a substitute for professional medical advice, diagnosis, or treatment.")

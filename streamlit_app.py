import streamlit as st 
from transformers import pipeline 
st.set_page_config( page_title="BioGPT Chatbot", page_icon="🧬", layout="centered" ) 
st.title("🧬 BioGPT Medical Assistant") 
st.markdown("---") 
# Page settings 
st.set_page_config(page_title='BioGPT Health Chatbot', page_icon="🩺") 
# Title and description 
st.title('BioGPT Health Chatbot') 
st.caption('Ask me any health or biology question!') 
# Load the AI model (cached so it loads only once) 
@st.cache_resource 
def load_model(): 
  return pipeline('text-generation', model='microsoft/biogpt') 
chatbot = load_model() 
# Store chat history 
if 'messages' not in st.session_state: 
  st.session_state.messages = [] 
  # Show previous messages 
  for msg in st.session_state.messages: 
    with st.chat_message(msg['role']): 
      st.write(msg['content']) 
  # Get user input 
  user_input = st.chat_input('Type your health question here...') 
  if user_input: 
    # Show user message 
    st.session_state.messages.append({'role': 'user', 'content': user_input}) 
    with st.chat_message('user'): st.write(user_input) 
    # Get AI answer 
    with st.chat_message('assistant'):
       with st.spinner('BioGPT is thinking...'): 
        result = chatbot(user_input, max_new_tokens=150, do_sample=False) 
        answer = result[0]['generated_text'] 
        st.write(answer) 
        st.session_state.messages.append({'role': 'assistant', 'content': answer}) 
        with st.sidebar: 
          # This places your Canva logo at the top of the sidebar 
          st.image("logo.png") 
          st.title("Navigation & Info") 
          st.markdown("---") 
          # Instructions for the user 
          st.header("How to use") 
          st.write(""" 1. Enter your medical query in the chat box. 
                   2. Review the AI-generated insights. 
                   3. *Note: For educational purposes only.* """) 
          st.markdown("---") 
          st.caption("Created by: **Snehal Mukherjee, Renisha Gracelin, Bhumiga, Lavanaya**")

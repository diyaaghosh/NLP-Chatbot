import nltk
nltk.download('punkt')

import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import streamlit as st
import time as t

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents and model data
with open('intents.json', encoding='utf-8') as f:
    intents = json.load(f)

FILE = 'data.pth'
data = torch.load(FILE)

input_size = data["input_size"]
output_size = data["output_size"]
model_state = data["model_state"]
hidden_size = data["hidden_size"]
tags = data["tags"]
all_words = data["all_words"]

# Load trained model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Page config
st.set_page_config(page_title="Chatbot", page_icon="💙")
st.title("💙 ChatBot")
st.subheader(" Your IT Help Desk Chatbot  !! 🕵‍♀")

# Initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input
user_input = st.chat_input("Type your IT support query here...")
if(user_input=="quit"):
    st.balloons()
    st.stop()
    

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar", None)):
        st.markdown(message["content"])

# If new user input
if user_input:
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "avatar": "👤"
    })

    # NLP + model
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, input_size)
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag in intent["tag"]:
                response = random.choice(intent["responses"])
                break
    else:
        response = "OOPS!! I don't understand. Can you rephrase that?"

        
    with st.chat_message("assistant", avatar="💙"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulate stream of response with milliseconds delay
        for chunk in response.split():
            full_response += chunk + " "
            t.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "avatar": "🤖"
    })


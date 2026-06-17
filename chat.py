import nltk
nltk.download('punkt')

import random
import json
import torch
import streamlit as st
import time as t

from database import (
    save_chat,
    load_chats,
    delete_chats,
    login,
    register
)

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize



# =============================
# Device
# =============================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)



# =============================
# Login System
# =============================

if "user_id" not in st.session_state:

    st.session_state.user_id = None



if st.session_state.user_id is None:


    st.set_page_config(
        page_title="Nexas Login",
        page_icon="🔐"
    )


    st.title(
        "🔐 Nexas IT Help Desk"
    )


    mode = st.selectbox(
        "Select Option",
        [
            "Login",
            "Register"
        ]
    )


    username = st.text_input(
        "Username"
    )


    password = st.text_input(
        "Password",
        type="password"
    )



    if mode == "Register":


        if st.button("Create Account"):


            if register(
                username,
                password
            ):

                st.success(
                    "Account created. Please login."
                )

            else:

                st.error(
                    "Username already exists"
                )



    else:


        if st.button("Login"):


            user = login(
                username,
                password
            )


            if user:


                st.session_state.user_id = user

                st.rerun()


            else:

                st.error(
                    "Invalid username or password"
                )



    st.stop()





# =============================
# Load Model
# =============================

with open(
    "intents.json",
    encoding="utf-8"
) as f:

    intents = json.load(f)



data = torch.load(
    "data.pth",
    map_location=device
)


input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]

all_words = data["all_words"]
tags = data["tags"]



model = NeuralNet(
    input_size,
    hidden_size,
    output_size
).to(device)


model.load_state_dict(
    data["model_state"]
)


model.eval()





# =============================
# Page
# =============================


st.set_page_config(
    page_title="Nexas IT Help Desk",
    page_icon="💙",
    layout="wide"
)



st.title(
    "💙 Nexas IT Help Desk"
)


st.caption(
    "AI Powered Private IT Support Assistant"
)




# =============================
# Session Messages
# =============================


if "messages" not in st.session_state:


    st.session_state.messages = load_chats(
        st.session_state.user_id
    )





# =============================
# Sidebar
# =============================


with st.sidebar:


    st.write(
        "👤 User ID:",
        st.session_state.user_id
    )


    st.divider()


    if st.button(
        "🗑 Delete Chat History"
    ):


        delete_chats(
            st.session_state.user_id
        )


        st.session_state.messages = []


        st.success(
            "Chat history deleted"
        )


        st.rerun()



    if st.button(
        "Logout"
    ):


        st.session_state.clear()

        st.rerun()





# =============================
# Display History
# =============================


for message in st.session_state.messages:


    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )





# =============================
# Chat Input
# =============================


user_input = st.chat_input(
    "Describe your IT problem..."
)




if user_input:


    # User message


    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_input
        )



    save_chat(
        st.session_state.user_id,
        "user",
        user_input
    )


    st.session_state.messages.append(
        {
            "role":"user",
            "content":user_input
        }
    )





    # =============================
    # NLP Processing
    # =============================


    sentence = tokenize(
        user_input
    )


    X = bag_of_words(
        sentence,
        all_words
    )


    X = X.reshape(
        1,
        input_size
    )


    X = torch.from_numpy(
        X
    ).to(device)



    output = model(X)


    result = torch.max(
        output,
        dim=1
    )


    predicted = result[1]


    tag = tags[
        predicted.item()
    ]


    probability = torch.softmax(
        output,
        dim=1
    )[0][predicted.item()]



    st.info(
        f"🗂 Issue Category: **{tag.upper()}**"
    )





    # =============================
    # Response
    # =============================


    if probability.item() > 0.75:


        response = (
            "I am checking your issue..."
        )


        for intent in intents["intents"]:


            if tag == intent["tag"]:


                response = random.choice(
                    intent["responses"]
                )

                break



    else:


        response = (
            "Sorry, I could not understand. "
            "Please explain your issue differently."
        )






    # =============================
    # Bot Reply
    # =============================


    with st.chat_message(
        "assistant"
    ):


        placeholder = st.empty()


        text = ""


        for word in response.split():

            text += word + " "

            t.sleep(0.05)


            placeholder.markdown(
                text + "▌"
            )


        placeholder.markdown(
            text
        )



    save_chat(
        st.session_state.user_id,
        "assistant",
        response
    )


    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":response
        }
    )
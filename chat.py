import streamlit as st

# MUST be first Streamlit command
st.set_page_config(
    page_title="Nexas IT Help Desk",
    page_icon="💙",
    layout="wide"
)


import nltk
nltk.download("punkt")


import random
import json
import torch
import time as t


from database import (
    save_chat,
    load_chats,
    delete_chats,
    login,
    register
)


from model import NeuralNet
from nltk_utils import (
    bag_of_words,
    tokenize
)



# =============================
# Device
# =============================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)



# =============================
# Session User
# =============================

if "user_id" not in st.session_state:
    st.session_state.user_id = None



# =============================
# Login/Register
# =============================

if st.session_state.user_id is None:


    st.title("🔐 Nexas IT Help Desk")


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
# Load intents
# =============================

with open(
    "intents.json",
    "r",
    encoding="utf-8"
) as f:

    intents = json.load(f)





# =============================
# Load GRU Model
# =============================


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
)


model.load_state_dict(
    data["model_state"]
)


model.to(device)

model.eval()





# =============================
# UI
# =============================


st.title(
    "💙 Nexas IT Help Desk"
)


st.caption(
    "AI Powered Private IT Support Assistant"
)




# =============================
# Load Messages
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
# Show old chats
# =============================


for message in st.session_state.messages:


    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )






# =============================
# Chat
# =============================


user_input = st.chat_input(
    "Describe your IT problem..."
)



if user_input:


    # User message

    with st.chat_message("user"):

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
    # NLP
    # =============================


    sentence = tokenize(
        user_input
    )


    X = bag_of_words(
        sentence,
        all_words
    )


    # Convert numpy -> torch

    X = torch.tensor(
        X,
        dtype=torch.float32
    )


    # add batch dimension

    X = X.unsqueeze(0)


    X = X.to(device)





    # Model prediction

    with torch.no_grad():

        output = model(X)



    probabilities = torch.softmax(
        output,
        dim=1
    )


    confidence, predicted = torch.max(
        probabilities,
        dim=1
    )



    tag = tags[
        predicted.item()
    ]



    st.info(
        f"🗂 Issue Category: **{tag.upper()}**"
    )






    # =============================
    # Response
    # =============================


    if confidence.item() > 0.75:


        response = (
            "I am checking your issue..."
        )


        for intent in intents["intents"]:


            if intent["tag"] == tag:


                response = random.choice(
                    intent["responses"]
                )

                break



    else:


        response = (
            "Sorry, I could not understand. "
            "Please explain your problem again."
        )





    # =============================
    # Bot reply animation
    # =============================


    with st.chat_message(
        "assistant"
    ):


        box = st.empty()


        text = ""


        for word in response.split():


            text += word + " "

            t.sleep(0.05)


            box.markdown(
                text + "▌"
            )


        box.markdown(
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
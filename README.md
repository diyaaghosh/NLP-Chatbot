# NLP Helpdesk Chatbot

An AI-powered IT Helpdesk Chatbot built using PyTorch, Natural Language Processing (NLP), Streamlit, and SQLite. The chatbot automatically classifies user issues, provides predefined support responses, and maintains secure user-specific chat histories through authentication and database storage.

---

## Features

* User Registration and Login
* Password Hashing for Secure Authentication
* Private User Chat History
* Delete Chat History Functionality
* NLP-based Intent Recognition
* Bag-of-Words Text Vectorization
* Feedforward Neural Network (FFNN)
* Streamlit Chat Interface
* SQLite Database Integration
* Easily Extendable Intents Dataset

---

## Technology Stack

* Python
* PyTorch
* NLTK
* Streamlit
* SQLite

---

## Project Architecture

```text
User
 |
 v
Streamlit Interface
 |
 v
NLP Preprocessing
(Tokenization + Stemming)
 |
 v
Bag of Words
 |
 v
PyTorch Neural Network
 |
 v
Intent Prediction
 |
 v
Response Generation
 |
 v
SQLite Database Storage
```

---

## NLP Pipeline

Example Input:

```text
"My laptop is not connecting to WiFi"
```

### Step 1: Tokenization

```text
["My", "laptop", "is", "not", "connecting", "to", "WiFi"]
```

### Step 2: Lowercase and Stemming

```text
["my", "laptop", "is", "not", "connect", "to", "wifi"]
```

### Step 3: Bag of Words

```text
[0,1,0,0,1,0,1...]
```

### Step 4: Intent Classification

```text
Network Issue
```

---

## Neural Network Architecture

The chatbot uses a Feedforward Neural Network (FFNN):

### Input Layer

Receives Bag-of-Words vectors.

### Hidden Layers

Processes extracted features using ReLU activation.

### Output Layer

Predicts the intent using Softmax probabilities.

---

## Database Design

### Users Table

Stores user authentication data.

| Column   | Description             |
| -------- | ----------------------- |
| id       | User ID                 |
| username | Login Username          |
| password | SHA-256 Hashed Password |

### Chats Table

Stores user conversations.

| Column  | Description      |
| ------- | ---------------- |
| id      | Message ID       |
| user_id | Linked User      |
| role    | User / Assistant |
| message | Chat Message     |
| time    | Timestamp        |

---

## Authentication Flow

### Registration

```text
User
 |
 v
Password
 |
 v
SHA-256 Hash
 |
 v
SQLite Database
```

### Login

```text
Entered Password
 |
 v
SHA-256 Hash
 |
 v
Compare with Stored Hash
 |
 v
Login Success
```

---

## Privacy

Each chat message is linked to a unique user ID.

```sql
SELECT *
FROM chats
WHERE user_id = ?
```

This ensures users can only access their own conversations.

---

## Chat History Management

### Save Chat

Every message is automatically stored in SQLite.

### Load Chat

Previous conversations are loaded after login.

### Delete Chat

Users can delete their own chat history.

```sql
DELETE FROM chats
WHERE user_id = ?
```

---

## Project Structure

```text
NLP-Chatbot/
│
├── chat.py
├── train.py
├── model.py
├── nltk_utils.py
├── database.py
├── intents.json
├── data.pth
├── users.db
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/NLP-Chatbot.git
cd NLP-Chatbot
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download NLTK Resources

```python
import nltk
nltk.download("punkt")
```

---

## Train Model

```bash
python train.py
```

This generates:

```text
data.pth
```

---

## Run Application

```bash
streamlit run chat.py
```

---

## Example Workflow

```text
Register
   |
   v
Login
   |
   v
Chat with Bot
   |
   v
Messages Stored in SQLite
   |
   v
Refresh Page
   |
   v
History Restored
   |
   v
Delete History (Optional)
```

---

## Future Improvements

* Sentiment Analysis
* Context-Aware Conversations
* Multi-Language Support
* Admin Dashboard
* Password Reset Functionality
* Deployment on Streamlit Cloud

---




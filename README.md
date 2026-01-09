# Integrated Customer Support System

An intelligent customer support system that combines actionable message detection with RAG (Retrieval-Augmented Generation) chatbot capabilities.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://cricbuzzlivestats-app.streamlit.app/](https://ntptwygoz34ofkupgb5wxi.streamlit.app/))

## 🎯 Features

- **Actionable Message Detection**: Automatically classifies if a message requires a response
- **RAG-Powered Chatbot**: Provides accurate responses using knowledge base
- **Dual Model Support**: Choose between Random Forest or LSTM for classification
- **Conversation History**: Maintains context across multiple interactions
- **Streamlit UI**: User-friendly web interface

## 🏗️ Architecture

```
User Message 
    ↓
Actionable Classifier (Random Forest / LSTM)
    ↓
├─ Actionable → RAG Chatbot → Response with Context
└─ Non-Actionable → Acknowledgment Message
```

## 📁 Project Structure

```
integrated_customer_support/
│
├── data/
│   ├── nlp_data/                          # Actionable classifier data
│   │   ├── social_media_actionable_dataset.csv
│   │   ├── messages_base.csv
│   │   └── messages_cleaned.csv
│   │
│   └── rag_data/                          # RAG system data
│       ├── raw/
│       │   └── kb.txt                     # Knowledge base
│       ├── processed/
│       │   └── chunks.json
│       └── vector_store/
│           └── faiss_index/
│
├── models/
│   └── actionable_detection/
│       ├── tfidf_vectorizer.pkl
│       ├── random_forest_model.pkl
│       └── lstm_model.h5
│
├── src/
│   ├── actionable_classifier/             # Classifier module
│   │   ├── __init__.py
│   │   ├── text_cleaner.py
│   │   └── classifier.py
│   │
│   ├── rag_system/                        # RAG module
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── data_loader.py
│   │   ├── text_preprocessing.py
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   ├── prompt_templates.py
│   │   ├── llm_gemini.py
│   │   └── rag_pipeline.py
│   │
│   └── integration/                       # Integration module
│       ├── __init__.py
│       └── unified_pipeline.py            # KEY: Combines both systems
│
├── app.py                                 # Main Streamlit app
├── build_index.py                         # Build RAG index
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd integrated_customer_support
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt

# Download spacy model
python -m spacy download en_core_web_sm
```

### 4. Set up environment variables

Create a `.env` file:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp
EMBEDDING_MODEL=models/text-embedding-004
```

### 5. Prepare your data

**For Actionable Classifier:**
- Place trained models in `models/actionable_detection/`
  - `tfidf_vectorizer.pkl`
  - `random_forest_model.pkl`
  - `lstm_model.h5`

**For RAG System:**
- Place your knowledge base at `data/rag_data/raw/kb.txt`

### 6. Build the RAG index

```bash
python build_index.py
```

This will:
- Load and preprocess the knowledge base
- Create text chunks
- Generate embeddings (may take several minutes)
- Build FAISS vector index

## 💻 Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### How It Works

1. **User sends a message**
2. **Actionable Classifier** analyzes the message
   - If **Actionable**: Routes to RAG chatbot
   - If **Non-Actionable**: Returns acknowledgment

3. **RAG Chatbot** (for actionable messages):
   - Retrieves relevant context from knowledge base
   - Generates contextual response using Gemini LLM
   - Maintains conversation history

### Example Interactions

**Actionable Messages:**
```
User: "My order hasn't arrived yet, please help"
→ Classification: Actionable (95% confidence)
→ RAG Response: [Detailed response about order tracking and policies]

User: "How do I reset my password?"
→ Classification: Actionable (98% confidence)
→ RAG Response: [Step-by-step password reset instructions]
```

**Non-Actionable Messages:**
```
User: "Thanks for your help!"
→ Classification: Non-Actionable (92% confidence)
→ Response: "You're welcome! I'm glad I could help..."

User: "I love this product!"
→ Classification: Non-Actionable (96% confidence)
→ Response: "Thank you for your positive feedback!..."
```

## 🔧 Configuration

### Classifier Settings

Choose between two classification models:
- **Random Forest**: Faster, interpretable, good for production
- **LSTM**: Deep learning, slightly more accurate

### RAG Settings

Adjust in `.env`:
```bash
CHUNK_SIZE=500              # Size of text chunks
CHUNK_OVERLAP=50            # Overlap between chunks
TOP_K_RESULTS=3             # Number of contexts to retrieve
TEMPERATURE=0.7             # LLM creativity (0-1)
MAX_TOKENS=1024             # Max response length
```

## 📊 Features in the UI

- **Classifier Model Selection**: Switch between Random Forest and LSTM
- **Classification Display**: See confidence scores and predictions
- **Context Viewer**: View retrieved knowledge base chunks
- **Relevance Scores**: See similarity scores for retrieved contexts
- **Conversation History**: Maintains context across messages
- **Statistics**: View system metrics

## 🧪 Testing

Test different message types:

**Actionable Examples:**
- "My order hasn't arrived yet"
- "How can I reset my password?"
- "The app keeps crashing"
- "Can someone assist me with login?"
- "My payment failed but money was deducted"

**Non-Actionable Examples:**
- "Thanks for your quick support!"
- "Amazing experience overall"
- "I love this product!"
- "Just sharing my thoughts"
- "The weather is nice today"

## 🔍 Key Components

### 1. Actionable Classifier (`src/actionable_classifier/`)
- Determines if message needs response
- Supports Random Forest and LSTM models
- Includes text cleaning and preprocessing

### 2. RAG System (`src/rag_system/`)
- Chunks and indexes knowledge base
- Retrieves relevant context
- Generates responses using Gemini LLM

### 3. Unified Pipeline (`src/integration/unified_pipeline.py`)
- Orchestrates both systems
- Routes messages based on classification
- Generates appropriate responses

## 🐛 Troubleshooting

### "Vector store not found"
```bash
python build_index.py
```

### "Spacy model not found"
```bash
python -m spacy download en_core_web_sm
```

### "Model files not found"
Ensure your trained classifier models are in:
```
models/actionable_detection/
├── tfidf_vectorizer.pkl
├── random_forest_model.pkl
└── lstm_model.h5
```

### API Rate Limits
The build process includes rate limiting for Gemini API. If you hit limits:
- Wait a few minutes
- Re-run `build_index.py` (it will resume)

## 📝 Notes

- First run of `build_index.py` takes time (5-15 minutes for embedding generation)
- FAISS index is saved locally and loaded instantly on subsequent runs
- Conversation history is maintained per session
- Classification models are cached for performance

## 🔐 Security

- Never commit `.env` file with API keys
- Add sensitive files to `.gitignore`
- Use environment variables for all secrets

## 🧑‍💻 Author

**Vignesh A**  
📍 India 🇮🇳  

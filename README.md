📄 Document Q&A Chatbot

A simple **Streamlit-based chatbot** that allows you to upload PDF/TXT documents and ask questions about them using semantic search and an intelligent Q&A system.

## 🚀 Features
- Upload PDF/TXT files  
- Automatic text extraction, chunking, and embedding  
- Semantic vector search  
- Interactive chat interface  
- Displays response confidence score  

## 🛠️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

📦 Project Structure
app.py
document_loader.py
chunker.py
embeddings.py
retriever.py
qa_chain.py
```

💡 How It Works
 - Upload your documents
 - System extracts text and generates chunks
 - Embeddings are created and stored
 - Vector search retrieves relevant chunks
 - Q&A model generates an answer
 - 
📜 License:
MIT License

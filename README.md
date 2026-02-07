# Legal-Document-Retrieval-System
AI-based legal document search system using SBERT semantic embeddings and Legal-BERT NER to retrieve relevant court judgments with high accuracy and fast query response.

This project is an AI-powered Legal Document Retrieval System designed to help users efficiently search and explore large collections of court judgments. Instead of relying on keyword-based search, the system uses semantic understanding and legal entity recognition to retrieve the most relevant legal documents based on the meaning of a query.

The system supports criminal law judgments and enables users to search using natural language queries such as legal sections, case types, courts, or legal issues.

🚀 Key Features

1. Semantic search over legal judgments using transformer-based embeddings
2. Named Entity Recognition (NER) to extract legal entities like statutes, courts, judges, and case numbers
3. Hybrid retrieval combining semantic similarity and entity matching
4. Fast document retrieval using FAISS indexing
5. Interactive web interface built using Streamlit
6. Direct access to original judgment PDFs from search results

🧠 Models Used

1. Sentence-BERT (SBERT)
Used for generating semantic embeddings of legal documents and user queries to capture contextual meaning.

2. BGE-base (BAAI/bge-base-en-v1.5)
A stronger embedding model used for higher-quality semantic representations and improved retrieval accuracy.

3. Legal-BERT (NER)
Used for extracting domain-specific legal entities such as statutes, courts, judges, and provisions from judgments.

🔄 How the System Works

- PDF Parsing & Preprocessing
Court judgment PDFs are converted into text and cleaned using NLP preprocessing steps.

- Semantic Embedding
Each cleaned judgment is converted into dense vector embeddings using SBERT/BGE models.

- FAISS Indexing
Embeddings are stored in a FAISS index for fast similarity-based retrieval.

- Named Entity Recognition
Legal-BERT extracts important legal entities and stores them for hybrid scoring.

- Hybrid Search
Search results are ranked using a combination of:

- Hybrid Score = (0.7 × Semantic Similarity) + (0.3 × Entity Match Score)


- User Interface
Users interact via a web UI to search, view summaries, highlighted entities, and access original PDFs.

📂 Dataset Description

- The dataset consists of Indian court judgments, primarily focused on criminal law cases.

🔹 Data Sources

1. Supreme Court of India (SCI)

https://www.sci.gov.in/judgements-case-no/

2. Manupatra Legal Database

https://www.manupatrafast.com/Defaults/training-manual-courts-judgments-caselaw-database.aspx

🔹 Dataset Structure

1. Raw PDFs of court judgments (used for final document viewing)
2. Cleaned and preprocessed text files (used for model processing and retrieval)
3. NER outputs stored in structured JSON format

⚠️ Due to dataset size and licensing restrictions, the raw datasets are not included in this repository.

🛠️ Technologies Used

1. Python
2. Hugging Face Transformers
3. Sentence-Transformers
4. FAISS
5. NLTK
6. pdfplumber
7. Streamlit
8. NumPy / Pandas

🧪 Evaluation Summary

-The system was evaluated using semantic similarity–based retrieval metrics:
-Precision@5: High relevance in top results
-Recall@5: Strong coverage of relevant documents
-F1-score: Balanced retrieval performance
-Comparative evaluation showed that BGE-base embeddings outperform MiniLM in precision, making it more suitable for legal document retrieval.

🖥️ Running the Application

1. Install dependencies:

pip install -r requirements.txt

2. Start the application:

streamlit run simple_legal_app.py

3. Open browser at:

http://localhost:8501

🔮 Future Enhancements

1. Query-based document summarization
2. Support for more legal domains (civil, constitutional, corporate law)
3. Advanced legal reasoning and citation linking
4. Role-based access and document analytics

👩‍💻 Author

Anukeerthana
Software Engineering | NLP | Legal AI|Document Retriever

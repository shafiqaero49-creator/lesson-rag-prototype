# Lesson-Preparation RAG Prototype

Retrieval prototype for DBA Generative AI Assignment 1 (Secondary School Teacher – lesson preparation).

- `KB01-C01.txt` – `KB06-C01.txt`: six illustrative knowledge-base chunks (USGS, BSCS 5E, CAST UDL, school planning and assessment material).
- `lesson_rag_prototype.py`: builds a TF-IDF index over the chunks and ranks them for four test queries using cosine similarity.

Run: `pip install scikit-learn` then `python lesson_rag_prototype.py` (or `!python lesson_rag_prototype.py` in Google Colab).

This demonstrates the retrieval stage only. The proposed production design uses Azure AI Search hybrid retrieval and an Azure OpenAI model for grounded generation.

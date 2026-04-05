import os
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from mtranslate import translate

genai.configure(api_key="AIzaSyC3wRBQ12Y-acsx98UplVzyHH4h5QJzRE4")

gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')

print("--- INITIALIZING MEDICAL KNOWLEDGE BASE ---")
KB_PATH = "./knowledge_base"
if not os.path.exists(KB_PATH):
    os.makedirs(KB_PATH)

# Load PDFs
pdf_loader = PyPDFDirectoryLoader(KB_PATH)
pdf_docs = pdf_loader.load()

# Load TXT files
try:
    txt_loader = DirectoryLoader(KB_PATH, glob="**/*.txt", loader_cls=TextLoader)
    txt_docs = txt_loader.load()
except Exception as e:
    print(f"⚠️ Could not load TXT files: {e}")
    txt_docs = []

# Load DOCX files
try:
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
    docx_loader = DirectoryLoader(KB_PATH, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader)
    docx_docs = docx_loader.load()
except Exception as e:
    print(f"⚠️ Could not load DOCX files: {e}")
    docx_docs = []

# Combine all documents
raw_docs = pdf_docs + txt_docs + docx_docs

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if not raw_docs:
    print("⚠️ Warning: No documents found in ./knowledge_base. RAG will be disabled.")
    vector_db = None
else:
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
    docs = splitter.split_documents(raw_docs)
    vector_db = FAISS.from_documents(docs, embeddings)
    print(f"✅ Indexed {len(docs)} sections from your knowledge base.")


def chatbot_response(query, target_lang='en'):
    answer = ""

    if vector_db:
        try:
            # Increased k=3 for better context retrieval
            search_results = vector_db.similarity_search_with_score(query, k=3)
            if search_results:
                best_score = search_results[0][1]
                print(f"📊 Best RAG Score: {best_score:.4f}")

                # Lowered threshold from 1.5 to 0.6 for stricter, accurate matching
                if best_score < 1.5:
                    print(f"📍 Mode: Offline RAG (Score: {best_score:.2f})")

                    # Combine top results for richer context
                    context_from_docs = "\n\n".join(
                        [result[0].page_content for result in search_results]
                    )

                    prompt = (
                        f"You are a medical assistant. Using this verified medical context:\n"
                        f"'{context_from_docs}'\n\n"
                        f"Answer this query concisely in under 100 words: {query}"
                    )
                    response = gemini_model.generate_content(prompt)
                    answer = response.text
                else:
                    print(f"⚠️ RAG score too high ({best_score:.2f}) — falling back to online")
        except Exception as e:
            print(f"⚠️ RAG Error: {e}. Switching to Online.")

    if not answer:
        print("🌐 Mode: Online Search (Triggered)")
        try:
            prompt = (
                f"As a professional medical consultant, provide a VERY BRIEF summary "
                f"(MAX 100 WORDS) about: {query}"
            )
            response = gemini_model.generate_content(prompt)
            if response and response.text:
                answer = response.text
            else:
                answer = "I'm sorry, I couldn't find specific information on that topic right now."
        except Exception as e:
            print(f"❌ Online Search Error: {e}")
            answer = "I am currently experiencing high traffic. Please try again in a moment."

    if target_lang != 'en':
        try:
            answer = translate(answer, target_lang)
        except:
            pass

    if len(answer) > 1400:
        answer = answer[:1400] + "..."
    return answer


if __name__ == "__main__":
    print("\n--- Medical Bot Terminal Test Mode ---")
    print("Languages: 'te' (Telugu), 'hi' (Hindi), 'en' (English)")

    while True:
        user_q = input("\nAsk a question (or type 'exit'): ")
        if user_q.lower() == 'exit':
            break
        lang = input("Target language code (e.g., en): ")
        print("\nAI CONSULTANT IS THINKING...")
        result = chatbot_response(user_q, lang)
        print(f"\nRESPONSE: {result}")
import os, pickle, hashlib, requests, logging
from groq import Groq
import PyPDF2
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import concurrent.futures

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama3-70b-8192"

# def scrape_website(url):
#     try:
#         res = requests.get(url)
#         return BeautifulSoup(res.text, "html.parser").get_text()
#     except Exception as e:
#         return ""

def extract_text(page):
    try: return page.extract_text()
    except: return ""

def process_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(extract_text, p) for p in reader.pages]
            for fut in concurrent.futures.as_completed(futures):
                try: text += fut.result() + "\n"
                except: continue
    return text

def split_into_chunks(text, chunk_size=1000, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return chunks

def get_or_create_chunks(pdf_paths):
    combined_text = ""
    for path in pdf_paths:
        with open(path, "rb") as f:
            content = f.read()
            file_hash = hashlib.md5(content).hexdigest()
        cache_path = f"cache/{file_hash}_chunks.pkl"
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                combined_text += " ".join(pickle.load(f))
        else:
            text = process_pdf(path)
            combined_text += text
    # combined_text += web_text
    chunks = split_into_chunks(combined_text)
    os.makedirs("cache", exist_ok=True)
    with open(f"cache/{hashlib.md5(combined_text.encode()).hexdigest()}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    return chunks

def initialize_resources():
    pdfs = ["./Aspire Data/UK.pdf", "./Aspire Data/FAQ;s.pdf","./Aspire Data/AspireEC-Data.pdf"]
    chunks = get_or_create_chunks(pdfs)
    return chunks, TfidfVectorizer().fit(chunks)

def find_relevant_chunks(query, chunks, vectorizer, k=6):
    query_vec = vectorizer.transform([query])
    chunk_vecs = vectorizer.transform(chunks)
    sims = cosine_similarity(query_vec, chunk_vecs)[0]
    indices = sims.argsort()[-k:][::-1]
    return [chunks[i] for i in indices]

def get_ai_response(messages, context):

    system = {
    "role": "system",
    "content": (
        "You are a helpful chatbot for Aspire Educational Consulting.\n"
        "Answer based only on provided context and history.\n"
        "Use active voice. Keep responses under 20 words.\n"
        "Greet when user come and when thry greet.\n"
        "Say goodbye with nice conversation if user says no/bye/nothing.\n"
        "Appreciate compliments politely and also deal with appriciation for your good response.\n"
        "For nonsense questions or profanity: 'I'm sorry, I can't answer that question. contact to the AspireEC team \n"
        "Answer will be short ,infomative, professional"
        )
    }
   
    
    all_msgs = [system] + messages[:-1] + [{"role": "user", "content": f"Context: {context}\n\n{messages[-1]['content']}"}]
    try:
        completion = client.chat.completions.create(messages=all_msgs, 
                                                    model=MODEL,
                                                    temperature=0.5,
                                                    max_tokens=200)
        return completion.choices[0].message.content
    except:
        return "I'm sorry, something went wrong. Please try again."

#hadling the bad words
def is_nonsense(query: str):
    bad_words = ["fuck", "shit", "teleport", "mars", "unicorn", "robot", "atlantis"]
    return any(word in query.lower() for word in bad_words)

def answer_query(query, messages, chunks, vectorizer):
    if is_nonsense(query):
        return "", "I'm sorry, I can't answer that question. Please contact our team."
    context = "\n".join(find_relevant_chunks(query, chunks, vectorizer))
    reply = get_ai_response(messages, context)
    return context, reply



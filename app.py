# ✅ Imports
import os
import re
import fitz  # PyMuPDF
from docx import Document
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from langdetect import detect
import streamlit as st
import openai
import cohere
import psycopg2
import psycopg2.extras
import json

# ✅ API Keys via st.secrets or env vars
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = st.secrets["COHERE_API_KEY"] if "COHERE_API_KEY" in st.secrets else os.getenv("COHERE_API_KEY")

# ✅ Database config
DB_HOST = st.secrets.get("DB_HOST", os.getenv("DB_HOST"))
DB_PORT = st.secrets.get("DB_PORT", os.getenv("DB_PORT", "8840"))
DB_USER = st.secrets.get("DB_USER", os.getenv("DB_USER"))
DB_PASSWORD = st.secrets.get("DB_PASSWORD", os.getenv("DB_PASSWORD"))
DB_NAME = st.secrets.get("DB_NAME", os.getenv("DB_NAME"))

# ✅ Initialize clients
client = openai.OpenAI(api_key=OPENAI_API_KEY)
cohere_client = cohere.Client(COHERE_API_KEY)

# ✅ DB connection
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    user=DB_USER,
    password=DB_PASSWORD,
    dbname=DB_NAME
)

# ✅ File extractor

def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        return "\n".join([p.text for p in Document(path).paragraphs])
    elif ext == ".pdf":
        try:
            with fitz.open(path) as doc:
                return "\n".join([page.get_text() for page in doc])
        except:
            images = convert_from_path(path)
            return "\n".join([pytesseract.image_to_string(img, lang='ara+eng') for img in images])
    elif ext in [".png", ".jpg", ".jpeg", ".tiff"]:
        return pytesseract.image_to_string(Image.open(path), lang='ara+eng')
    else:
        raise ValueError("Unsupported file type")

# ✅ Language detection

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# ✅ Document type

def detect_doc_type(text):
    prompt = f"""
You are an AI assistant. Identify the legal document type. Choose one:
- Contract
- Lease Agreement
- Employment Agreement
- Legal Memo
- Law or Regulation
- Court Decision
- Other

Reply with only the category name.

Document:
{text[:2000]}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a legal classifier."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# ✅ GPT chunking

def chunk_text_with_gpt(text, language, doc_type):
    if language.lower() == "arabic":
        prompt = f"""
أنت مساعد قانوني. قسم المستند التالي إلى أقسام قانونية واضحة ومترابطة.
احتفظ بالنص كما هو، وصحح فقط ما هو غير واضح دون تغيير المعنى.
ابدأ كل قسم بالتنسيق التالي:
--- Section [#] ---
Header: [عنوان أو "None"]
Content:
[النص الكامل بعد التصحيح]

النص:
{text}
"""
    else:
        prompt = f"""
You are a legal assistant. Split the document into logical legal sections.
Keep original language and wording, only clean up text format.
Use this format:
--- Section [#] ---
Header: [title or "None"]
Content:
[cleaned section text]

Document:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a multilingual legal document chunking assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

# ✅ Parse GPT sections

def extract_sections(gpt_output):
    pattern = r"--- Section \\d+ ---\\s*Header:\\s*(.*?)\\nContent:\\n(.*?)(?=\\n--- Section|\\Z)"
    matches = re.findall(pattern, gpt_output, re.DOTALL)
    return [{"header": h.strip(), "content": c.strip()} for h, c in matches]

# ✅ Embed with Cohere

def embed_with_cohere(texts):
    response = cohere_client.embed(
        texts=texts,
        model="embed-multilingual-v3.0",
        input_type="search_document",
        truncate="RIGHT"
    )
    embeddings = response.embeddings
    padded_embeddings = [e + [0.0] * (2000 - len(e)) for e in embeddings]
    return padded_embeddings

# ✅ Similarity search

def find_similar_chunks(query_embedding, top_k=5):
    vector_str = json.dumps(query_embedding)
    sql = """
    SELECT 
        c.id AS chunk_id,
        c.text AS chunk_text,
        1 - (e.vector <=> %s::vector) AS similarity
    FROM embedding e
    JOIN chunk c ON c.id = e.chunk_id
    WHERE 1 - (e.vector <=> %s::vector) > 0.5
    ORDER BY e.vector <=> %s::vector
    LIMIT %s;
    """
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    conn.rollback()
    cur.execute(sql, (vector_str, vector_str, vector_str, top_k))
    results = cur.fetchall()
    cur.close()
    return results

# ✅ Legal compliance check

def check_legal_compliance(query_section, retrieved_chunk):
    language = detect(query_section)
    if language == 'ar':
        prompt = f"""
أنت خبير قانوني. مطلوب منك تحديد ما إذا كان القسم المرجعي التالي متوافقًا مع المبادئ القانونية أو القوانين المعمول بها كما ورد في القسم المسترجع.

🔹 القسم المرجعي:
{query_section}

🔸 القسم المسترجع:
{retrieved_chunk}

هل النص المرجعي يتوافق قانونيًا مع ما ورد في القسم المسترجع؟
أجب بالتنسيق التالي فقط:

Verdict: ✅ متوافق / ❌ غير متوافق / ⚠️ يحتاج مراجعة
سبب:
- [سبب 1]
- [سبب 2]
        """
    else:
        prompt = f"""
You are a legal expert. Assess whether the reference section is legally compliant with the principles or requirements described in the retrieved section.

🔹 Reference Section:
{query_section}

🔸 Retrieved Section:
{retrieved_chunk}

Is the Reference Section legally compliant with the Retrieved Section?
Reply only in this format:

Verdict: ✅ Compliant / ❌ Non-compliant / ⚠️ Needs review
Reason:
- [Reason 1]
- [Reason 2]
        """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a legal compliance assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# ✅ Streamlit App

st.title("📑 Lebanese Legal Compliance Checker")
uploaded_file = st.file_uploader("Upload a legal document (PDF, DOCX, or image):")

if uploaded_file is not None:
    with st.spinner("🔍 Processing document..."):
        with open("temp_uploaded_file", "wb") as f:
            f.write(uploaded_file.read())

        raw_text = extract_text("temp_uploaded_file")
        lang = detect_language(raw_text)
        doc_type = detect_doc_type(raw_text)
        gpt_output = chunk_text_with_gpt(raw_text, lang, doc_type)
        sections = extract_sections(gpt_output)
        texts = [s["content"] for s in sections]
        embeddings = embed_with_cohere(texts)

        for i, emb in enumerate(embeddings):
            st.subheader(f"🔍 Section {i+1} - {sections[i]['header']}")
            similar_chunks = find_similar_chunks(emb, top_k=5)
            if not similar_chunks:
                st.warning("❌ No relevant chunks found in the database.")
            for j, chunk in enumerate(similar_chunks):
                st.markdown(f"**Match #{j+1} — Similarity: {round(chunk['similarity'], 4)}**")
                st.markdown("**Retrieved Section:**")
                st.code(chunk["chunk_text"][:1000])
                verdict = check_legal_compliance(sections[i]["content"], chunk["chunk_text"])
                st.markdown(f"**Result:**\n{verdict}")

        os.remove("temp_uploaded_file")


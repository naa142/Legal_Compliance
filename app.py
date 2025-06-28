import os
import re
import json
import fitz  # PyMuPDF
import streamlit as st
import psycopg2
import pytesseract
import cohere
import openai
from PIL import Image
from docx import Document
from langdetect import detect
from pdf2image import convert_from_path
import psycopg2.extras

# 📌 Load API keys and DB credentials securely from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]
cohere_client = cohere.Client(st.secrets["COHERE_API_KEY"])

conn = psycopg2.connect(
    host=st.secrets["DB_HOST"],
    port=st.secrets["DB_PORT"],
    user=st.secrets["DB_USER"],
    password=st.secrets["DB_PASSWORD"],
    dbname=st.secrets["DB_NAME"]
)

# ✅ File extractors
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
        return ""

# ✅ Detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# ✅ Detect doc type
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
    response = openai.ChatCompletion.create(
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
ابدأ كل قسم بالتنسيق التالي:
--- Section [#] ---
Header: [عنوان أو "None"]
Content:
[النص الكامل]

النص:
{text}
"""
    else:
        prompt = f"""
You are a legal assistant. Split the document into logical legal sections.
Use this format:
--- Section [#] ---
Header: [title or "None"]
Content:
[cleaned section text]

Document:
{text}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a legal chunking assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

# ✅ Extract GPT sections
def extract_sections(gpt_output):
    pattern = r"--- Section \d+ ---\s*Header:\s*(.*?)\nContent:\n(.*?)(?=\n--- Section|\Z)"
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
    padded = [e + [0.0] * (2000 - len(e)) for e in embeddings]
    return padded

# ✅ Search top similar chunks
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
    return cur.fetchall()

# ✅ Evaluate legal compliance
def check_legal_compliance(reference, retrieved):
    language = detect(reference)
    if language == 'ar':
        prompt = f"""
أنت مساعد قانوني. هل القسم التالي متوافق مع المبادئ القانونية الموضحة في القسم المسترجع؟

🔹 القسم المرجعي:
{reference}

🔸 القسم المسترجع:
{retrieved}

أجب بهذا الشكل فقط:

Verdict: ✅ متوافق / ❌ غير متوافق / ⚠️ يحتاج مراجعة
سبب:
- [سبب 1]
- [سبب 2]
"""
    else:
        prompt = f"""
You are a legal expert. Is the following section legally compliant with the retrieved legal reference?

🔹 Reference Section:
{reference}

🔸 Retrieved Section:
{retrieved}

Reply in this format only:

Verdict: ✅ Compliant / ❌ Not Compliant / ⚠️ Needs Review
Reason:
- [Reason 1]
- [Reason 2]
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a legal compliance assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# ✅ Streamlit UI
st.set_page_config(page_title="Legal Compliance Checker", layout="wide")
st.title("📜 Legal Compliance Checker")

uploaded_file = st.file_uploader("Upload a legal document (.pdf or .docx)", type=["pdf", "docx"])
top_k = st.slider("Number of similar legal chunks to retrieve", min_value=1, max_value=10, value=5)

if uploaded_file:
    with st.spinner("Extracting and analyzing document..."):
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())
        raw_text = extract_text(uploaded_file.name)
        lang = detect_language(raw_text)
        doc_type = detect_doc_type(raw_text)
        gpt_output = chunk_text_with_gpt(raw_text, lang, doc_type)
        sections = extract_sections(gpt_output)
        embeddings = embed_with_cohere([s["content"] for s in sections])

    for i, emb in enumerate(embeddings):
        st.subheader(f"📘 Section {i+1}")
        st.markdown(f"**Header**: {sections[i]['header']}")
        st.markdown(f"**Text:** {sections[i]['content'][:500]}...")  # Show preview

        similar_chunks = find_similar_chunks(emb, top_k=top_k)
        if not similar_chunks:
            st.warning("❌ No relevant legal references found in the database.")
        else:
            for j, chunk in enumerate(similar_chunks):
                st.markdown(f"---\n**Match #{j+1}** — Chunk ID: `{chunk['chunk_id']}` — Similarity: `{round(chunk['similarity'], 4)}`")
                verdict = check_legal_compliance(sections[i]["content"], chunk["chunk_text"])
                st.markdown(f"**Verdict:**\n```\n{verdict}\n```")

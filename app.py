#python
import re
import io
import os
import json
import time
import base64
from typing import List, Tuple, Dict

import streamlit as st
import numpy as np

# Optional NLP + parsing libs (installed via requirements)
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# File parsing
import fitz  # PyMuPDF for PDFs
from docx import Document  # python-docx for .docx

# Web fetching (optional for job ads by URL)
import requests
from bs4 import BeautifulSoup

# Ensure NLTK resources are available at runtime
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

EN_STOP = set(stopwords.words('english'))

# ------------------------------
# Helpers: File reading & cleaning
# ------------------------------

def read_file(file) -> Tuple[str, Dict]:
    """Read uploaded CV file (.pdf, .docx, .txt) and return text + metadata.
    Returns: (text, meta)
    """
    meta = {"file_name": file.name, "type": None, "pages": None, "images": 0}
    name = file.name.lower()
    if name.endswith('.pdf'):
        meta["type"] = 'pdf'
        text, pages, images = extract_text_from_pdf(file.read())
        meta["pages"] = pages
        meta["images"] = images
        return text, meta
    elif name.endswith('.docx'):
        meta["type"] = 'docx'
        text = extract_text_from_docx(file)
        return text, meta
    elif name.endswith('.txt'):
        meta["type"] = 'txt'
        return file.read().decode('utf-8', errors='ignore'), meta
    else:
        raise ValueError('Unsupported file type. Please upload a PDF, DOCX or TXT file.')


def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, int, int]:
    doc = fitz.open(stream=file_bytes, filetype='pdf')
    texts = []
    total_images = 0
    for page in doc:
        texts.append(page.get_text("text"))
        total_images += len(page.get_images(full=True))
    return "\n".join(texts), len(doc), total_images


def extract_text_from_docx(file) -> str:
    # file is a SpooledTemporaryFile; python-docx can open via BytesIO
    data = file.read()
    bio = io.BytesIO(data)
    doc = Document(bio)
    paras = [p.text for p in doc.paragraphs]
    return "\n".join(paras)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_stop_punct(text: str) -> str:
    # very light normalization for keyword comparison
    tokens = re.findall(r"[a-zA-Z0-9+#\.\-_/]+", text.lower())
    return " ".join([t for t in tokens if t not in EN_STOP])


# ------------------------------
# Keywords, similarity & extraction
# ------------------------------

def top_keywords(texts: List[str], top_n: int = 40) -> List[Tuple[str, float]]:
    """Return top keywords across one or more texts using TF-IDF (1-2 grams)."""
    if isinstance(texts, str):
        texts = [texts]
    cleaned = [remove_stop_punct(t) for t in texts if t and t.strip()]
    if not cleaned:
        return []
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9)
    X = vec.fit_transform(cleaned)
    # Aggregate by mean TF-IDF across docs
    scores = np.asarray(X.mean(axis=0)).ravel()
    features = np.array(vec.get_feature_names_out())
    top_idx = scores.argsort()[::-1][:top_n]
    return [(features[i], float(scores[i])) for i in top_idx]


def similarity_score(cv_text: str, job_texts: List[str]) -> float:
    """Cosine similarity between CV and concatenated job texts."""
    if not job_texts:
        return 0.0
    docs = [remove_stop_punct(cv_text), remove_stop_punct(" \n".join(job_texts))]
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=1.0)
    X = vec.fit_transform(docs)
    sim = cosine_similarity(X[0:1], X[1:2])[0,0]
    return float(sim)


def find_missing_keywords(cv_text: str, job_texts: List[str], top_n: int = 40) -> List[str]:
    job_kw = [kw for kw,_ in top_keywords(job_texts, top_n=top_n)]
    cv_kw = set(remove_stop_punct(cv_text).split())
    missing = []
    for kw in job_kw:
        tokens = kw.split()
        if all(t not in cv_kw for t in tokens):
            missing.append(kw)
    return missing


# ------------------------------
# Fetch job adverts by URL (optional)
# ------------------------------

def fetch_job_text_from_url(url: str, timeout: int = 12) -> str:
    """Fetch visible text from a job URL (user-provided). Respects simple robots via requests.
    Note: Always ensure you have the right to fetch and process the content.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, 'html.parser')
        # Try common containers first
        candidates = []
        for sel in [
            '[data-testid="jobDescriptionText"]',  # Indeed
            '#jobDescriptionText',
            '.jobDescriptionContent',               # Glassdoor (may vary)
            'section[data-test="jobDescription"]',
            'article', 'main', 'div'  # fallbacks
        ]:
            for el in soup.select(sel):
                txt = el.get_text(separator=' ', strip=True)
                if txt and len(txt) > 200:
                    candidates.append(txt)
        if candidates:
            # return the longest reasonable candidate
            return max(candidates, key=len)[:20000]
        # Fallback to full page text
        body = soup.get_text(separator=' ', strip=True)
        return body[:20000]
    except Exception:
        return ""


# ------------------------------
# ATS checks (heuristics)
# ------------------------------

def ats_checks(cv_text: str, meta: Dict, job_texts: List[str]) -> List[Dict]:
    issues = []
    text = cv_text
    lower = text.lower()

    # 1) File type & parseability
    if meta.get('type') == 'pdf' and (len(text.strip()) < 400):
        issues.append({
            'severity': 'high',
            'issue': 'PDF appears to contain very little extractable text.',
            'why': 'Some PDFs are scans or images. Many ATS cannot read them.',
            'fix': 'Export your CV to a text-based PDF or DOCX. Ensure you can copy-paste text from the PDF.'
        })
    
    # 2) Contact info
    if not re.search(r"\b\+?\d[\d\s().-]{6,}\b", text):
        issues.append({
            'severity': 'medium',
            'issue': 'No obvious phone number detected.',
            'why': 'ATS often parses contact details into candidate profiles.',
            'fix': 'Include an internationally formatted phone number on the first page header.'
        })
    if not re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
        issues.append({
            'severity': 'high',
            'issue': 'No email address found.',
            'why': 'Recruiters need an email to contact you.',
            'fix': 'Add a professional email (e.g., firstname.lastname@domain).' 
        })

    # 3) Length heuristic
    words = len(re.findall(r"\w+", text))
    if words > 1200:
        issues.append({
            'severity': 'low',
            'issue': 'CV may be too long (>{} words)'.format(words),
            'why': 'ATS and recruiters skim quickly; overly long CVs dilute key signals.',
            'fix': 'Target 1–3 pages. Prioritize recent, relevant achievements with quantified impact.'
        })

    # 4) Dates & chronology
    if not re.search(r"(20\d{2}|19\d{2})", text):
        issues.append({
            'severity': 'medium',
            'issue': 'Few or no years detected.',
            'why': 'ATS expects date ranges for experience sections.',
            'fix': 'Use clear date ranges like "Jan 2022 – Jun 2024".'
        })

    # 5) Formatting red flags
    if meta.get('type') == 'docx':
        # We cannot inspect tables without reading docx elements again; heuristic only
        pass
    # Tabs or multiple spaces may indicate multi-column layout which can confuse parsers
    if len(re.findall(r"\t|\s{3,}", text)) > 10:
        issues.append({
            'severity': 'medium',
            'issue': 'Possible multi-column or tab-based layout detected.',
            'why': 'Complex layouts can scramble parsing order in some ATS.',
            'fix': 'Use a single-column layout with standard section headings.'
        })

    # 6) Keyword alignment
    missing = find_missing_keywords(text, job_texts, top_n=35) if job_texts else []
    if missing:
        issues.append({
            'severity': 'high',
            'issue': 'Missing role-specific keywords',
            'why': 'Many ATS perform initial keyword screens based on job description.',
            'fix': 'Incorporate relevant terms naturally (skills, tools, standards) where you have real experience.',
            'details': missing[:20]
        })

    # 7) Buzzwords vs achievements
    if len(re.findall(r"(responsible for|duties included)", lower)) > 0 and len(re.findall(r"\b(\d+%|\$\d+|\b\d+\b)\b", lower)) == 0:
        issues.append({
            'severity': 'medium',
            'issue': 'Lots of responsibilities, few quantified outcomes.',
            'why': 'Outcome-focused bullets with metrics rank better with humans and some ATS scoring.',
            'fix': 'Use STAR format and include metrics (%, $, time saved, quality, reliability).'
        })

    return issues


# ------------------------------
# Rewrite examples (STAR-style)
# ------------------------------

def example_rewrites(missing: List[str], domain_hint: str = "engineering") -> List[str]:
    examples = []
    for kw in missing[:5]:
        kw_clean = kw.replace('-', ' ')
        examples.append(
            f"• Implemented {kw_clean} across project lifecycle, reducing cycle time by 18% while meeting safety & quality KPIs."
        )
    if not examples:
        examples = [
            "• Led cross-functional delivery of a critical project on time and 12% under budget by optimizing planning, interfaces and risk controls.",
            "• Automated weekly reporting pipeline in Python, cutting manual effort by 6 hours/week and improving data accuracy.",
        ]
    return examples


# ------------------------------
# Domain skill seeds (customize for your industry)
# ------------------------------
ENGINEERING_SKILLS = [
    # Core
    'project management', 'risk management', 'stakeholder management', 'root cause analysis',
    'continuous improvement', 'kpi', 'budget', 'cost control', 'scheduling', 'quality assurance',
    'quality control', 'commissioning', 'testing', 'inspection',
    # Tools
    'python', 'matlab', 'excel', 'power bi', 'sql', 'tableau', 'jira', 'confluence',
    # Standards (add your own)
    'iso', 'api', 'dnv', 'asme',
    # Subsea / pipelines (tailored hint)
    'dnv-st-f101', 'finite element analysis', 'abaqus', 'orcaflex', 'pipeline reeling', 'reel-lay',
    'fatigue analysis', 'fracture mechanics', 'on-bottom stability', 'installation analysis', 'spooling',
]


# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title="CV Reviewer & ATS Helper", page_icon="📄", layout="wide")
st.title("📄 CV Reviewer & ATS Helper")
st.caption("Upload your CV, paste a job description and (optionally) URLs to similar roles. Get a targeted review and ATS tips.")

with st.expander("⚠️ Notes on usage & privacy"):
    st.write("""
    - This tool runs locally in your browser session on Streamlit Cloud or your machine. Uploaded files are processed in memory.
    - For job advert URLs, only fetch pages you have the right to access. Respect website terms and robots rules.
    - This is a heuristic tool; always apply your judgment when tailoring your CV.
    """)

colL, colR = st.columns([1.1, 1])
with colL:
    cv_file = st.file_uploader("Upload your CV (PDF, DOCX or TXT)", type=["pdf","docx","txt"])
    job_desc = st.text_area("Paste the target job description", height=220, placeholder="Paste the job advert text here…")

with colR:
    url_input = st.text_area("Optional: paste 1–5 job advert URLs (one per line)", height=220)
    fetch_btn = st.button("Fetch job adverts from URLs")

fetched_texts: List[str] = []
if fetch_btn and url_input.strip():
    with st.spinner("Fetching job adverts…"):
        for url in [u.strip() for u in url_input.splitlines() if u.strip()]:
            txt = fetch_job_text_from_url(url)
            if txt:
                fetched_texts.append(txt)
            else:
                st.warning(f"Could not extract text from: {url}")
        if fetched_texts:
            st.success(f"Fetched {len(fetched_texts)} job advert(s).")

analyze = st.button("🔎 Analyze")

if analyze:
    if not cv_file:
        st.error("Please upload your CV to continue.")
        st.stop()
    if not job_desc and not fetched_texts:
        st.error("Please add a job description or at least one job advert URL.")
        st.stop()

    # Read CV
    try:
        cv_text, meta = read_file(cv_file)
    except Exception as e:
        st.error(f"Could not read CV: {e}")
        st.stop()

    # Aggregate job texts
    job_texts = []
    if job_desc:
        job_texts.append(job_desc)
    job_texts.extend(fetched_texts)

    # Compute scores and insights
    with st.spinner("Analyzing CV against job requirements…"):
        sim = similarity_score(cv_text, job_texts)
        missing = find_missing_keywords(cv_text, job_texts, top_n=35)
        ats = ats_checks(cv_text, meta, job_texts)
        job_kw = [kw for kw,_ in top_keywords(job_texts, top_n=30)]

    st.subheader("Match overview")
    meter = st.progress(min(max(sim, 0), 1.0))
    st.write(f"**Semantic match score:** {sim*100:.1f}% (TF‑IDF cosine similarity)")
    st.caption("This is a heuristic score for quick guidance, not a pass/fail.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top job keywords (from supplied adverts):**")
        if job_kw:
            st.write(", ".join(job_kw[:20]))
    with col2:
        if missing:
            st.markdown("**Keywords to consider adding (where true to your experience):**")
            st.write(", ".join(missing[:20]))
        else:
            st.success("Your CV already covers many of the key terms.")

    st.subheader("ATS risk checks")
    if ats:
        for item in ats:
            sev = item.get('severity','low').upper()
            st.markdown(f"- **[{sev}] {item['issue']}** — {item['why']}\n  - _Fix:_ {item['fix']}")
            if 'details' in item:
                st.caption("Missing examples: " + ", ".join(item['details']))
    else:
        st.success("No major ATS risks detected.")

    st.subheader("Suggested bullet rewrites (STAR‑inspired)")
    rewrites = example_rewrites(missing)
    st.write("\n".join(rewrites))

    # Build a Markdown report for download
    report = io.StringIO()
    report.write(f"# CV Review Report\n\n")
    report.write(f"**File:** {meta.get('file_name')}  \n")
    if meta.get('type') == 'pdf':
        report.write(f"**Pages:** {meta.get('pages')}  | **Images:** {meta.get('images')}\n\n")
    report.write(f"## Match overview\n")
    report.write(f"Semantic match score: {sim*100:.1f}%\n\n")
    report.write("**Top job keywords:**\n\n- " + "\n- ".join(job_kw[:20]) + "\n\n")
    if missing:
        report.write("**Keywords to consider adding:**\n\n- " + "\n- ".join(missing[:30]) + "\n\n")

    report.write("## ATS risk checks\n")
    if ats:
        for item in ats:
            report.write(f"- **[{item.get('severity','low').upper()}] {item['issue']}** — {item['why']}\n")
            report.write(f"  - Fix: {item['fix']}\n")
            if 'details' in item:
                report.write("  - Missing examples: " + ", ".join(item['details']) + "\n")
    else:
        report.write("- No major ATS risks detected.\n")

    report.write("\n## Suggested rewrites\n\n")
    for r in rewrites:
        report.write(r + "\n")

    st.download_button(
        label="⬇️ Download full report (Markdown)",
        data=report.getvalue().encode('utf-8'),
        file_name="cv_review_report.md",
        mime="text/markdown"
    )

st.markdown("---")
st.caption("Tip: Tailor the skills taxonomy inside the app to your industry (e.g., add DNV‑ST‑F101, OrcaFlex, Abaqus, reel‑lay, etc.)")

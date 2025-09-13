# 📄 CV Reviewer & ATS Helper (Streamlit)

A lightweight Streamlit app that lets candidates:

- **Review an uploaded CV** against a pasted job description and **(optionally)** similar job adverts by URL (e.g., Indeed, Glassdoor).
- Get a **comprehensive improvement report** based on missing keywords, semantic similarity and writing guidance.
- See **ATS risk flags** (parseability, contact info, layout, keyword alignment) and practical fixes.

> **Privacy:** Files are processed in memory during your session. No data is stored server-side by this app.

## 🚀 Demo locally

```bash
# 1) Clone this repo
git clone https://github.com/your-username/cv-reviewer-streamlit.git
cd cv-reviewer-streamlit

# 2) Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## ☁️ Deploy on Streamlit Community Cloud

1. Push this project to a **public GitHub repo**.
2. Go to [share.streamlit.io](https://share.streamlit.io/) (Streamlit Community Cloud) and **Deploy from GitHub**.
3. Set the app entry point to `app.py` and click deploy.

> Optional: If you later add any API keys (e.g., to integrate a jobs API), store them in **Streamlit Secrets** (Settings → Secrets) and access via `st.secrets`.

## 🔧 Features & approach

- **File parsing**: PDF via PyMuPDF, DOCX via python-docx, TXT.
- **Keyword extraction**: TF–IDF (1–2 grams) to surface salient terms from job adverts.
- **Semantic match score**: Cosine similarity between CV and combined job descriptions.
- **ATS heuristics**: Flags for parseability (e.g., scanned PDFs), missing contact info, excessive length, multi‑column hints, missing role keywords, outcome vs duty phrasing.
- **Rewrite suggestions**: STAR‑style bullet prompts using missing keywords where appropriate.
- **Jobs by URL**: Paste 1–5 job advert URLs; the app will fetch and extract the description text for you.

> **Respect site terms:** Only fetch content you are allowed to access. Many job boards prohibit scraping. Prefer copy‑pasting job text or using official APIs where available.

## 🔌 Optional: Integrate a jobs API

Instead of pasting URLs, you can integrate a jobs API (e.g., Adzuna, Jooble, or RapidAPI's JSearch). Add your key in Streamlit Secrets and modify the code to fetch postings programmatically. Always review the API's terms of use and attribution requirements.

## 🧩 Customize for your domain

Open `app.py` and expand the `ENGINEERING_SKILLS` list to include domain‑specific terms. For example, for subsea pipeline & reel‑lay roles:

- DNV‑ST‑F101, API 1111, ASME B31, FEA, fatigue analysis
- OrcaFlex, Abaqus, MATLAB, Python
- Reel‑lay, spooling, installation analysis, on‑bottom stability

This boosts keyword coverage and guidance relevance.

## 📁 Project structure

```
cv-reviewer-streamlit/
├─ app.py                 # Streamlit app (single‑file)
├─ requirements.txt       # Python deps for Streamlit Cloud
├─ .streamlit/
│  └─ secrets.toml.example
└─ README.md
```

## 🛡️ License

[MIT](LICENSE)
# 📄 AI Resume ATS Checker

Upload a resume (PDF, DOCX or TXT) and get:

- An **ATS score** out of 100 with a category breakdown
- Strengths, weaknesses and **missing keywords**
- Prioritised **improvements** and example line rewrites
- A quick rule-based ATS checklist (contact info, sections, numbers, length)
- Optional **job description** matching for better keyword analysis

Built with [Streamlit](https://streamlit.io) and the [Groq](https://groq.com) API.

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set your Groq key (get one at https://console.groq.com):

```bash
export GROQ_API_KEY="your_key_here"      # Windows PowerShell: $env:GROQ_API_KEY="your_key_here"
streamlit run app.py
```

Or skip this and paste the key into the app's sidebar.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (`app.py`, `requirements.txt`, `README.md`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
5. Click **Deploy**.

Never commit your API key to GitHub.

## Notes

- Scanned/image-only PDFs have no extractable text; use a text-based PDF or DOCX.
- The score is an AI estimate, not the output of a real ATS. Use it as guidance.
- Resumes are sent to Groq for analysis.

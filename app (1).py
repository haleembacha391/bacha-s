"""AI Resume ATS Checker - Streamlit + Groq."""

import json
import os
import re
from io import BytesIO

import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader

MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
MAX_CHARS = 12000  # keep prompts well within the model context window

SYSTEM_PROMPT = (
    "You are an expert ATS (Applicant Tracking System) analyst and professional "
    "resume reviewer. You are strict, specific and honest. You respond ONLY with "
    "a single valid JSON object and nothing else."
)

USER_PROMPT = """Analyze the resume below{jd_clause}.

Return JSON with EXACTLY this structure:
{{
  "overall_score": <integer 0-100>,
  "category_scores": {{
    "formatting": <integer 0-100>,
    "keywords": <integer 0-100>,
    "impact_and_achievements": <integer 0-100>,
    "structure_and_sections": <integer 0-100>,
    "readability_and_grammar": <integer 0-100>
  }},
  "summary": "<2-3 sentence overall assessment>",
  "strengths": ["<specific strength>", "..."],
  "weaknesses": ["<specific weakness>", "..."],
  "missing_keywords": ["<keyword or skill>", "..."],
  "improvements": [
    {{"priority": "High|Medium|Low", "issue": "<what is wrong>", "fix": "<exact, actionable fix>"}}
  ],
  "rewrite_examples": [
    {{"before": "<weak line copied from the resume>", "after": "<stronger rewritten line>"}}
  ]
}}

Rules:
- Base every point on the actual resume text. Do not invent facts, jobs or numbers.
- "rewrite_examples": at most 3. In "after", use placeholders like [X%] where a metric is unknown.
- {jd_rule}
- Give 5-8 items in "improvements", sorted by priority.

RESUME:
\"\"\"
{resume}
\"\"\"
{jd_block}"""


# --------------------------------------------------------------------------- #
# File reading
# --------------------------------------------------------------------------- #
def extract_text(uploaded_file) -> str:
    """Extract plain text from a PDF, DOCX or TXT upload."""
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ValueError("This PDF is password-protected.")
        pages = [(p.extract_text() or "") for p in reader.pages]
        return "\n".join(pages).strip()

    if name.endswith(".docx"):
        doc = Document(BytesIO(data))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        parts.append(cell.text.strip())
        return "\n".join(parts).strip()

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore").strip()

    raise ValueError("Unsupported file type. Please upload a PDF, DOCX or TXT file.")


# --------------------------------------------------------------------------- #
# Rule-based checks (no AI needed)
# --------------------------------------------------------------------------- #
def basic_checks(text: str) -> list:
    """Return a list of (label, passed, detail) tuples."""
    lower = text.lower()
    words = len(text.split())
    checks = [
        (
            "Email address",
            bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)),
            "ATS and recruiters need a contact email.",
        ),
        (
            "Phone number",
            bool(re.search(r"(\+?\d[\d\s().-]{8,}\d)", text)),
            "Include a reachable phone number.",
        ),
        (
            "LinkedIn / portfolio link",
            bool(re.search(r"linkedin\.com|github\.com|portfolio|behance\.net", lower)),
            "Add a LinkedIn or portfolio URL.",
        ),
        (
            "Experience section",
            bool(re.search(r"experience|employment|work history|internship", lower)),
            "Use a standard heading such as 'Experience'.",
        ),
        (
            "Education section",
            bool(re.search(r"education|academic|university|college|degree", lower)),
            "Use a standard heading such as 'Education'.",
        ),
        (
            "Skills section",
            bool(re.search(r"skills|technologies|technical proficiency|competencies", lower)),
            "Use a clear 'Skills' heading so ATS can parse it.",
        ),
        (
            "Measurable results (numbers / %)",
            len(re.findall(r"\d+\s?%|\$\s?\d|\b\d{2,}\b", text)) >= 3,
            "Quantify achievements with numbers, percentages or amounts.",
        ),
        (
            "Length (250-1000 words)",
            250 <= words <= 1000,
            f"Your resume has about {words} words.",
        ),
    ]
    return checks


# --------------------------------------------------------------------------- #
# AI analysis
# --------------------------------------------------------------------------- #
def _clamp(value, default=0) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def _as_list(value) -> list:
    return value if isinstance(value, list) else []


def parse_json(raw: str) -> dict:
    """Parse JSON from a model reply, tolerating code fences and extra text."""
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def normalize(result: dict) -> dict:
    """Make sure every field exists with a safe type and range."""
    cats = result.get("category_scores") or {}
    keys = [
        "formatting",
        "keywords",
        "impact_and_achievements",
        "structure_and_sections",
        "readability_and_grammar",
    ]
    cat_scores = {k: _clamp(cats.get(k)) for k in keys}
    overall = result.get("overall_score")
    if overall is None and cat_scores:
        overall = sum(cat_scores.values()) / len(cat_scores)

    improvements = []
    for item in _as_list(result.get("improvements")):
        if isinstance(item, dict):
            improvements.append(
                {
                    "priority": str(item.get("priority", "Medium")).title(),
                    "issue": str(item.get("issue", "")),
                    "fix": str(item.get("fix", "")),
                }
            )
        elif isinstance(item, str):
            improvements.append({"priority": "Medium", "issue": item, "fix": ""})

    rewrites = [
        {"before": str(r.get("before", "")), "after": str(r.get("after", ""))}
        for r in _as_list(result.get("rewrite_examples"))
        if isinstance(r, dict)
    ]

    return {
        "overall_score": _clamp(overall),
        "category_scores": cat_scores,
        "summary": str(result.get("summary", "")),
        "strengths": [str(s) for s in _as_list(result.get("strengths"))],
        "weaknesses": [str(s) for s in _as_list(result.get("weaknesses"))],
        "missing_keywords": [str(s) for s in _as_list(result.get("missing_keywords"))],
        "improvements": improvements,
        "rewrite_examples": rewrites,
    }


def analyze_resume(client, model: str, resume_text: str, job_description: str = "") -> dict:
    jd = job_description.strip()
    prompt = USER_PROMPT.format(
        jd_clause=" against the job description provided" if jd else "",
        jd_rule=(
            "Use the job description to judge keyword match; 'missing_keywords' must come from it."
            if jd
            else "No job description was given, so judge keywords against common industry standards for the candidate's field."
        ),
        resume=resume_text[:MAX_CHARS],
        jd_block=f'\nJOB DESCRIPTION:\n"""\n{jd[:4000]}\n"""\n' if jd else "",
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=2500,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        # Some models reject response_format; retry once without it.
        if "response_format" in str(exc).lower() or "json" in str(exc).lower():
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0.2, max_tokens=2500
            )
        else:
            raise
    return normalize(parse_json(resp.choices[0].message.content))


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def get_api_key() -> str:
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:  # no secrets file present
        key = ""
    return key or os.environ.get("GROQ_API_KEY", "")


def score_label(score: int) -> str:
    if score >= 80:
        return "🟢 Excellent"
    if score >= 60:
        return "🟡 Good, needs work"
    return "🔴 Needs major improvement"


def render_results(result: dict, checks: list):
    st.divider()
    score = result["overall_score"]
    left, right = st.columns([1, 2])
    with left:
        st.metric("ATS Score", f"{score} / 100")
        st.caption(score_label(score))
        st.progress(score / 100)
    with right:
        st.subheader("Summary")
        st.write(result["summary"] or "No summary returned.")

    st.subheader("Score breakdown")
    names = {
        "formatting": "Formatting",
        "keywords": "Keywords",
        "impact_and_achievements": "Impact & achievements",
        "structure_and_sections": "Structure & sections",
        "readability_and_grammar": "Readability & grammar",
    }
    cols = st.columns(len(names))
    for col, (key, label) in zip(cols, names.items()):
        val = result["category_scores"][key]
        col.metric(label, val)
        col.progress(val / 100)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("✅ Strengths")
        for s in result["strengths"] or ["None identified."]:
            st.markdown(f"- {s}")
    with c2:
        st.subheader("⚠️ Weaknesses")
        for s in result["weaknesses"] or ["None identified."]:
            st.markdown(f"- {s}")

    if result["missing_keywords"]:
        st.subheader("🔑 Missing keywords")
        st.write(", ".join(f"`{k}`" for k in result["missing_keywords"]))

    st.subheader("🛠️ Recommended improvements")
    icons = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    order = {"High": 0, "Medium": 1, "Low": 2}
    for item in sorted(result["improvements"], key=lambda i: order.get(i["priority"], 1)):
        with st.expander(f"{icons.get(item['priority'], '🟡')} {item['priority']}: {item['issue'][:90]}"):
            st.markdown(f"**Issue:** {item['issue']}")
            st.markdown(f"**Fix:** {item['fix']}")

    if result["rewrite_examples"]:
        st.subheader("✍️ Example rewrites")
        for ex in result["rewrite_examples"]:
            st.markdown(f"**Before:** {ex['before']}")
            st.markdown(f"**After:** {ex['after']}")
            st.write("")

    st.subheader("📋 Quick ATS checklist")
    for label, ok, detail in checks:
        st.markdown(f"{'✅' if ok else '❌'} **{label}**" + ("" if ok else f" - {detail}"))

    st.download_button(
        "⬇️ Download report (JSON)",
        data=json.dumps(result, indent=2),
        file_name="ats_report.json",
        mime="application/json",
    )


# --------------------------------------------------------------------------- #
# Main app
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="AI Resume ATS Checker", page_icon="📄", layout="wide")
    st.title("📄 AI Resume ATS Checker")
    st.write("Upload your resume to get an ATS score and specific suggestions to improve it.")

    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = get_api_key()
        if api_key:
            st.success("Groq API key loaded.")
        else:
            api_key = st.text_input("Groq API key", type="password", help="Get one free at console.groq.com")
        model = st.selectbox("Model", MODELS)
        st.caption("Your resume is sent to Groq for analysis. Don't upload anything you can't share.")

    uploaded = st.file_uploader("Upload resume (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"])
    job_description = st.text_area(
        "Job description (optional, improves keyword matching)", height=150
    )

    if not st.button("Analyze resume", type="primary"):
        return
    if uploaded is None:
        st.warning("Please upload a resume first.")
        return
    if not api_key:
        st.error("Please provide your Groq API key in the sidebar.")
        return

    try:
        text = extract_text(uploaded)
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        return
    if len(text) < 100:
        st.error(
            "Almost no text could be extracted. If your resume is a scanned image, "
            "ATS systems can't read it either - export a text-based PDF or DOCX instead."
        )
        return

    with st.spinner("Analyzing your resume..."):
        try:
            result = analyze_resume(Groq(api_key=api_key), model, text, job_description)
        except json.JSONDecodeError:
            st.error("The AI returned an unreadable response. Please try again.")
            return
        except Exception as exc:
            st.error(f"Groq API error: {exc}")
            return

    render_results(result, basic_checks(text))


if __name__ == "__main__":
    main()

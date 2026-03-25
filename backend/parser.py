import io
import json
import re
import PyPDF2
from groq import Groq
import os


client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def _parse_json_response(raw: str) -> dict:
    raw = re.sub(r'```(?:json)?\s*|\s*```', '', raw).strip()
    start, end = raw.index('{'), raw.rindex('}')
    return json.loads(raw[start:end + 1])


def parse_resume(file_bytes: bytes):
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        prompt = f"""
            Extract the following from this resume and respond in JSON only:
            {{
            "skills": ["list of technical skills"],
            "total_years_experience": <number, calculated from years in the most 
                relevant role. (For example, if the sample resumé has 3 years as
                a 'software engineer' or anything related to the term and 1 year
                as a 'programming teacher' only consider the years for most relevant role
                like 'software engineer' terms)>,
            "seniority_level": "entry" | "mid" | "senior" | "lead"
            }}

            Rules for seniority_level:
            - entry: 0-2 years total
            - mid: 2-5 years total
            - senior: 5-8 years total
            - lead: 8+ years total

            Resume:
            {text}
        """
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        parsed = _parse_json_response(response.choices[0].message.content)
        skills = parsed.get("skills", [])
        years = float(parsed.get("total_years_experience") or 0)
        seniority = parsed.get("seniority_level", "entry")
        print("Candidate's resume info: ", parsed, skills, years, seniority)
        return text, skills, years, seniority
    except Exception as e:
        raise ValueError(f"Parsing error - Resume parsing failed: {str(e)}")


def parse_job_requirements(job_description: str) -> dict:
    text = job_description.lower()

    # --- min_years_required ---
    # Only matches work/professional experience — not education years (e.g. "4 years of study")
    work_years_patterns = [
        r'(\d+)\s*\+\s*years?\s+of\s+(?:professional|work|industry|software|hands[- ]on|relevant|commercial)\s+experience',
        r'(\d+)\s*or more years?\s+of\s+(?:professional|work|industry|software|hands[- ]on|relevant|commercial)\s+experience',
        r'at least\s+(\d+)\s+years?\s+of\s+(?:professional|work|industry|software|hands[- ]on|relevant|commercial)\s+experience',
        r'minimum\s+(?:of\s+)?(\d+)\s+years?\s+of\s+(?:professional|work|industry|software|hands[- ]on|relevant|commercial)\s+experience',
        r'(\d+)\s*[-–]\s*\d+\s+years?\s+of\s+(?:professional|work|industry|software|hands[- ]on|relevant|commercial)\s+experience',
        r'(\d+)\s*\+\s*years?\s+(?:professional|work|industry)\s+experience',
        r'(\d+)\s+years?\s+of\s+(?:proven|demonstrated|relevant)\s+(?:work\s+)?experience',
    ]
    matches = []
    for pattern in work_years_patterns:
        for match in re.finditer(pattern, text):
            value = float(match.group(1))
            if value <= 30:  # sanity check — discard implausible values
                matches.append(value)
    min_years = min(matches) if matches else None

    # --- seniority_level ---
    if re.search(r'\b(lead|principal|staff engineer|head of)\b', text):
        seniority = "lead"
    elif re.search(r'\b(senior|sr\.?)\b', text):
        seniority = "senior"
    elif re.search(r'\b(mid[- ]?level|intermediate|associate)\b', text):
        seniority = "mid"
    elif re.search(r'\b(junior|jr\.?|entry[- ]?level|graduate|new grad)\b', text):
        seniority = "entry"
    else:
        seniority = "any"

    return {"min_years_required": min_years, "seniority_level": seniority}

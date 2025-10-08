import io
import PyPDF2
from openai import OpenAI


client = OpenAI()


def parse_resume(file_bytes: bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    prompt = f"Extract key skills as a list from this resume text: {text[:4000]}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    skills_text = response.choices[0].message.content
    skills = [s.strip() for s in skills_text.split(',') if s.strip()]
    return text, skills
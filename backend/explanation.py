from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def explain_match(resume_text, job_title, job_desc, score):
    try:
        prompt = f"Candidate: {resume_text[:1000]}\nJob: {job_title}\nDescription: {job_desc}\nFit score: {score}.\nExplain why this is a good fit and how to improve callback chances."
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        raise ValueError(f"Explanation error - Explain match failed: {str(e)}")
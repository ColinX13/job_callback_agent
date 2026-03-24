from groq import Groq
import os
import json

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def explain_match(resume_text, job_title, job_desc, score):
    system_prompt = """You are an expert career coach and hiring advisor.
    Analyze the fit between the candiadate's resume and the job posting.
    Respond with valid JSON only - no prose, no markdown, and no explanation aside from the JSON.
    """
    user_prompt = f"""
    Resume: 
    {resume_text}

    Job Title: {job_title}
    Job Description: {job_desc}
    Match Score: {round(score * 100, 1)}%

    Respond with this JSON structure:
    {{
    "summary": "2-3 sentence explanation of the match",
    "strengths": ["specific strength 1", "specific strength 2", "specific strength 3"],
    "gaps": [
        {{
        "skill": "skill or requirement name",
        "required": "what the job needs",
        "user_has": "what the candidate has",
        "closeable": true
        }}
    ],
    "quick_wins": ["specific action to improve callback chances", "another action"]
    }}
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        raise ValueError(f"Explanation error - Explain match failed: {str(e)}")
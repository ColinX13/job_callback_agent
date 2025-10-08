from openai import OpenAI
client = OpenAI()


def explain_match(resume_text, job_title, job_desc, score):
    prompt = f"Candidate: {resume_text[:1000]}\nJob: {job_title}\nDescription: {job_desc}\nFit score: {score}.\nExplain why this is a good fit and how to improve callback chances."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content
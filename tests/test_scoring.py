import pytest
from unittest.mock import patch, MagicMock
from backend.db import SessionLocal
from backend.scoring import rank_jobs, skill_overlap, cosine_sim
from backend.models import Jobs

@pytest.fixture()
def db_session():
    db = SessionLocal()
    yield db
    db.close()

def test_rank_jobs_with_string_embeddings():
    # Test parsing string embeddings (valid JSON)
    mock_db = MagicMock()
    mock_job = MagicMock()
    mock_job.embedding = '[0.1, 0.2, 0.3]'  # String embedding
    mock_job.skills = ['python']  # Already list
    mock_job.id = 1
    mock_job.title = 'Test Job'
    mock_job.company = 'Test Co'
    mock_job.description = 'Desc'
    mock_db.query.return_value.all.return_value = [mock_job]
    
    with patch('backend.scoring.cosine_sim', return_value=0.8), \
         patch('backend.scoring.skill_overlap', return_value=0.5):
        result = rank_jobs(mock_db, 'text', [0.1, 0.2, 0.3], ['python'])
        
        assert len(result) == 1
        assert result[0]['score'] == 0.71  # 0.7*0.8 + 0.3*0.5


def test_rank_jobs_with_invalid_string_embeddings():
    # Test fallback for invalid string embeddings
    mock_db = MagicMock()
    mock_job = MagicMock()
    mock_job.embedding = 'invalid json'  # Invalid string
    mock_job.skills = ['python']
    mock_job.id = 1
    mock_job.title = 'Test Job'
    mock_job.company = 'Test Co'
    mock_job.description = 'Desc'
    mock_db.query.return_value.all.return_value = [mock_job]
    
    with patch('backend.scoring.cosine_sim', return_value=0.0) as mock_cos:  # Should be called with [] and resume_emb
        result = rank_jobs(mock_db, 'text', [0.1, 0.2, 0.3], ['python'])
        
        mock_cos.assert_called_with([0.1, 0.2, 0.3], [])  # job_emb falls back to []


def test_rank_jobs_with_string_skills():
    # Test parsing string skills (valid JSON)
    mock_db = MagicMock()
    mock_job = MagicMock()
    mock_job.embedding = [0.1, 0.2, 0.3]
    mock_job.skills = '["python", "sql"]'  # String skills
    mock_job.id = 1
    mock_job.title = 'Test Job'
    mock_job.company = 'Test Co'
    mock_job.description = 'Desc'
    mock_db.query.return_value.all.return_value = [mock_job]
    
    with patch('backend.scoring.cosine_sim', return_value=0.8), \
         patch('backend.scoring.skill_overlap', return_value=1.0) as mock_overlap:
        result = rank_jobs(mock_db, 'text', [0.1, 0.2, 0.3], ['python'])
        
        mock_overlap.assert_called_with(['python'], ['python', 'sql'])


def test_rank_jobs_with_invalid_string_skills():
    # Test fallback for invalid string skills
    mock_db = MagicMock()
    mock_job = MagicMock()
    mock_job.embedding = [0.1, 0.2, 0.3]
    mock_job.skills = 'invalid json'  # Invalid string
    mock_job.id = 1
    mock_job.title = 'Test Job'
    mock_job.company = 'Test Co'
    mock_job.description = 'Desc'
    mock_db.query.return_value.all.return_value = [mock_job]
    
    with patch('backend.scoring.cosine_sim', return_value=0.8), \
         patch('backend.scoring.skill_overlap', return_value=0.0) as mock_overlap:
        result = rank_jobs(mock_db, 'text', [0.1, 0.2, 0.3], ['python'])
        
        mock_overlap.assert_called_with(['python'], [])  # job_skills falls back to []

def test_cosine_sim():
    a = [1, 0]
    b = [0, 1]
    assert cosine_sim(a, b) == 0.0

def test_skill_overlap():
    assert skill_overlap(["Python"], ["Python", "js"]) == 0.5

def test_rank_jobs(db_session):
    # Clear existing jobs
    db_session.query(Jobs).delete()
    db_session.commit()

    # Add test jobs to the database
    job1 = Jobs(title="Test Job 1", company="Test Company", description="Test Description", skills=["Python"], embedding=[0.1, 0.2, 0.3])
    job2 = Jobs(title="Test Job 2", company="Test Company", description="Test Description", skills=["Java"], embedding=[0.4, 0.5, 0.6])
    db_session.add(job1)
    db_session.add(job2)
    db_session.commit()
    
    resume_text = "Test Text"
    resume_emb = [0.1, 0.2, 0.3]
    resume_skills = ["Python"]
    ranked = rank_jobs(db_session, resume_text, resume_emb, resume_skills)
    assert len(ranked) == 2
    assert ranked[0]["title"] == "Test Job 1"
    assert ranked[1]["title"] == "Test Job 2"

def test_rank_jobs_exception():
    mock_db = MagicMock()
    mock_job = MagicMock()
    mock_job.embedding = [0.1, 0.2, 0.3]
    mock_job.skills = ['python']
    mock_db.query.return_value.all.return_value = [mock_job]
    
    with patch('backend.scoring.cosine_sim', side_effect=Exception("Sim error")):
        with pytest.raises(ValueError, match="Scoring error - Rank jobs failed: Sim error"):
            rank_jobs(mock_db, 'text', [0.1, 0.2, 0.3], ['python'])
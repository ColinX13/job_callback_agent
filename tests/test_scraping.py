import pytest
import requests
from unittest.mock import patch, MagicMock
from backend.ingestion.scraping import fetch_jobs, normalize_job, ingest_jobs
from backend.models import Jobs



def test_fetch_jobs_success():
    mock_response = MagicMock()
    mock_response.json.return_value = {"jobs": [{"title": "Test Job"}]}
    
    with patch('backend.ingestion.scraping.requests.get', return_value=mock_response) as mock_get:
        jobs = fetch_jobs()
        assert jobs == [{"title": "Test Job"}]
        mock_get.assert_called_once_with("https://remotive.com/api/remote-jobs?category=software-dev", timeout=10)


def test_fetch_jobs_api_error():
    with patch('backend.ingestion.scraping.requests.get', side_effect=requests.RequestException("Network error")):
        with pytest.raises(ValueError, match="Scraping Error - Error fetching jobs from API: Network error"):
            fetch_jobs()


def test_fetch_jobs_invalid_response():
    mock_response = MagicMock()
    mock_response.json.return_value = {"invalid": "data"}  # Missing "jobs" key
    
    with patch('backend.ingestion.scraping.requests.get', return_value=mock_response):
        with pytest.raises(ValueError, match="Scraping Error - Unexpected API response structure"):
            fetch_jobs()


def test_normalize_job():
    job_data = {
        "title": "Software Engineer",
        "company_name": "Test Co",
        "description": "Build apps",
        "candidate_required_location": "Worldwide",
        "tags": ["Python", "SQL"],
        "job_type": "Full-time"
    }
    
    with patch('backend.ingestion.scraping.embed_text', return_value=[0.1] * 384) as mock_embed:
        normalized = normalize_job(job_data)
        
        assert normalized["title"] == "Software Engineer"
        assert normalized["company"] == "Test Co"
        assert normalized["description"] == "Build apps"
        assert normalized["remote"] is True
        assert normalized["skills"] == ["Python", "SQL"]
        assert normalized["embedding"] == [0.1] * 384
        mock_embed.assert_called_once_with("Software Engineer Build apps Full-time")


def test_ingest_jobs_success():
    mock_jobs = [
        {"title": "Job 1", "company_name": "Co 1", "description": "Desc 1", "candidate_required_location": "Worldwide", "tags": ["Skill1"]},
        {"title": "Job 2", "company_name": "Co 2", "description": "Desc 2", "candidate_required_location": "Local", "tags": ["Skill2"]}
    ]
    
    with patch('backend.ingestion.scraping.fetch_jobs', return_value=mock_jobs), \
         patch('backend.ingestion.scraping.normalize_job') as mock_normalize, \
         patch('backend.ingestion.scraping.SessionLocal') as mock_session_class, \
         patch('backend.ingestion.scraping.embed_text', return_value=[0.1] * 384):
        
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None  # No duplicates
        
        # Mock normalize_job returns
        mock_normalize.side_effect = [
            {"title": "Job 1", "company": "Co 1", "description": "Desc 1", "remote": True, "skills": ["Skill1"], "embedding": [0.1]*384},
            {"title": "Job 2", "company": "Co 2", "description": "Desc 2", "remote": False, "skills": ["Skill2"], "embedding": [0.1]*384}
        ]
        
        ingest_jobs()
        
        assert mock_session.add.call_count == 2
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()


def test_ingest_jobs_with_duplicates():
    mock_jobs = [{"title": "Job 1", "company_name": "Co 1", "description": "Desc 1", "candidate_required_location": "Worldwide", "tags": []}]
    
    with patch('backend.ingestion.scraping.fetch_jobs', return_value=mock_jobs), \
         patch('backend.ingestion.scraping.normalize_job', return_value={"title": "Job 1", "company": "Co 1", "description": "Desc 1", "remote": True, "skills": [], "embedding": [0.1]*384}), \
         patch('backend.ingestion.scraping.SessionLocal') as mock_session_class:
        
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = MagicMock()  # Duplicate exists
        
        ingest_jobs()
        
        mock_session.add.assert_not_called()  # No insertion
        mock_session.commit.assert_called_once()


def test_ingest_jobs_error():
    with patch('backend.ingestion.scraping.fetch_jobs', side_effect=Exception("Fetch error")), \
         patch('backend.ingestion.scraping.SessionLocal') as mock_session_class:
        
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        with pytest.raises(ValueError, match="Scraping Error - Ingestion error: Fetch error"):
            ingest_jobs()
        
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()
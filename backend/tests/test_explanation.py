import json
import pytest
from unittest.mock import patch, MagicMock
from backend.explanation import explain_match, enforce_closeable_rules


MOCK_JSON_CONTENT = json.dumps({
    "summary": "Strong match with relevant Python experience.",
    "strengths": ["Python expertise", "Backend experience", "Problem solving"],
    "gaps": [
        {"skill": "Docker", "required": "2 years", "user_has": "some exposure", "closeable": True},
        {"skill": "Hands-on development experience", "required": "8 years", "user_has": "3 years", "closeable": True}
    ],
    "quick_wins": ["Get Docker certified", "Contribute to open source"]
})


def make_mock_response(content=MOCK_JSON_CONTENT):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


# --- explain_match ---

def test_explain_match_success():
    with patch('backend.explanation.client.chat.completions.create', return_value=make_mock_response()) as mock_create:
        result = explain_match("I am a Python developer", "Software Engineer", "Build scalable apps", 0.92)

        assert isinstance(result, dict)
        assert "summary" in result
        assert "strengths" in result
        assert "gaps" in result
        assert "quick_wins" in result
        mock_create.assert_called_once()
        assert mock_create.call_args.kwargs["model"] == "llama-3.3-70b-versatile"
        assert mock_create.call_args.kwargs["messages"][0]["role"] == "system"
        assert mock_create.call_args.kwargs["messages"][1]["role"] == "user"


def test_explain_match_prompt_includes_inputs():
    with patch('backend.explanation.client.chat.completions.create', return_value=make_mock_response()) as mock_create:
        explain_match("My resume text", "Data Scientist", "Analyze data", 0.85)

        prompt = mock_create.call_args.kwargs["messages"][1]["content"]
        assert "My resume text" in prompt
        assert "Data Scientist" in prompt
        assert "Analyze data" in prompt
        assert "85.0" in prompt


def test_explain_match_full_resume_in_prompt():
    long_resume = "x" * 2000
    with patch('backend.explanation.client.chat.completions.create', return_value=make_mock_response()) as mock_create:
        explain_match(long_resume, "Engineer", "Description", 0.75)

        prompt = mock_create.call_args.kwargs["messages"][1]["content"]
        assert "x" * 2000 in prompt


def test_explain_match_with_experience_context_in_prompt():
    experience = {
        "candidate_years": 3,
        "candidate_seniority": "mid",
        "job_min_years": 8,
        "job_seniority": "senior"
    }
    with patch('backend.explanation.client.chat.completions.create', return_value=make_mock_response()) as mock_create:
        explain_match("Resume text", "Senior Engineer", "Build systems", 0.6, experience=experience)

        prompt = mock_create.call_args.kwargs["messages"][1]["content"]
        assert "3 years" in prompt
        assert "mid" in prompt
        assert "8+" in prompt
        assert "senior" in prompt


def test_explain_match_closeable_enforced_for_large_gap():
    experience = {"candidate_years": 3, "candidate_seniority": "mid", "job_min_years": 8, "job_seniority": "senior"}
    with patch('backend.explanation.client.chat.completions.create', return_value=make_mock_response()):
        result = explain_match("Resume", "Engineer", "Desc", 0.5, experience=experience)

    # Gap of 8 - 3 = 5 years > 1.5, should be forced to closeable: False
    exp_gap = next(g for g in result["gaps"] if "development experience" in g["skill"])
    assert exp_gap["closeable"] is False


def test_explain_match_closeable_not_overridden_for_small_gap():
    experience = {"candidate_years": 3, "candidate_seniority": "mid", "job_min_years": None, "job_seniority": "any"}
    with patch('backend.explanation.client.chat.completions.create', return_value=make_mock_response()):
        result = explain_match("Resume", "Engineer", "Desc", 0.8, experience=experience)

    # Docker gap has "2 years" required, candidate has 3 — gap_size = 2-3 = -1 (not > 1.5), stays closeable
    docker_gap = next(g for g in result["gaps"] if g["skill"] == "Docker")
    assert docker_gap["closeable"] is True


def test_explain_match_api_error():
    with patch('backend.explanation.client.chat.completions.create', side_effect=Exception("API unavailable")):
        with pytest.raises(ValueError, match="Explanation error - Explain match failed: API unavailable"):
            explain_match("Resume", "Title", "Desc", 0.5)


# --- enforce_closeable_rules ---

def test_enforce_closeable_rules_large_gap():
    gaps = [{"skill": "Experience", "required": "8 years", "user_has": "3 years", "closeable": True}]
    result = enforce_closeable_rules(gaps, candidate_years=3.0)
    assert result[0]["closeable"] is False


def test_enforce_closeable_rules_small_gap_unchanged():
    gaps = [{"skill": "Experience", "required": "4 years", "user_has": "3 years", "closeable": True}]
    result = enforce_closeable_rules(gaps, candidate_years=3.0)
    assert result[0]["closeable"] is True


def test_enforce_closeable_rules_title_signal():
    gaps = [{"skill": "Role", "required": "staff engineer", "user_has": "senior", "closeable": True}]
    result = enforce_closeable_rules(gaps, candidate_years=10.0)
    assert result[0]["closeable"] is False


def test_enforce_closeable_rules_no_years_in_required():
    gaps = [{"skill": "Communication", "required": "Strong written skills", "user_has": "Good", "closeable": True}]
    result = enforce_closeable_rules(gaps, candidate_years=3.0)
    assert result[0]["closeable"] is True

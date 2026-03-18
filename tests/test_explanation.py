import pytest
from unittest.mock import patch, MagicMock
from backend.explanation import explain_match


def test_explain_match_success():
    mock_choice = MagicMock()
    mock_choice.message.content = "This is a great fit because..."

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch('backend.explanation.client.chat.completions.create', return_value=mock_response) as mock_create:
        result = explain_match("I am a Python developer", "Software Engineer", "Build scalable apps", 0.92)

        assert result == "This is a great fit because..."
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args
        assert call_kwargs.kwargs["model"] == "llama-3.1-8b-instant"
        assert call_kwargs.kwargs["messages"][0]["role"] == "user"


def test_explain_match_prompt_includes_inputs():
    mock_choice = MagicMock()
    mock_choice.message.content = "Some explanation"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch('backend.explanation.client.chat.completions.create', return_value=mock_response) as mock_create:
        explain_match("My resume text", "Data Scientist", "Analyze data", 0.85)

        prompt = mock_create.call_args.kwargs["messages"][0]["content"]
        assert "My resume text" in prompt
        assert "Data Scientist" in prompt
        assert "Analyze data" in prompt
        assert "0.85" in prompt


def test_explain_match_truncates_resume_to_1000_chars():
    long_resume = "x" * 2000
    mock_choice = MagicMock()
    mock_choice.message.content = "Explanation"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch('backend.explanation.client.chat.completions.create', return_value=mock_response) as mock_create:
        explain_match(long_resume, "Engineer", "Description", 0.75)

        prompt = mock_create.call_args.kwargs["messages"][0]["content"]
        # Resume is sliced to [:1000], so the prompt should not contain more than 1000 x's
        assert "x" * 1001 not in prompt
        assert "x" * 1000 in prompt


def test_explain_match_api_error():
    with patch('backend.explanation.client.chat.completions.create', side_effect=Exception("API unavailable")):
        with pytest.raises(ValueError, match="Explanation error - Explain match failed: API unavailable"):
            explain_match("Resume", "Title", "Desc", 0.5)

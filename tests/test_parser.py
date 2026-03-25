import pytest
from unittest.mock import patch, MagicMock
from backend.parser import parse_resume


def test_parse_resume_success():
    # Mock PDF content
    mock_pdf_bytes = b"fake pdf content"
    
    # Mock PyPDF2 to return dummy text
    with patch('backend.parser.PyPDF2.PdfReader') as mock_reader_class:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Sample resume text with skills."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_class.return_value = mock_reader
        
        # Mock Groq API response
        with patch('backend.parser.client.chat.completions.create') as mock_create:
            mock_response = MagicMock()
            mock_response.choices[0].message.content = '{"skills": ["Python", "JavaScript", "SQL"], "total_years_experience": 3, "seniority_level": "mid"}'
            mock_create.return_value = mock_response

            text, skills, years, seniority = parse_resume(mock_pdf_bytes)

            assert text == "Sample resume text with skills."
            assert skills == ["Python", "JavaScript", "SQL"]
            assert years == 3.0
            assert seniority == "mid"
            mock_reader_class.assert_called_once()
            mock_create.assert_called_once()


def test_parse_resume_pdf_error():
    # Test PDF reading failure
    mock_pdf_bytes = b"invalid pdf"
    
    with patch('backend.parser.PyPDF2.PdfReader', side_effect=Exception("PDF error")):
        with pytest.raises(ValueError, match="Parsing error - Resume parsing failed: PDF error"):
            parse_resume(mock_pdf_bytes)


def test_parse_resume_api_error():
    # Mock PDF success, but API failure
    mock_pdf_bytes = b"fake pdf content"
    
    with patch('backend.parser.PyPDF2.PdfReader') as mock_reader_class:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Sample text."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_class.return_value = mock_reader
        
        with patch('backend.parser.client.chat.completions.create', side_effect=Exception("API error")):
            with pytest.raises(ValueError, match="Parsing error - Resume parsing failed: API error"):
                parse_resume(mock_pdf_bytes)
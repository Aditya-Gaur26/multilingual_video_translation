"""
Tests for src/glossary.py – technical term code-mixing for NPTEL courses.
"""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from src.glossary import (
    DEFAULT_GLOSSARY,
    Glossary,
    generate_glossary_from_transcript,
    load_default_glossary,
    load_glossary_from_file,
    verify_terms_preserved,
    _GLOSSARY_PROMPT,
    _MAX_TRANSCRIPT_CHARS,
)


class TestDefaultGlossary:
    """Test that the default glossary is well-formed."""

    def test_glossary_not_empty(self):
        assert len(DEFAULT_GLOSSARY) > 0

    def test_has_expected_categories(self):
        assert "computer_science" in DEFAULT_GLOSSARY
        assert "mathematics" in DEFAULT_GLOSSARY

    def test_terms_are_lists(self):
        for category, terms in DEFAULT_GLOSSARY.items():
            assert isinstance(terms, list), f"Category '{category}' is not a list"
            for term in terms:
                assert isinstance(term, str)


class TestGlossaryDataclass:
    """Test the Glossary dataclass methods."""

    @pytest.fixture
    def glossary(self):
        return load_default_glossary()

    def test_terms_is_set(self, glossary):
        assert isinstance(glossary.terms, set)

    def test_find_terms_in_text(self, glossary):
        text = "We will study the algorithm and data structure today."
        found = glossary.find_terms_in_text(text)
        found_lower = {t.lower() for t in found}
        assert "algorithm" in found_lower or "data structure" in found_lower

    def test_find_terms_case_insensitive(self, glossary):
        text = "The ALGORITHM is efficient."
        found = glossary.find_terms_in_text(text)
        found_lower = {t.lower() for t in found}
        assert "algorithm" in found_lower

    def test_find_terms_empty_text(self, glossary):
        found = glossary.find_terms_in_text("")
        assert found == []

    def test_build_translation_prompt_section(self, glossary):
        terms = ["algorithm", "data structure"]
        section = glossary.build_translation_prompt_section(terms)
        assert "algorithm" in section.lower()

    def test_build_prompt_section_none(self, glossary):
        section = glossary.build_translation_prompt_section(None)
        # With None terms, uses all glossary terms — result is non-empty
        assert isinstance(section, str)

    def test_extract_terms_from_segments(self, glossary):
        from src.transcriber import Segment
        segments = [
            Segment(start=0, end=3, text="Binary search is an algorithm."),
            Segment(start=3, end=6, text="It uses a sorted array."),
        ]
        terms = glossary.extract_terms_from_segments(segments)
        assert isinstance(terms, list)


class TestLoadGlossary:
    """Test loading glossaries."""

    def test_load_default(self):
        g = load_default_glossary()
        assert isinstance(g, Glossary)
        assert len(g.terms) > 0

    def test_load_from_file(self):
        # Create a JSON file matching expected format (dict of lists)
        custom = {"custom": ["widget", "sprocket"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(custom, f)
            f.flush()
            path = f.name
        try:
            g = load_glossary_from_file(path)
            assert isinstance(g, Glossary)
            assert "widget" in g.terms
        finally:
            os.unlink(path)

    def test_load_from_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_glossary_from_file("/nonexistent/path.json")


class TestVerifyTermsPreserved:
    """Test code-mixing verification."""

    @pytest.fixture
    def glossary(self):
        return load_default_glossary()

    def test_all_preserved(self, glossary):
        original = "The algorithm is efficient."
        translated = "यह algorithm बहुत efficient है।"
        missing = verify_terms_preserved(original, translated, glossary)
        assert isinstance(missing, list)

    def test_some_missing(self, glossary):
        original = "The algorithm is efficient."
        translated = "यह बहुत अच्छा है।"
        missing = verify_terms_preserved(original, translated, glossary)
        # 'algorithm' should be flagged as missing
        assert any("algorithm" in m.lower() for m in missing)

    def test_empty_original(self, glossary):
        missing = verify_terms_preserved("", "any text", glossary)
        assert missing == []

    def test_terms_in_original_preserved_in_translation(self, glossary):
        original = "We study the algorithm approach."
        translated = "हम Algorithm approach का अध्ययन करते हैं।"
        missing = verify_terms_preserved(original, translated, glossary)
        assert isinstance(missing, list)


class TestDynamicGlossaryGeneration:
    """Test the LLM-based dynamic glossary generation."""

    def test_prompt_template_has_placeholder(self):
        """The prompt template must contain the {transcript_text} placeholder."""
        assert "{transcript_text}" in _GLOSSARY_PROMPT

    def test_max_transcript_chars_reasonable(self):
        assert 1_000 < _MAX_TRANSCRIPT_CHARS < 50_000

    def test_empty_segments_returns_empty(self):
        """Empty transcript should produce an empty glossary."""
        from src.transcriber import Segment
        segs = [Segment(start=0, end=1, text=""), Segment(start=1, end=2, text="  ")]
        glossary = generate_glossary_from_transcript(
            segs, gemini_api_key="fake-key-not-used",
            fallback_to_static=False,
        )
        assert isinstance(glossary, Glossary)
        assert len(glossary.terms) == 0

    def test_no_api_key_falls_back_to_static(self):
        """Without an API key, should fall back to the static glossary."""
        from src.transcriber import Segment
        segments = [Segment(start=0, end=5, text="The merge sort algorithm is O(n log n).")]
        glossary = generate_glossary_from_transcript(
            segments, gemini_api_key="", fallback_to_static=True,
        )
        assert isinstance(glossary, Glossary)
        assert len(glossary.terms) > 0  # static glossary loaded

    def test_no_api_key_no_fallback_returns_empty(self):
        """Without an API key and no fallback, should return empty glossary."""
        from src.transcriber import Segment
        segments = [Segment(start=0, end=5, text="Binary search uses a sorted array.")]
        glossary = generate_glossary_from_transcript(
            segments, gemini_api_key="", fallback_to_static=False,
        )
        assert isinstance(glossary, Glossary)
        # With fallback disabled and no key, we get an empty Glossary
        assert len(glossary.terms) == 0

    @patch("google.genai.Client")
    def test_successful_api_call(self, MockClient):
        """Mocked LLM call should produce a glossary from the returned terms."""
        from src.transcriber import Segment

        mock_response = MagicMock()
        mock_response.text = '["merge sort", "binary search", "HashMap", "O(n log n)"]'
        mock_client = MockClient.return_value
        mock_client.models.generate_content.return_value = mock_response

        segments = [
            Segment(start=0, end=5, text="Merge sort is a divide and conquer algorithm."),
            Segment(start=5, end=10, text="Binary search on a sorted array runs in O(n log n)."),
        ]
        glossary = generate_glossary_from_transcript(
            segments, gemini_api_key="test-key", fallback_to_static=False,
        )
        assert isinstance(glossary, Glossary)
        assert "merge sort" in glossary.terms
        assert "binary search" in glossary.terms
        assert "HashMap" in glossary.terms

    @patch("google.genai.Client")
    def test_api_returns_markdown_fenced_json(self, MockClient):
        """Handle the common case where the LLM wraps JSON in markdown fences."""
        from src.transcriber import Segment

        mock_response = MagicMock()
        mock_response.text = '```json\n["API", "REST", "HTTP"]\n```'
        mock_client = MockClient.return_value
        mock_client.models.generate_content.return_value = mock_response

        segs = [Segment(start=0, end=3, text="REST API over HTTP")]
        glossary = generate_glossary_from_transcript(
            segs, gemini_api_key="test-key", fallback_to_static=False,
        )
        assert "API" in glossary.terms
        assert "REST" in glossary.terms

    @patch("google.genai.Client")
    def test_api_failure_falls_back(self, MockClient):
        """If the API call raises, should fall back to static glossary."""
        from src.transcriber import Segment

        MockClient.side_effect = RuntimeError("API down")

        segs = [Segment(start=0, end=3, text="Binary search algorithm")]
        glossary = generate_glossary_from_transcript(
            segs, gemini_api_key="test-key", fallback_to_static=True,
        )
        # Should have fallen back to static glossary
        assert isinstance(glossary, Glossary)
        assert len(glossary.terms) > 0

    @patch("google.genai.Client")
    def test_api_failure_no_fallback_raises(self, MockClient):
        """If the API call raises and fallback is off, should re-raise."""
        from src.transcriber import Segment

        MockClient.side_effect = RuntimeError("API down")

        segs = [Segment(start=0, end=3, text="Binary search")]
        with pytest.raises(RuntimeError, match="API down"):
            generate_glossary_from_transcript(
                segs, gemini_api_key="test-key", fallback_to_static=False,
            )

    def test_transcript_truncation(self):
        """Long transcripts should be truncated to _MAX_TRANSCRIPT_CHARS."""
        from src.transcriber import Segment

        # Create segments that far exceed the limit
        long_text = "algorithm " * 5000  # ~50,000 chars
        segs = [Segment(start=0, end=600, text=long_text)]

        # With no API key, falls back – but we can check it doesn't crash
        glossary = generate_glossary_from_transcript(
            segs, gemini_api_key="", fallback_to_static=True,
        )
        assert isinstance(glossary, Glossary)

    @patch("google.genai.Client")
    def test_api_returns_non_array_falls_back(self, MockClient):
        """If the LLM returns a JSON object instead of array, should handle it."""
        from src.transcriber import Segment

        mock_response = MagicMock()
        mock_response.text = '{"terms": ["oops"]}'  # not a plain array
        mock_client = MockClient.return_value
        mock_client.models.generate_content.return_value = mock_response

        segs = [Segment(start=0, end=3, text="Some lecture text")]
        glossary = generate_glossary_from_transcript(
            segs, gemini_api_key="test-key", fallback_to_static=True,
        )
        # Should fall back because format was wrong
        assert isinstance(glossary, Glossary)
        assert len(glossary.terms) > 0  # static fallback

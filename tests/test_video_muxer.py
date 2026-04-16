"""
Tests for src/video_muxer.py – MKV muxing, preview MP4, and subtitle font
selection.
"""

import inspect

import pytest

from src.video_muxer import (
    mux_to_mkv,
    create_preview_mp4,
    _lang_to_iso639_2,
    _get_subtitle_font,
    _SUBTITLE_FONTS,
    _FALLBACK_FONT,
)


class TestIsoLanguageMapping:
    """Test lang code → ISO-639-2 conversion."""

    def test_english(self):
        assert _lang_to_iso639_2("en") == "eng"

    def test_hindi(self):
        assert _lang_to_iso639_2("hi") == "hin"

    def test_telugu(self):
        assert _lang_to_iso639_2("te") == "tel"

    def test_odia(self):
        assert _lang_to_iso639_2("od") == "ori"

    def test_unknown_passthrough(self):
        assert _lang_to_iso639_2("xx") == "xx"


class TestSubtitleFonts:
    """Test subtitle font selection for code-mixed text."""

    def test_hindi_font(self):
        font = _get_subtitle_font("hi")
        assert "Devanagari" in font or "Unicode" in font or "Arial" in font

    def test_telugu_font(self):
        font = _get_subtitle_font("te")
        assert "Telugu" in font or "Unicode" in font or "Arial" in font

    def test_odia_font(self):
        font = _get_subtitle_font("od")
        assert "Oriya" in font or "Unicode" in font or "Arial" in font

    def test_english_font(self):
        font = _get_subtitle_font("en")
        assert isinstance(font, str)
        assert len(font) > 0

    def test_unknown_falls_back(self):
        font = _get_subtitle_font("zz")
        assert font == _FALLBACK_FONT

    def test_none_falls_back(self):
        font = _get_subtitle_font(None)
        assert font == _FALLBACK_FONT

    def test_all_configured_fonts_are_strings(self):
        for lang, font in _SUBTITLE_FONTS.items():
            assert isinstance(font, str)
            assert len(font) > 0


class TestMuxToMkvSignature:
    """Test mux_to_mkv has expected signature."""

    def test_signature_params(self):
        sig = inspect.signature(mux_to_mkv)
        params = list(sig.parameters.keys())
        assert "original_video" in params
        assert "audio_tracks" in params
        assert "subtitle_tracks" in params
        assert "output_path" in params


class TestCreatePreviewMp4Signature:
    """Test create_preview_mp4 has expected signature."""

    def test_signature_params(self):
        sig = inspect.signature(create_preview_mp4)
        params = list(sig.parameters.keys())
        assert "original_video" in params
        assert "dubbed_audio" in params
        assert "output_path" in params
        assert "subtitle_path" in params
        assert "subtitle_lang_code" in params

    def test_subtitle_path_optional(self):
        sig = inspect.signature(create_preview_mp4)
        param = sig.parameters["subtitle_path"]
        assert param.default is None

    def test_subtitle_lang_code_optional(self):
        sig = inspect.signature(create_preview_mp4)
        param = sig.parameters["subtitle_lang_code"]
        assert param.default is None

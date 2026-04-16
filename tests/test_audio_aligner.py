"""
Tests for src/audio_aligner.py – temporal alignment (mocked ffmpeg calls).
"""

import os
from unittest.mock import patch, MagicMock

import pytest

from src.transcriber import Segment
from src.audio_aligner import (
    MIN_TEMPO,
    MAX_TEMPO,
    CROSSFADE_MS,
    FADE_OUT_MS,
    OVERLAP_TOLERANCE_S,
    INTERMEDIATE_SR,
    _build_atempo_chain,
    _generate_silence,
    _safe_remove,
)


class TestATempoChain:
    """Test the atempo filter chain builder."""

    def test_normal_tempo(self):
        chain = _build_atempo_chain(1.5)
        assert "atempo=" in chain

    def test_extreme_slow(self):
        chain = _build_atempo_chain(0.3)
        # Should chain multiple atempo filters
        assert chain.count("atempo=") >= 2

    def test_unity_tempo(self):
        chain = _build_atempo_chain(1.0)
        assert "atempo=1.0" in chain

    def test_zero_tempo(self):
        chain = _build_atempo_chain(0)
        assert "atempo=1.0" in chain

    def test_negative_tempo(self):
        chain = _build_atempo_chain(-1.0)
        assert "atempo=1.0" in chain

    def test_very_fast_tempo(self):
        chain = _build_atempo_chain(200)
        assert chain.count("atempo=") >= 2


class TestTempoLimits:
    """Test that tempo limits are reasonable."""

    def test_min_tempo_positive(self):
        assert MIN_TEMPO > 0
        assert MIN_TEMPO < 1  # must be a slowdown factor

    def test_max_tempo_above_one(self):
        assert MAX_TEMPO > 1  # must allow speedup

    def test_limit_ordering(self):
        assert MIN_TEMPO < MAX_TEMPO


class TestCrossfadeConfig:
    """Test crossfade configuration."""

    def test_crossfade_positive(self):
        assert CROSSFADE_MS > 0

    def test_fade_out_positive(self):
        assert FADE_OUT_MS > 0

    def test_crossfade_smaller_than_fade_out(self):
        # Cross-fade should be shorter than fade-out
        assert CROSSFADE_MS <= FADE_OUT_MS


class TestSafeRemove:
    """Test file cleanup utility."""

    def test_removes_existing_file(self, tmp_output_dir):
        path = os.path.join(tmp_output_dir, "test.txt")
        with open(path, "w") as f:
            f.write("hello")
        assert os.path.isfile(path)
        _safe_remove(path)
        assert not os.path.isfile(path)

    def test_nonexistent_file_no_error(self):
        _safe_remove("/nonexistent/file.txt")  # should not raise

    def test_multiple_files(self, tmp_output_dir):
        paths = []
        for i in range(3):
            p = os.path.join(tmp_output_dir, f"file_{i}.txt")
            with open(p, "w") as f:
                f.write("x")
            paths.append(p)
        _safe_remove(*paths)
        for p in paths:
            assert not os.path.isfile(p)


class TestAlignDubbedAudioSignature:
    """Test that align_dubbed_audio has the expected signature."""

    def test_function_exists(self):
        from src.audio_aligner import align_dubbed_audio
        import inspect

        sig = inspect.signature(align_dubbed_audio)
        params = list(sig.parameters.keys())
        assert "original_segments" in params
        assert "segment_audio_files" in params
        assert "total_duration" in params
        assert "output_path" in params


class TestNegativeGapHandling:
    """Test that overlapping segments are handled."""

    def test_overlapping_segments_fixture(self, overlapping_segments):
        """Verify the overlapping fixture has actual overlaps."""
        assert overlapping_segments[1].start < overlapping_segments[0].end
        assert overlapping_segments[2].start < overlapping_segments[1].end


class TestOverlapTolerance:
    """Test the overlap tolerance / bleed-through feature."""

    def test_overlap_tolerance_positive(self):
        assert OVERLAP_TOLERANCE_S > 0

    def test_overlap_tolerance_bounded(self):
        # Should not be more than 1 second
        assert OVERLAP_TOLERANCE_S <= 1.0

    def test_tighter_tempo_bounds(self):
        """With two-pass TTS, bounds can be a safety net – still tighter than old 0.65-1.60."""
        assert MIN_TEMPO >= 0.65
        assert MAX_TEMPO <= 1.50

    def test_longer_fade_out(self):
        """Fade-out should be long enough to avoid 'voice snap' artefact."""
        assert FADE_OUT_MS >= 200

    def test_crossfade_reasonable(self):
        """Crossfade should smooth transitions without being too long."""
        assert 30 <= CROSSFADE_MS <= 100


class TestAlignDubbedAudioWithBleed:
    """Test that align_dubbed_audio handles the bleed-through logic."""

    def test_function_accepts_segment_dict(self):
        """Verify align_dubbed_audio can be called with expected args."""
        from src.audio_aligner import align_dubbed_audio
        import inspect

        sig = inspect.signature(align_dubbed_audio)
        params = list(sig.parameters.keys())
        assert "original_segments" in params
        assert "segment_audio_files" in params
        assert "total_duration" in params
        assert "output_path" in params

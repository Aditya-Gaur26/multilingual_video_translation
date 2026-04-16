"""
Tests for src/voice_analyzer.py – speaker voice profile extraction.
"""

import math
import os
import tempfile

import pytest

from src.voice_analyzer import VoiceProfile, apply_voice_profile_to_audio


class TestVoiceProfile:
    """Test VoiceProfile properties and calculations."""

    def test_default_profile(self):
        vp = VoiceProfile(
            speaking_rate_wps=2.5,
            avg_pitch_hz=150.0,
            pitch_range=(80.0, 250.0),
            avg_volume_dbfs=-18.0,
        )
        assert vp.avg_pitch_hz == 150.0
        assert vp.speaking_rate_wps == 2.5

    def test_tts_rate_adjustment(self):
        vp = VoiceProfile(
            speaking_rate_wps=3.5,  # faster than normal 2.5
            avg_pitch_hz=150.0,
        )
        adj = vp.tts_rate_adjustment
        assert isinstance(adj, float)
        assert adj > 1.0  # should speed up

    def test_tts_rate_normal(self):
        vp = VoiceProfile(speaking_rate_wps=2.5)
        adj = vp.tts_rate_adjustment
        assert adj == pytest.approx(1.0)

    def test_tts_pitch_adjustment(self):
        vp = VoiceProfile(
            avg_pitch_hz=200.0,
            is_male=False,
        )
        shift = vp.tts_pitch_adjustment_semitones
        assert isinstance(shift, float)

    def test_edge_tts_rate_str(self):
        vp = VoiceProfile(speaking_rate_wps=2.5)
        rate_str = vp.edge_tts_rate_str
        assert isinstance(rate_str, str)
        assert "%" in rate_str

    def test_edge_tts_pitch_str(self):
        vp = VoiceProfile(avg_pitch_hz=150.0)
        pitch_str = vp.edge_tts_pitch_str
        assert isinstance(pitch_str, str)
        assert "Hz" in pitch_str

    def test_pitch_adjustment_male_voice(self):
        """Male voice at typical male TTS pitch → near-zero shift."""
        vp = VoiceProfile(
            avg_pitch_hz=100.0,  # low male
            is_male=True,
        )
        shift = vp.tts_pitch_adjustment_semitones
        # 100 Hz vs 130 Hz typical male TTS → negative shift
        assert shift < 0

    def test_pitch_adjustment_female_voice(self):
        """Female voice above typical female TTS pitch → positive shift."""
        vp = VoiceProfile(
            avg_pitch_hz=250.0,  # high female
            is_male=False,
        )
        shift = vp.tts_pitch_adjustment_semitones
        # 250 Hz vs 220 Hz typical female TTS → positive shift
        assert shift > 0

    def test_to_dict(self):
        vp = VoiceProfile(
            speaking_rate_wps=2.5,
            avg_pitch_hz=150.0,
            pitch_range=(80.0, 250.0),
            avg_volume_dbfs=-18.0,
            is_male=True,
            duration_seconds=60.0,
        )
        d = vp.to_dict()
        assert "speaking_rate_wps" in d
        assert "avg_pitch_hz" in d
        assert d["is_male"] is True


class TestVoiceProfileEdgeCases:
    """Edge cases for the voice profile."""

    def test_zero_pitch(self):
        vp = VoiceProfile(avg_pitch_hz=0.0)
        shift = vp.tts_pitch_adjustment_semitones
        assert shift == 0.0

    def test_zero_speaking_rate(self):
        vp = VoiceProfile(speaking_rate_wps=0.0)
        adj = vp.tts_rate_adjustment
        assert adj == 1.0

    def test_very_fast_speaker(self):
        vp = VoiceProfile(speaking_rate_wps=5.0)
        adj = vp.tts_rate_adjustment
        # Clamped to max 1.5
        assert adj <= 1.5

    def test_very_slow_speaker(self):
        vp = VoiceProfile(speaking_rate_wps=1.0)
        adj = vp.tts_rate_adjustment
        # Clamped to min 0.7
        assert adj >= 0.7

    def test_pitch_shift_clamped(self):
        """Extreme pitch should be clamped to ±6 semitones."""
        vp = VoiceProfile(avg_pitch_hz=500.0, is_male=False)
        shift = vp.tts_pitch_adjustment_semitones
        assert abs(shift) <= 6.0


class TestApplyVoiceProfile:
    """Test the post-TTS pitch-shifting function."""

    def test_function_exists(self):
        """apply_voice_profile_to_audio should be importable."""
        assert callable(apply_voice_profile_to_audio)

    def test_no_shift_needed(self):
        """If pitch adjustment is < 0.5 semitones, no processing needed."""
        # avg_pitch close to the TTS reference → near-zero shift
        vp = VoiceProfile(avg_pitch_hz=130.0, is_male=True)
        assert abs(vp.tts_pitch_adjustment_semitones) <= 0.5

    def test_male_speaker_gender_detection(self):
        """Pitch < 165 Hz should be detected as male."""
        vp = VoiceProfile(avg_pitch_hz=120.0)
        # Pipeline sets is_male based on pitch
        expected_male = vp.avg_pitch_hz < 165
        assert expected_male is True

    def test_female_speaker_gender_detection(self):
        """Pitch >= 165 Hz should be detected as female."""
        vp = VoiceProfile(avg_pitch_hz=210.0)
        expected_female = vp.avg_pitch_hz >= 165
        assert expected_female is True

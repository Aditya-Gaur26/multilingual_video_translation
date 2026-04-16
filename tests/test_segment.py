"""
Tests for the Segment dataclass (src/transcriber.py).
"""

import pytest
from src.transcriber import Segment
from src.filler_detector import FillerInfo


class TestSegmentDataclass:
    """Test Segment creation and basic properties."""

    def test_create_segment(self):
        seg = Segment(start=0.0, end=3.5, text="Hello world")
        assert seg.start == 0.0
        assert seg.end == 3.5
        assert seg.text == "Hello world"
        assert seg.fillers == []

    def test_duration_positive(self):
        seg = Segment(start=1.0, end=5.0, text="Test")
        assert seg.duration == pytest.approx(4.0)

    def test_duration_zero(self):
        seg = Segment(start=3.0, end=3.0, text="Zero")
        assert seg.duration == pytest.approx(0.0)

    def test_duration_negative_clamped(self):
        seg = Segment(start=5.0, end=3.0, text="Negative")
        assert seg.duration == pytest.approx(0.0)

    def test_fillers_default_empty(self):
        seg = Segment(start=0.0, end=1.0, text="Test")
        assert seg.fillers == []

    def test_segment_with_fillers(self):
        filler = FillerInfo(word="um", start=0.1, end=0.3, index_in_text=0)
        seg = Segment(start=0.0, end=2.0, text="Um, hello", fillers=[filler])
        assert len(seg.fillers) == 1
        assert seg.fillers[0].word == "um"


class TestSegmentSerialization:
    """Test to_dict / from_dict round-tripping."""

    def test_to_dict_basic(self):
        seg = Segment(start=1.0, end=5.5, text="Hello")
        d = seg.to_dict()
        assert d["start"] == 1.0
        assert d["end"] == 5.5
        assert d["text"] == "Hello"
        assert "fillers" not in d  # empty fillers omitted

    def test_to_dict_with_fillers(self):
        filler = FillerInfo(word="uh", start=0.5, end=0.7, index_in_text=3)
        seg = Segment(start=0.0, end=2.0, text="So uh test", fillers=[filler])
        d = seg.to_dict()
        assert "fillers" in d
        assert len(d["fillers"]) == 1
        assert d["fillers"][0]["word"] == "uh"

    def test_from_dict_basic(self):
        d = {"start": 2.0, "end": 6.0, "text": "Hello"}
        seg = Segment.from_dict(d)
        assert seg.start == 2.0
        assert seg.end == 6.0
        assert seg.text == "Hello"
        assert seg.fillers == []

    def test_from_dict_with_fillers(self):
        d = {
            "start": 0.0,
            "end": 3.0,
            "text": "Um hello",
            "fillers": [
                {"word": "Um", "start": 0.0, "end": 0.3, "index_in_text": 0},
            ],
        }
        seg = Segment.from_dict(d)
        assert len(seg.fillers) == 1
        assert isinstance(seg.fillers[0], FillerInfo)
        assert seg.fillers[0].word == "Um"

    def test_round_trip(self):
        filler = FillerInfo(word="ah", start=1.0, end=1.2, index_in_text=5)
        original = Segment(start=0.5, end=4.0, text="Well ah yes", fillers=[filler])
        d = original.to_dict()
        restored = Segment.from_dict(d)
        assert restored.start == original.start
        assert restored.end == original.end
        assert restored.text == original.text
        assert len(restored.fillers) == 1
        assert restored.fillers[0].word == "ah"


class TestSegmentRepr:
    """Test string representation."""

    def test_repr_basic(self):
        seg = Segment(start=0.0, end=3.0, text="Hello world")
        r = repr(seg)
        assert "0.00-3.00" in r
        assert "Hello world" in r

    def test_repr_with_fillers(self):
        filler = FillerInfo(word="um", start=0.0, end=0.2, index_in_text=0)
        seg = Segment(start=0.0, end=2.0, text="Um hello", fillers=[filler])
        r = repr(seg)
        assert "1F" in r  # filler count marker

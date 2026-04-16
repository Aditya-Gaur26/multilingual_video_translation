"""
Shared fixtures for the NPTEL pipeline test suite.
"""

import os
import sys
import tempfile

import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transcriber import Segment


# ── Segment fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def sample_segments() -> list[Segment]:
    """A small set of English transcript segments for testing."""
    return [
        Segment(start=0.0, end=3.5, text="Welcome to this lecture on algorithms."),
        Segment(start=3.8, end=8.2, text="Today we will discuss binary search trees."),
        Segment(start=8.5, end=12.0, text="A binary search tree is a data structure."),
        Segment(start=12.5, end=16.0, text="Um, let's start with the definition."),
        Segment(start=16.5, end=20.0, text="The time complexity is O of log n."),
    ]


@pytest.fixture
def sample_segments_with_fillers() -> list[Segment]:
    """Segments containing filler words with metadata."""
    from src.filler_detector import FillerInfo
    return [
        Segment(
            start=0.0, end=4.0,
            text="Um, welcome to this lecture.",
            fillers=[FillerInfo(word="Um", start=0.0, end=0.3, index_in_text=0)],
        ),
        Segment(start=4.5, end=8.0, text="Today we discuss algorithms."),
        Segment(
            start=8.5, end=13.0,
            text="So, uh, the algorithm works like this.",
            fillers=[FillerInfo(word="uh", start=9.0, end=9.3, index_in_text=4)],
        ),
    ]


@pytest.fixture
def overlapping_segments() -> list[Segment]:
    """Segments with timestamp overlaps (common STT artifact)."""
    return [
        Segment(start=0.0, end=5.0, text="First sentence."),
        Segment(start=4.0, end=8.0, text="Second sentence overlaps."),
        Segment(start=7.5, end=12.0, text="Third sentence also overlaps."),
    ]


@pytest.fixture
def zero_duration_segments() -> list[Segment]:
    """Segments with zero or negative duration."""
    return [
        Segment(start=3.0, end=3.0, text="Zero duration."),
        Segment(start=5.0, end=4.5, text="Negative duration."),
        Segment(start=7.0, end=10.0, text="Normal segment."),
    ]


@pytest.fixture
def translated_segments() -> dict[str, list[Segment]]:
    """Translated segments for Hindi and Telugu."""
    return {
        "hi": [
            Segment(start=0.0, end=3.5, text="इस algorithm पर व्याख्यान में आपका स्वागत है।"),
            Segment(start=3.8, end=8.2, text="आज हम binary search tree पर चर्चा करेंगे।"),
        ],
        "te": [
            Segment(start=0.0, end=3.5, text="algorithm పై ఈ ఉపన్యాసానికి స్వాగతం."),
            Segment(start=3.8, end=8.2, text="ఈ రోజు binary search tree గురించి చర్చిద్దాం."),
        ],
    }


# ── Temp directory fixture ───────────────────────────────────────────────────

@pytest.fixture
def tmp_output_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory(prefix="nptel_test_") as d:
        yield d

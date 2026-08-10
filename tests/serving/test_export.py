"""The artifacts a bass player checks, and the properties that make them checkable.

Nothing here scores a transcription -- there is no ground truth on a real song, which
is the whole reason this path exists. What is testable is whether the artifact can be
*used*: that the CSV points at a specific note, and that the preview is audible on the
hardware someone will actually play it on.
"""

import csv

import numpy as np
import pytest
import soundfile as sf

from stemify.audio.features import SAMPLE_RATE
from stemify.music.notes import NoteEvent
from stemify.serving.export import HARMONICS, note_name, notes_csv, preview_wav, render


@pytest.fixture
def notes():
    return [
        NoteEvent(onset=1.0, offset=1.5, midi_pitch=40.0, confidence=0.9),
        NoteEvent(onset=0.0, offset=0.4, midi_pitch=45.0, confidence=0.3),
        NoteEvent(onset=2.0, offset=2.6, midi_pitch=52.0),
    ]


def test_the_csv_is_in_time_order(notes, tmp_path):
    """A reviewer follows the song, so the row order has to be the order they hear the
    notes in -- otherwise finding "the wrong one around 0:12" means scanning."""
    rows = list(csv.DictReader(notes_csv(notes, tmp_path / "n.csv").open()))
    assert [r["onset_s"] for r in rows] == ["0.000", "1.000", "2.000"]
    assert [int(r["index"]) for r in rows] == [0, 1, 2]


def test_the_csv_names_notes_the_way_a_player_would(notes, tmp_path):
    """`40` is not a thing anyone plays. `E2` is."""
    rows = list(csv.DictReader(notes_csv(notes, tmp_path / "n.csv").open()))
    assert [r["note"] for r in rows] == ["A2", "E2", "E3"]


def test_the_csv_carries_confidence(notes, tmp_path):
    """The column that decides whether review is 20 notes or 500 of them."""
    rows = list(csv.DictReader(notes_csv(notes, tmp_path / "n.csv").open()))
    assert [r["confidence"] for r in rows] == ["0.300", "0.900", "1.000"]


def test_note_name_is_ascii(notes):
    """Sharps as `#`, not `♯`: this lands in a spreadsheet and in a chat message."""
    assert note_name(46.0) == "A#2"


# --------------------------------------------------------------------------- #
# The preview has to be audible, which is not automatic for bass              #
# --------------------------------------------------------------------------- #
def test_the_rendered_tone_has_energy_above_the_fundamental():
    """The reason `HARMONICS` exists. An open E is 41 Hz; laptop speakers and earbuds
    reproduce almost nothing there, so a pure sine preview is *silent* on the hardware
    anyone will check it on. The overtones carry the pitch into a band that plays, and
    the ear supplies the missing fundamental."""
    audio = render([NoteEvent(onset=0.0, offset=1.0, midi_pitch=28.0)], 1.0)  # E1, 41 Hz
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(audio.size, 1 / SAMPLE_RATE)

    def energy_near(hz, width=6.0):
        return float(spectrum[np.abs(freqs - hz) < width].max())

    assert len(HARMONICS) > 1
    assert energy_near(41.2 * 2) > 0.1 * energy_near(41.2)
    assert energy_near(41.2 * 3) > 0.05 * energy_near(41.2)


def test_notes_land_where_they_were_asked_to(tmp_path):
    """A preview whose timing drifts is worse than none -- it would send a reviewer
    hunting for a model error that is the renderer's."""
    audio = render([NoteEvent(onset=1.0, offset=1.2, midi_pitch=45.0)], 2.0)
    loud = np.abs(audio) > 0.05 * np.abs(audio).max()
    assert not loud[: int(0.95 * SAMPLE_RATE)].any()
    assert loud[int(1.0 * SAMPLE_RATE) : int(1.2 * SAMPLE_RATE)].any()
    assert not loud[int(1.3 * SAMPLE_RATE) :].any()


def test_the_preview_mixes_over_the_original(notes, tmp_path):
    """Judging "does this line up" needs both in one file; two players cannot answer it.
    The mix must be at least as long as the source, or the tail is silently dropped."""
    src = tmp_path / "src.wav"
    sf.write(src, np.zeros(int(4.0 * SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE)

    solo, _ = sf.read(preview_wav(notes, tmp_path / "solo.wav"))
    mixed, _ = sf.read(preview_wav(notes, tmp_path / "mix.wav", source=src))
    assert mixed.size > solo.size
    assert mixed.size >= int(4.0 * SAMPLE_RATE)


def test_the_preview_leaves_headroom(notes, tmp_path):
    """Overlapping notes sum, and a clipped preview reads as distortion the model
    caused. Compared with a tolerance because the file is 16-bit PCM: 0.9 does not land
    on a sample value, so the round trip returns 0.9000244."""
    audio, _ = sf.read(preview_wav(notes, tmp_path / "p.wav"))
    assert float(np.abs(audio).max()) == pytest.approx(0.9, abs=1e-3)


def test_an_empty_transcription_still_writes_playable_files(tmp_path):
    """A model that finds nothing must not crash the path that would have shown you
    that it found nothing."""
    assert notes_csv([], tmp_path / "n.csv").exists()
    audio, _ = sf.read(preview_wav([], tmp_path / "p.wav"))
    assert audio.size > 0

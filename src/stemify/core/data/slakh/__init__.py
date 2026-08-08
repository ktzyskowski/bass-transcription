"""Loader for Slakh2100: 1710 synthesized multitracks with aligned MIDI.

The scale dataset, and the only one on the roadmap that is **CC BY 4.0** --
commercially clean, unlike IDMT-SMT-Bass and MDB-bass-synth. Audio was rendered
from MIDI with 187 sampled instrument patches, so the MIDI describes the audio
exactly and labels are free.

What it uniquely provides:

  - **Note-level ground truth.** MIDI carries onsets and offsets, not just a
    contour, so Slakh is what eventually unlocks note segmentation and
    note-level P/R/F. MDB-bass-synth can never do that.
  - **Mix and isolated stem for identical material**, which turns "does the bass
    survive an arrangement" and "does source separation earn its place" into
    controlled experiments rather than guesses.

Layout on disk::

    <root>/<split>/TrackXXXXX/
        metadata.yaml       per-stem instrument class, program, render flags
        mix.flac            the full mix
        stems/SNN.flac      one per rendered source
        MIDI/SNN.mid        the MIDI that produced that stem
        all_src.mid

Splits are ``train``/``validation``/``test``, plus ``omitted`` holding tracks
whose MIDI duplicates another track's -- the redux release moves them aside so
the same material cannot land on both sides of a split. ``index`` skips
``omitted`` unless asked for it explicitly.

Two things differ from the annotated datasets and are worth knowing:

**The reference grid is a choice, not a given.** MDB-bass-synth ships a 2.9 ms
annotation grid; MIDI has no grid at all, so ``REFERENCE_HOP_S`` is ours to set.

**Bass MIDI is not guaranteed monophonic.** Slakh's MIDI comes from Lakh, i.e.
real transcriptions, which contain octave doubling and overlapping releases. The
project assumes one bass note at a time (see docs/pitch-primer.md section 5), so
``reference`` resolves overlaps by taking the lowest sounding pitch and
``polyphony_rate`` reports how often it had to -- a track with a high rate
violates the premise and should probably be filtered out rather than trusted.


Three modules, split from a single 453-line one along what changes together:

  - item    -- BassStem: what one stem is, and the constants describing it.
  - index   -- walking the tree: resolve_root, read_metadata,
    bass_stem_ids, index. Corpus *layout* knowledge.
  - dataset -- SlakhDataset: the filtering policy over that index.

Everything callers already imported is re-exported here, so
from stemify.core.data.slakh import SlakhDataset is unchanged.
"""

from stemify.core.data.slakh.dataset import SlakhDataset
from stemify.core.data.slakh.index import (
    ARCHIVE_DIR, SPLITS, bass_stem_ids, index, read_metadata, resolve_root,
)
# REFERENCE_HOP_S comes from ``midi_contour`` rather than through ``item``: the grid
# belongs to the shared MIDI->contour gate now, and re-exporting it from here keeps
# ``from stemify.core.data.slakh import REFERENCE_HOP_S`` working for callers outside.
from stemify.core.data.midi_contour import REFERENCE_HOP_S
from stemify.core.data.slakh.item import BASS_TRANSPOSE, BassStem

__all__ = [
    "ARCHIVE_DIR", "BASS_TRANSPOSE", "BassStem", "REFERENCE_HOP_S", "SPLITS",
    "SlakhDataset", "bass_stem_ids", "index", "read_metadata", "resolve_root",
]

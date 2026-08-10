"""``stemify-transcribe``: a song in, notes a bass player can check out.

    uv run stemify-transcribe song.mp3 \
        --checkpoint data/cache/semicrf_pitch_aux2.epoch4.pt --note-bias 1.5

Every number this project reports comes from Slakh, which is **synthesised**. This is
the path that finds out what that is worth on a recording, and it deliberately produces
no metric -- there is no ground truth for a song, so the output is a listen and a
worksheet.

``--note-bias`` is not optional in spirit. A semi-CRF checkpoint decoded at 0 emits
roughly a third of the notes it believes in (docs/experiments.md), so transcribing
without one measures the readout rather than the model. The value belongs to whoever
tuned it on validation; the flag makes that explicit instead of burying a constant.
"""

import argparse
import logging
from pathlib import Path

from stemify.semicrf import SemiCRFDecoder
from stemify.music.notes import NoteEvent
from stemify.serving.audio import sources_from
from stemify.serving.export import notes_csv, preview_wav
from stemify.training.checkpoint import Checkpoint

logger = logging.getLogger(__name__)


def transcribe(
    audio: Path,
    checkpoint: Path,
    out_dir: Path,
    *,
    separate: bool = True,
    note_bias: float = 0.0,
    device: str | None = None,
    solo_preview: bool = False,
) -> list[NoteEvent]:
    """Transcribe one file; write ``<stem>.notes.csv`` and ``<stem>.preview.wav``."""
    sources = sources_from(audio, separate=separate, device=device)
    logger.info("bass stem: %s", sources.bass)

    ck = Checkpoint.load(checkpoint)
    transcriber = ck.transcriber(device=device)
    cues = transcriber.encoder.encode(sources)
    decoder = transcriber.decoder
    # `note_bias` is a semi-CRF concept, and `isinstance` rather than a name check
    # because that is what makes the call below type-check: the `Decoder` protocol has
    # no such parameter, and a string comparison narrows nothing. Refused rather than
    # ignored for a greedy checkpoint -- silently dropping it would report a
    # transcription under a correction that never applied.
    if isinstance(decoder, SemiCRFDecoder):
        notes = decoder.decode(cues, note_bias=note_bias, with_confidence=True)
    elif note_bias:
        raise ValueError(
            f"--note-bias is a semi-CRF decode correction, but this checkpoint decodes "
            f"through {decoder.name!r}, which has no such parameter"
        )
    else:
        notes = decoder.decode(cues)

    stem = out_dir / audio.stem
    csv_path = notes_csv(notes, Path(f"{stem}.notes.csv"))
    wav_path = preview_wav(
        notes, Path(f"{stem}.preview.wav"),
        source=None if solo_preview else audio,
    )
    logger.info(
        "%d notes  %.1f-%.1f s  ->  %s, %s",
        len(notes),
        min((n.onset for n in notes), default=0.0),
        max((n.offset for n in notes), default=0.0),
        csv_path, wav_path,
    )
    return notes


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", type=Path, help="a song, or an isolated bass recording")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("transcriptions"))
    p.add_argument("--no-separate", action="store_true",
                   help="the input is already an isolated bass; skip demucs. Separating "
                        "an isolated stem costs minutes and adds artifacts the model "
                        "never trained on.")
    p.add_argument("--note-bias", type=float, default=0.0,
                   help="semi-CRF decode correction, tuned on validation. Decoding at 0 "
                        "emits ~a third of the notes the model believes in.")
    p.add_argument("--device", default=None)
    p.add_argument("--solo-preview", action="store_true",
                   help="render the transcription alone instead of over the original. "
                        "Better for hearing what the model thinks; worse for judging "
                        "whether it lines up.")
    args = p.parse_args()
    transcribe(
        args.audio, args.checkpoint, args.out_dir,
        separate=not args.no_separate, note_bias=args.note_bias,
        device=args.device, solo_preview=args.solo_preview,
    )


if __name__ == "__main__":
    main()

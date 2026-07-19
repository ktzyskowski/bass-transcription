# Bass Pitch for Non-Bassists

A refresher on the acoustics and music conventions behind pitch estimation,
written for engineers who don't play bass and don't have music-theory
background. Almost every design choice in this project is downstream of the one
fact in the first section, so start there.

## 1. The one fact everything hinges on: a note is not a single frequency

When an instrument plays a note, it doesn't produce one clean frequency. It
produces a **fundamental** frequency (the actual pitch, written **f₀**) *plus* a
stack of **harmonics** (overtones) at exactly 2×, 3×, 4×, 5×… the fundamental.

For the lowest note on a 4-string bass, "low E":

| What | Frequency | Note name | Relationship to f₀ |
|---|---|---|---|
| Fundamental (f₀) | 41 Hz | E1 | the true pitch |
| 2nd harmonic | 82 Hz | E2 | **one octave up** |
| 3rd harmonic | 123 Hz | B2 | octave + a fifth |
| 4th harmonic | 164 Hz | E3 | two octaves up |
| 5th harmonic | 207 Hz | G#3 | two octaves + a third |

The *relative loudness* of those harmonics is what makes a bass sound like a
bass and not a trumpet — that's **timbre**.

The critical part for us: **on bass, the fundamental is often the quietest part
of the note.** 41 Hz is so low that many speakers can't reproduce it, and the
instrument and recording chain naturally put more energy into the 82 Hz and
123 Hz harmonics than into the 41 Hz fundamental itself.

## 2. Why octave errors are *the* bass problem

A naive pitch detector finds the strongest frequency and calls that the pitch.
On a low bass note the strongest frequency is often the **82 Hz harmonic, not
the 41 Hz fundamental** — so the detector answers "E2" when the truth is "E1".
That's an **octave error**: right note letter, wrong height, off by exactly a
factor of 2 in frequency.

These errors are **structured, not random**. The model isn't guessing — it's
locking onto a real, loud, musically-related frequency that happens to be a
harmonic. That's why it's *confidently* wrong and why naive fixes don't dislodge
it.

> Observed in our own data: ~46% of the CQT peaks on the IDMT isolated notes
> landed on the +12-semitone (octave) harmonic rather than the fundamental.
> That is exactly this effect.

## 3. How pitches are named and numbered (why "MIDI 28–56" isn't arbitrary)

Music divides the frequency axis into equal steps called **semitones**. Twelve
semitones = one **octave** = a *doubling* of frequency. So pitch is a **log
scale**: each semitone multiplies frequency by 2^(1/12) ≈ 1.059.

**MIDI numbers** label those steps with integers:

- MIDI 28 = E1 = 41 Hz — the lowest note on a standard bass
- each +1 = one semitone up
- a 4-string bass tops out around MIDI 56 (G#3)

So "28-class classification" literally means: the bass plays ~28 distinct notes,
and estimating pitch = picking which of 28 log-spaced frequency bins is active.
**An octave error = being 12 classes off.**

## 4. Exact vs chroma (a.k.a. RPA vs RCA)

Because octave errors dominate, we always measure accuracy two ways:

- **Exact accuracy (RPA, "raw pitch accuracy")** — right note *including* octave.
  E1 vs E2 → wrong.
- **Chroma accuracy (RCA, "raw chroma accuracy")** — right note *ignoring*
  octave. E1 vs E2 → right (both are "E").

**The gap between them is your octave-error rate.** If exact is low but chroma is
high, you don't have a pitch problem, you have an octave problem — which is a
targeted fix (octave-aware loss, log-frequency Viterbi, harmonic-aware input),
not a data-volume problem. `mir_eval.melody` computes RPA and RCA directly.

## 5. Why "monophonic" makes bass tractable

A piano or guitar plays **chords** — several notes at once (**polyphonic**);
separating simultaneous pitches from a mixture is genuinely hard. A bass almost
always plays **one note at a time** — a single line (**monophonic**). So at any
instant there is exactly one f₀ to find, not a blend to untangle.

This is the single biggest reason bass pitch is more tractable than general
music transcription, and why off-the-shelf trackers (pyin, PESTO, CREPE) can do
well — they only ever have to commit to one answer. Exploit it.

## 6. Two ways to output pitch: frame-level vs note-level

- **Frame-level** — slice the audio into ~10 ms windows and estimate the pitch
  in each. Output is a pitch *contour* over time. It doesn't need to know where
  notes begin or end, and it is trivially self-labeling: if you generated the
  audio from MIDI, you know the pitch in every frame. PESTO/CREPE work this way.
- **Note-level** — group frames into discrete notes ("E1 from 0.5 s to 0.9 s").
  This additionally needs each note's **onset** (start) and **offset** (end).
  It's what an actual tab or score requires.

Relationship: **note-level = frame-level pitch + onset segmentation.** Starting
frame-level isolates the pure "what pitch" question from the separate "where are
the note boundaries" question.

## 7. String/fret — and why this project ignores it (for now)

A bass has 4 strings tuned E-A-D-G low-to-high, each a **fourth** (5 semitones)
apart. Pressing a string against a **fret** (metal bar) raises its pitch one
semitone per fret.

The catch: **the same pitch lives in several places.** Because strings are 5
semitones apart, fret 5 on one string produces the *identical pitch* as the next
open string. So most notes can be played 2–4 different ways, all correct, and a
human picks whichever is comfortable.

That is why fret labels are **subjective**: pitch has one right answer (physics),
but "which string and fret" has several (a human choice). Scoping fret out
removes exactly the part that needs musical judgment — leaving pitch, which is
fully objective and machine-validatable.

> Consequence for evaluation, if fret ever comes back: score string/fret as
> *playable-equivalence* (is this a valid position for the pitch?), never exact
> match, and get objective ground truth from a hex/divided pickup rather than a
> transcriber's chosen fingering.

## 8. Glossary: music term → what it means for the model

| Term | Plain meaning | ML framing |
|---|---|---|
| f₀ / fundamental | the true pitch frequency | the label you want |
| harmonics / overtones | 2×, 3×… copies stacked on top | structured distractors → octave errors |
| timbre | relative loudness of the harmonics | why bass ≠ trumpet; a nuisance variable |
| semitone / octave | one step / 12 steps = ×2 frequency | classes on a log-frequency axis; octave = 12 apart |
| MIDI number | integer name for each semitone | class index (bass ≈ 28–56) |
| monophonic | one note at a time | one label per instant — tractable |
| polyphonic | many notes at once | a mixture to untangle — hard |
| onset / offset | note start / end | segmentation, a *separate* problem from pitch |
| octave error | right note, wrong height (×2 off) | confident, structured misclassification |
| RPA / RCA | exact / octave-invariant accuracy | their gap = octave-error rate |
| technique (slide, etc.) | articulations that smear pitch | rare, ambiguous edge cases — ignore for now |

## The one sentence to remember

On bass, the pitch *class* is usually easy and the *octave* is the hard part,
because the fundamental is weak and its harmonics are loud.
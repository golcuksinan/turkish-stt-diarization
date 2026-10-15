<div align="center">

# Turkish STT with speaker diarization

**Transcribes Turkish recordings and labels who said what, word by word.**
A personal tool built on WhisperX (faster-whisper large-v3) and pyannote, plus a small benchmark script that
measures the result against hand-written references.

</div>

<details>
<summary>Table of contents</summary>

- [About the project](#about-the-project)
- [How it works](#how-it-works)
- [Measuring instead of guessing](#measuring-instead-of-guessing)
- [Built with](#built-with)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Limitations](#limitations)
- [License](#license)

</details>

## About the project

I wanted transcripts of Turkish meetings and lectures that keep the speakers apart, including the short
interruptions that most pipelines smooth away. Plain WhisperX output had three recurring problems on these
recordings:

- **Speaker labels flip in the middle of a sentence.** Diarization boundaries do not land on word boundaries,
  so one person's sentence comes back split across two speakers.
- **Whisper invents text over silence.** It gets stuck in repetition loops, or it produces memorized YouTube
  phrases such as *"İzlediğiniz için teşekkür ederim"* ("thanks for watching").
- **Turkish names and terms come out wrong.** Proper nouns and English loanwords are misheard, for example
  *Letterboxd* becomes *"Leatherbox"*.

The tool itself is two scripts. `main.py` transcribes and labels a recording. `bench.py` measures how good the
result is. The second one is the more interesting part: it tells an actual improvement apart from a change
that only looks like one.

Output is one line per speaker turn:

```
[00:00:01.654 --> 00:00:04.210] [SPEAKER_00]  Başlayalım mı?
[00:00:04.530 --> 00:00:09.880] [SPEAKER_01]  Olur, önce geçen haftanın notlarını özetleyeyim.
[00:00:10.120 --> 00:00:10.560] [SPEAKER_00]  Tamam.
```

## How it works

`main.py` runs seven stages on a CUDA GPU. Each stage's model is freed before the next one loads, because
VRAM is the limit on a 16 GB card.

1. **Model check:** downloads `Systran/faster-whisper-large-v3` (~3 GB) on the first run.
2. **Audio preparation:** ffmpeg converts the audio to 16 kHz mono with an 80 Hz high-pass filter and loudness
   normalization, so a quiet participant does not fall under the voice-activity threshold. There is deliberately
   no denoiser: on clean recordings it removed words.
3. **Transcription:** WhisperX with beam size 10, `best_of` 10 and patience 2.0. Accuracy matters more than speed
   here. Numbers are spelled out as words, because the alignment model cannot place digits in time, and words
   without timestamps cannot be given a speaker.
4. **Word alignment:** a Turkish wav2vec2 model gives every word its own start and end time.
5. **Diarization:** `pyannote/speaker-diarization-community-1` produces speaker turns.
6. **Speaker assignment:** each word gets the speaker whose turns overlap it the most. This happens per word
   rather than per segment, since speakers often change in the middle of a segment.
7. **Clean-up and output.**

The clean-up chain is where most of the logic lives. Each step targets one failure:

| Step | Problem it fixes |
|---|---|
| `strip_repetitions` | Repetition loops over silence, either a single word or a 3–5 word phrase repeated many times |
| `drop_hallucinations` | Memorized YouTube phrases. The whole sentence is dropped, since that is how the phrase always arrives |
| `assign_speakers` | A short word on a turn boundary flipping to the neighbour over a few milliseconds. Near-ties go to the previous word's speaker |
| `snap_to_sentences` | A small block inside one sentence carrying another speaker's label. It is handed back to the sentence's main speaker unless there is a pause around it |
| `smooth_speakers` | One- or two-word label flickers between two turns of the same speaker |
| `build_lines` | Breathing pauses splitting a turn into fragments |

Several of these have explicit safeguards against removing real speech. For example, `smooth_speakers` only
absorbs a short turn if it sits within a second of the speech on both sides. A short reply isolated in silence is a real
interruption, and keeping those was the point of the project.

Two smaller details:

- **Casing.** Python's `"İ".casefold()` returns `i` plus a combining dot, so a lowercase pattern does not match
  capitalized text. Both scripts normalize `I→ı` and `İ→i` first.
- **Term biasing.** Names and jargon go in as `hotwords`, a plain comma-separated list. An earlier attempt
  used a full sentence as `initial_prompt`, and Whisper repeated that sentence back over silence for 30 seconds.

## Measuring instead of guessing

`bench.py` has three modes. They answer different questions, and mixing them up was the main trap:

| Mode | What it tells you | What it cannot tell you |
|---|---|---|
| `ozet` | Descriptive statistics: number of turns, speaker switches, `UNKNOWN` share, talk time per speaker | Whether any of it is correct |
| `karsilastir` | How much two runs differ, as a timeline diff (DER with optimal speaker mapping) and a text diff | Which of the two runs is right |
| `olc` | Real error rates against a hand-written reference: DER and WER, or WDER and WER | — |

The reference for `olc` can be a few minutes long. It can also be written **without timestamps**, as plain
`Speaker: what they said` lines, since setting timestamps by hand is the expensive part. In that case the
words are aligned with Levenshtein distance, speaker labels are matched one-to-one with the Hungarian algorithm,
and the script reports the **word diarization error rate (WDER)**: the share of words given to the wrong speaker.

Results on the first five minutes of one meeting recording (267 words, 2 speakers, timestamp-free reference):

| Run | WER | WDER |
|---|---|---|
| Configuration A | 22.5 % | 19.7 % |
| Configuration B | 31.8 % | 17.2 % |
| Configuration C (latest settings) | 24.7 % | 26.2 % |
| C's words + DiariZen `wavlm-large-s80-md-v2` | — | 33.3 % |
| C's words + DiariZen `wavlm-large-s80-md` | — | 37.8 % |

Two things came out of this:

- **Tuning without a reference made speaker attribution worse.** Configuration C is the result of tuning with no
  reference to check against, and it has the highest speaker error of the three. Without a reference, the
  statistics can mislead: removing short turns is exactly what `smooth_speakers` does, so "fewer short turns"
  is not evidence of improvement.
- **DiariZen did not beat pyannote here.** On identical ASR output, both DiariZen models made more speaker
  errors, and its clustering ignores the expected speaker count. It always found four speakers.

The sample is small, and the recording has overlapping speech and room echo, so treat these numbers as
indicative only.

## Built with

[![Python][Python-badge]][Python-url]
[![PyTorch][PyTorch-badge]][PyTorch-url]
[![Hugging Face][HF-badge]][HF-url]
[![FFmpeg][FFmpeg-badge]][FFmpeg-url]

- [WhisperX](https://github.com/m-bain/whisperX) 3.8 with the
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper) backend and
  [`Systran/faster-whisper-large-v3`](https://huggingface.co/Systran/faster-whisper-large-v3)
- [`mpoyraz/wav2vec2-xls-r-300m-cv7-turkish`](https://huggingface.co/mpoyraz/wav2vec2-xls-r-300m-cv7-turkish)
  for word alignment (WhisperX's default for Turkish)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) 4 with
  [`speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [pyannote.metrics](https://github.com/pyannote/pyannote-metrics) for DER and SciPy for the Hungarian matching
  in `bench.py`

## Getting started

### Requirements

- Linux and an NVIDIA GPU with CUDA. It was developed on an RTX 5070 Ti with 16 GB of memory. There is no
  CPU path.
- Python 3.12
- ffmpeg on `PATH`. Without it the script falls back to the raw audio and skips the preparation step.
- A Hugging Face account that has accepted the terms of
  [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1),
  and an access token from that account

### Installation

```sh
git clone https://github.com/golcuksinan/turkish-stt-diarization.git
cd turkish-stt-diarization
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-frozen.txt
cp .env.example .env    # then put your Hugging Face token in .env
```

`requirements-frozen.txt` pins the exact environment the tool was run in. On a different CUDA setup,
`pip install whisperx==3.8.6` is a lighter starting point.

`HF_TOKEN` is read from `.env`, but a `HF_TOKEN` that is already set in the environment takes priority.

## Usage

```sh
python main.py recording.mp4 --konusmaci 3 --terimler "Ayşe, Mehmet, GitHub, deploy"
```

Three settings change with every recording. Any of them you leave out is asked for when the script starts:

| Option | Meaning |
|---|---|
| `recording.mp4` | Any audio or video file ffmpeg can read |
| `--konusmaci` | How many people speak: `3`, a range such as `2-4`, or `?` for unknown. This is the biggest single quality lever. Use a range if some participants may stay silent: a fixed `4` makes pyannote split a real speaker in two to fill the empty slot |
| `--terimler` | Names and recurring terms in this recording, comma-separated. They bias recognition toward these spellings. A list left over from another recording is worse than none, because its terms show up in places they were never said |
| `--cikti` | Output file. The default is `transkripts/yeni/<recording name>.txt`. An existing file is never overwritten; the name gets `-2`, `-3` and so on |

When there is no terminal to ask in, for example when another program runs the script, nothing is asked: the
recording is required, the speaker count is treated as unknown and the term list as empty.

The other settings (beam size, voice-activity thresholds, smoothing limits) are constants at the top of
`main.py`. Each has a comment explaining what it does and why it has its current value. The comments are in
Turkish.

Measuring:

```sh
python bench.py ozet "transkripts/*.txt"
python bench.py karsilastir old.txt new.txt
python bench.py olc reference.txt new.txt
```

## Limitations

- **CUDA only.** Everything runs in `float16` on the GPU.
- **GPU memory.** Transcription with `BATCH_SIZE = 8` did not fit in about 5 GB of free memory, while
  `BATCH_SIZE = 1` did. Batch size only changes speed, so lower it when other programs share the GPU.
- **Overlapping speech and echo.** When two people talk at once, or a room echoes, diarization is the weak
  point. Every configuration above split one real speaker across two or more labels.
- **Turkish only, in practice.** The language is a setting, but the hallucination patterns and casing rules are
  written for Turkish.
- **Code comments and console output are in Turkish.**

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE).

<!-- Badges -->
[Python-badge]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[PyTorch-badge]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[HF-badge]: https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black
[HF-url]: https://huggingface.co/
[FFmpeg-badge]: https://img.shields.io/badge/FFmpeg-007808?style=for-the-badge&logo=ffmpeg&logoColor=white
[FFmpeg-url]: https://ffmpeg.org/

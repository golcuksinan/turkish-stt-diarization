"""
Transkript kalite ölçümü.

Üç mod var ve hangisinin ne söylediğine dikkat et:

  ozet          Tek tek dosyaların betimleyici istatistikleri.
  karsilastir   İki çıktının FARKI. Hangisinin doğru olduğunu SÖYLEMEZ.
  olc           Elle etiketlenmiş referansa karşı gerçek DER + WER. Tek gerçek ölçüm bu.
                Referans "Kim: ne dedi" biçiminde (zaman damgasız) da olabilir;
                o zaman DER yerine kelime bazlı konuşmacı hatası (WDER) verilir.

Kullanım:
    python bench.py ozet "transkripts/baseline/*.txt" "transkripts/yeni/*.txt"
    python bench.py karsilastir "transkripts/baseline/X.txt" "transkripts/yeni/X.txt"
    python bench.py olc "transkripts/referans/X.txt" "transkripts/yeni/X.txt"
"""

import argparse
import glob
import os
import re
import statistics
import string
import sys
from dataclasses import dataclass

# main.py'nin ürettiği satır formatı:
# [00:00:01.654 --> 00:00:02.494] [SPEAKER_02]  Takım lideri.
LINE_RE = re.compile(
    r"^\[(?P<start>\d+:\d{2}:\d{2}\.\d{3})\s*-->\s*(?P<end>\d+:\d{2}:\d{2}\.\d{3})\]"
    r"\s*\[(?P<speaker>[^\]]+)\]\s*(?P<text>.*)$"
)

UNKNOWN = "UNKNOWN"

# Bu süreden kısa turlar "flicker" (konuşmacı sıçraması) sayılır.
SHORT_TURN = 0.60


@dataclass
class Turn:
    start: float
    end: float
    speaker: str
    text: str

    @property
    def duration(self):
        return max(0.0, self.end - self.start)

    @property
    def words(self):
        return self.text.split()


# =========================================================
# Ayrıştırma
# =========================================================
def parse_timestamp(value):
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_transcript(path):
    turns, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            match = LINE_RE.match(line)
            if not match:
                bad += 1
                continue
            turns.append(
                Turn(
                    start=parse_timestamp(match["start"]),
                    end=parse_timestamp(match["end"]),
                    speaker=match["speaker"].strip(),
                    text=match["text"].strip(),
                )
            )
    if bad:
        print(f"  uyarı: {path} içinde {bad} satır ayrıştırılamadı", file=sys.stderr)
    return turns


def expand_paths(patterns):
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if not matched:
            print(f"uyarı: '{pattern}' hiçbir dosyayla eşleşmedi", file=sys.stderr)
        paths.extend(matched)
    return paths


# =========================================================
# Betimleyici istatistikler
# =========================================================
def describe(turns):
    if not turns:
        return {}

    labeled = [t for t in turns if t.speaker != UNKNOWN]
    unknown = [t for t in turns if t.speaker == UNKNOWN]
    speakers = sorted({t.speaker for t in labeled})

    durations = [t.duration for t in turns]
    total_speech = sum(durations)
    span = turns[-1].end - turns[0].start

    switches = sum(
        1 for a, b in zip(turns, turns[1:]) if a.speaker != b.speaker
    )
    short_turns = [t for t in turns if t.duration < SHORT_TURN]
    word_counts = [len(t.words) for t in turns]

    per_speaker = {
        speaker: sum(t.duration for t in labeled if t.speaker == speaker)
        for speaker in speakers
    }

    return {
        "Satır (tur) sayısı": len(turns),
        "Konuşmacı sayısı": len(speakers),
        "Konuşmacı değişimi": switches,
        "Toplam konuşma (sn)": round(total_speech, 1),
        "Kayıt aralığı (sn)": round(span, 1),
        "Konuşma yoğunluğu %": pct(total_speech, span),
        "UNKNOWN tur": len(unknown),
        "UNKNOWN süre %": pct(sum(t.duration for t in unknown), total_speech),
        f"Kısa tur (<{SHORT_TURN}sn)": len(short_turns),
        "Kısa tur %": pct(len(short_turns), len(turns)),
        "Tur süresi medyan (sn)": round(statistics.median(durations), 2),
        "Tur süresi ort. (sn)": round(statistics.fmean(durations), 2),
        "Kelime sayısı": sum(word_counts),
        "Tur başına kelime (medyan)": round(statistics.median(word_counts), 1),
        "_per_speaker": per_speaker,
        "_total_speech": total_speech,
    }


def pct(part, whole):
    return round(100.0 * part / whole, 1) if whole else 0.0


def print_stats(named_stats):
    """İstatistikleri yan yana sütunlar halinde bas."""
    names = list(named_stats)
    keys = [k for k in next(iter(named_stats.values())) if not k.startswith("_")]

    label_width = max(len(k) for k in keys) + 2
    col_width = max(max(len(n) for n in names), 12) + 2

    header = " " * label_width + "".join(n.rjust(col_width) for n in names)
    print(header)
    print("-" * len(header))
    for key in keys:
        row = key.ljust(label_width)
        row += "".join(str(named_stats[n].get(key, "-")).rjust(col_width) for n in names)
        print(row)

    print("\nKonuşmacı başına süre:")
    for name in names:
        stats = named_stats[name]
        parts = [
            f"{speaker}={duration:.0f}sn({pct(duration, stats['_total_speech'])}%)"
            for speaker, duration in sorted(stats["_per_speaker"].items())
        ]
        print(f"  {name}: {', '.join(parts) or '-'}")


# =========================================================
# DER (pyannote.metrics) — zaman ekseninde konuşmacı uyuşmazlığı
# =========================================================
def to_annotation(turns, include_unknown=False):
    from pyannote.core import Annotation, Segment

    annotation = Annotation()
    for i, turn in enumerate(turns):
        if turn.speaker == UNKNOWN and not include_unknown:
            continue
        if turn.duration <= 0:
            continue
        # Aynı (start, end) çakışmalarında segment ezilmesin diye track adı veriyoruz.
        annotation[Segment(turn.start, turn.end), f"t{i}"] = turn.speaker
    return annotation


def diarization_diff(ref_turns, hyp_turns):
    import warnings

    from pyannote.metrics.diarization import DiarizationErrorRate

    reference = to_annotation(ref_turns)
    hypothesis = to_annotation(hyp_turns)

    metric = DiarizationErrorRate()
    with warnings.catch_warnings():
        # UEM vermiyoruz; pyannote iki dosyanın birleşimini kullanıyor, istediğimiz de bu.
        warnings.filterwarnings("ignore", message=".*uem.*")
        detail = metric(reference, hypothesis, detailed=True)
        mapping = metric.optimal_mapping(reference, hypothesis)
    return detail, mapping


def print_der(detail, mapping, ref_name, hyp_name, as_error):
    total = detail["total"]
    title = "DER (hata oranı)" if as_error else "Zaman ekseninde farklılık"
    print(f"\n{title}: {detail['diarization error rate'] * 100:.1f}%")
    print(f"  referans konuşma süresi : {total:.0f} sn  ({ref_name})")
    print(f"  karışma (confusion)     : {detail['confusion']:.0f} sn  "
          f"({pct(detail['confusion'], total)}%)  → yanlış konuşmacıya yazılmış")
    print(f"  atlanan (missed)        : {detail['missed detection']:.0f} sn  "
          f"({pct(detail['missed detection'], total)}%)  → referansta var, {hyp_name} kaçırmış")
    print(f"  fazladan (false alarm)  : {detail['false alarm']:.0f} sn  "
          f"({pct(detail['false alarm'], total)}%)  → referansta yok, {hyp_name} uydurmuş")
    print(f"  konuşmacı eşlemesi      : {mapping or '-'}")


# =========================================================
# WER — metin doğruluğu
# =========================================================
def normalize_words(text):
    """Türkçe'ye uygun küçültme + noktalama temizliği."""
    text = text.replace("I", "ı").replace("İ", "i").lower()
    table = str.maketrans("", "", string.punctuation + "“”‘’…–—")
    return text.translate(table).split()


def word_error_rate(ref_words, hyp_words):
    """Klasik Levenshtein tabanlı WER. Ek bağımlılık istemesin diye elde yazıldı."""
    previous = list(range(len(hyp_words) + 1))
    for i, ref_word in enumerate(ref_words, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp_words, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    distance = previous[-1]
    return distance / len(ref_words) if ref_words else 0.0, distance


# =========================================================
# Zaman damgasız referans — "Kim: ne dedi" biçimi
# =========================================================
# Elle süre ayarlamak zahmetli olduğu için referans şöyle de yazılabilir:
#   SPEAKER_00: Toplantıya biraz geç kaldım, kusura bakmayın.
#   SPEAKER_02: Sorun değil, daha yeni başladık.
# Etiket isimleri serbest (çıktıyla en iyi eşleme kendiliğinden bulunur). İki nokta
# içermeyen satır bir önceki konuşmacının devamıdır, # ile başlayan satır yorumdur,
# parantez içi "(gülüşme)", "(?)" gibi notlar konuşma sayılmaz.
NAMED_LINE_RE = re.compile(r"^(?P<speaker>[^:\[\]#()]{1,30}?)\s*:\s*(?P<text>.*)$")
ANNOTATION_RE = re.compile(r"\([^)]*\)")


def is_timed_transcript(path):
    """Dosya main.py'nin zaman damgalı biçiminde mi?"""
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("#"):
                return bool(LINE_RE.match(line))
    return False


def parse_named_transcript(path):
    """'Kim: metin' satırlarını (konuşmacı, kelime) listesine çevirir."""
    words, speaker = [], None
    with open(path, encoding="utf-8") as f:
        for number, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            match = NAMED_LINE_RE.match(line)
            if match:
                speaker, text = match["speaker"].strip(), match["text"]
            elif speaker is None:
                print(f"  uyarı: {path}:{number} konuşmacısız satır atlandı", file=sys.stderr)
                continue
            else:
                text = line
            text = ANNOTATION_RE.sub(" ", text)
            words.extend((speaker, word) for word in normalize_words(text))
    return words


def speaker_words(turns):
    """Zaman damgalı turları (konuşmacı, kelime) listesine çevirir."""
    return [(t.speaker, word) for t in turns for word in normalize_words(t.text)]


def align_words(ref_words, hyp_words):
    """
    Levenshtein hizalaması + geri izleme. Hipotezin SONU bedava: referans kaydın
    ilk birkaç dakikası olabilir, çıktının geri kalanı hata sayılmasın.
    Döner: (mesafe, [(ref_index, hyp_index)] eşleşen/yerine geçen çiftler, hyp'in
    kullanılan uzunluğu).
    """
    n, m = len(ref_words), len(hyp_words)
    # Tam matris gerekiyor (geri izleme için); 5 dakikalık referansta ~1M hücre.
    table = [list(range(m + 1))]
    for i in range(1, n + 1):
        row = [i] + [0] * m
        prev = table[-1]
        ref_word = ref_words[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ref_word == hyp_words[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)
        table.append(row)

    end = min(range(m + 1), key=lambda j: table[n][j])
    pairs, i, j = [], n, end
    while i > 0 and j > 0:
        cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
        if table[i][j] == table[i - 1][j - 1] + cost:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif table[i][j] == table[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return table[n][end], pairs, end


def word_speaker_error(ref_speakers, hyp_speakers, pairs):
    """
    Hizalanmış kelime çiftleri üzerinden konuşmacı hatası (WDER). Etiket isimleri
    birbirini tutmadığından önce en çok kelimeyi doğru tutturan bire bir eşleme
    bulunur (Macar algoritması), sonra eşlemeye uymayanlar sayılır.
    """
    from scipy.optimize import linear_sum_assignment

    refs = sorted({ref_speakers[i] for i, _ in pairs})
    hyps = sorted({hyp_speakers[j] for _, j in pairs if hyp_speakers[j] != UNKNOWN})
    counts = [[0] * len(hyps) for _ in refs]
    for i, j in pairs:
        if hyp_speakers[j] != UNKNOWN:
            counts[refs.index(ref_speakers[i])][hyps.index(hyp_speakers[j])] += 1

    mapping = {}
    if refs and hyps:
        rows, cols = linear_sum_assignment(counts, maximize=True)
        mapping = {hyps[c]: refs[r] for r, c in zip(rows, cols)}

    errors = sum(1 for i, j in pairs if mapping.get(hyp_speakers[j]) != ref_speakers[i])
    return errors, mapping, refs, hyps, counts


def olc_named(ref_path, hyp_path):
    ref_name, hyp_name = short_name(ref_path), short_name(hyp_path)
    ref = parse_named_transcript(ref_path)
    hyp = speaker_words(parse_transcript(hyp_path))
    if not ref or not hyp:
        print("HATA: dosyalardan biri boş veya ayrıştırılamadı", file=sys.stderr)
        return 1

    ref_speakers, ref_words = zip(*ref)
    hyp_speakers, hyp_words = zip(*hyp)
    distance, pairs, used = align_words(ref_words, hyp_words)

    print(f"Zaman damgasız referans: {ref_name} ({len(ref_words)} kelime, "
          f"{len(set(ref_speakers))} konuşmacı)")
    print(f"Hipotezin ilk {used}/{len(hyp_words)} kelimesi referansa hizalandı, "
          f"gerisi ölçüm dışı.")

    print(f"\nWER: {distance / len(ref_words) * 100:.1f}%  "
          f"({distance} hata / {len(ref_words)} kelime)")

    errors, mapping, refs, hyps, counts = word_speaker_error(ref_speakers, hyp_speakers, pairs)
    print(f"\nKonuşmacı hatası (WDER): {pct(errors, len(pairs))}%  "
          f"({errors} / {len(pairs)} hizalanmış kelime yanlış kişiye yazılmış)")
    print(f"  konuşmacı eşlemesi: {mapping or '-'}")

    # Kim kiminle karıştırılıyor: satır = referans, sütun = çıktının etiketi.
    width = max([len(s) for s in refs + hyps] + [8]) + 2
    print(f"\n  Kelime sayıları (satır: referans, sütun: {hyp_name}):")
    print("  " + " " * width + "".join(h.rjust(width) for h in hyps))
    for r, ref_speaker in enumerate(refs):
        print("  " + ref_speaker.ljust(width) + "".join(str(c).rjust(width) for c in counts[r]))
    unknown = sum(1 for _, j in pairs if hyp_speakers[j] == UNKNOWN)
    if unknown:
        print(f"  ({unknown} kelime UNKNOWN etiketli, hata sayıldı)")
    print("\nNot: zaman damgası olmadığı için DER hesaplanamaz; WDER sadece hizalanan")
    print("kelimelerin konuşmacısını ölçer — atlanan konuşma WER'e yansır.")
    return 0


# =========================================================
# Modlar
# =========================================================
def cmd_ozet(args):
    paths = expand_paths(args.dosyalar)
    if not paths:
        return 1
    named_stats = {}
    for path in paths:
        turns = parse_transcript(path)
        if not turns:
            print(f"uyarı: {path} boş veya ayrıştırılamadı", file=sys.stderr)
            continue
        named_stats[short_name(path)] = describe(turns)
    if not named_stats:
        return 1
    print_stats(named_stats)
    print("\nNot: bunlar betimleyici sayılar, doğruluk ölçüsü DEĞİL.")
    print("Gerçek kalite için 'olc' moduna elle etiketlenmiş bir referans ver.")
    return 0


def cmd_karsilastir(args):
    ref_turns = parse_transcript(args.baseline)
    hyp_turns = parse_transcript(args.yeni)
    if not ref_turns or not hyp_turns:
        print("HATA: dosyalardan biri boş veya ayrıştırılamadı", file=sys.stderr)
        return 1

    ref_name, hyp_name = short_name(args.baseline), short_name(args.yeni)
    print_stats({ref_name: describe(ref_turns), hyp_name: describe(hyp_turns)})

    detail, mapping = diarization_diff(ref_turns, hyp_turns)
    print_der(detail, mapping, ref_name, hyp_name, as_error=False)

    ref_words = normalize_words(" ".join(t.text for t in ref_turns))
    hyp_words = normalize_words(" ".join(t.text for t in hyp_turns))
    rate, distance = word_error_rate(ref_words, hyp_words)
    print(f"\nMetin farkı: {rate * 100:.1f}%  ({distance} kelime, referans {len(ref_words)} kelime)")

    print("\n" + "!" * 70)
    print("DİKKAT: baseline bir GERÇEK DEĞER (ground truth) değil, sadece eski çıktı.")
    print("Buradaki sayılar 'ne kadar değişti'yi ölçer, 'hangisi doğru'yu değil.")
    print("Ayrıca 'kısa tur' sayısındaki düşüş main.py'deki yumuşatmanın tanımı gereği")
    print("olan şeydir — kendi başına iyileşme kanıtı sayılmaz.")
    print("!" * 70)
    return 0


def cmd_olc(args):
    if not is_timed_transcript(args.referans):
        return olc_named(args.referans, args.yeni)

    ref_turns = parse_transcript(args.referans)
    hyp_turns = parse_transcript(args.yeni)
    if not ref_turns or not hyp_turns:
        print("HATA: dosyalardan biri boş veya ayrıştırılamadı", file=sys.stderr)
        return 1

    ref_name, hyp_name = short_name(args.referans), short_name(args.yeni)

    # Referans elle etiketlenmiş kısa bir bölüm olabilir; ölçümü o aralıkla sınırla.
    ref_start, ref_end = ref_turns[0].start, ref_turns[-1].end
    hyp_turns = [t for t in hyp_turns if t.end > ref_start and t.start < ref_end]
    if not hyp_turns:
        print(f"HATA: {hyp_name} içinde {ref_start:.0f}-{ref_end:.0f}sn aralığında satır yok",
              file=sys.stderr)
        return 1
    print(f"Ölçüm aralığı: {ref_start:.0f}-{ref_end:.0f} sn ({ref_end - ref_start:.0f} sn)")

    print_stats({ref_name: describe(ref_turns), hyp_name: describe(hyp_turns)})

    detail, mapping = diarization_diff(ref_turns, hyp_turns)
    print_der(detail, mapping, ref_name, hyp_name, as_error=True)

    ref_words = normalize_words(" ".join(t.text for t in ref_turns))
    hyp_words = normalize_words(" ".join(t.text for t in hyp_turns))
    rate, distance = word_error_rate(ref_words, hyp_words)
    print(f"\nWER: {rate * 100:.1f}%  ({distance} hata / {len(ref_words)} kelime)")
    return 0


def short_name(path):
    parent = os.path.basename(os.path.dirname(path))
    return f"{parent}/{os.path.basename(path)}" if parent else os.path.basename(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mod", required=True)

    p_ozet = sub.add_parser("ozet", help="dosyaların betimleyici istatistikleri")
    p_ozet.add_argument("dosyalar", nargs="+")
    p_ozet.set_defaults(func=cmd_ozet)

    p_kars = sub.add_parser("karsilastir", help="iki çıktı arasındaki fark (doğruluk değil)")
    p_kars.add_argument("baseline")
    p_kars.add_argument("yeni")
    p_kars.set_defaults(func=cmd_karsilastir)

    p_olc = sub.add_parser("olc", help="elle etiketli referansa karşı DER + WER")
    p_olc.add_argument("referans")
    p_olc.add_argument("yeni")
    p_olc.set_defaults(func=cmd_olc)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

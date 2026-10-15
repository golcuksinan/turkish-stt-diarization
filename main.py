"""
Türkçe toplantı kayıtları için transkript + konuşmacı ayrımı (diarization).

Akış:
    1. Model yoksa indir
    2. Sesi ffmpeg ile normalize et (16 kHz mono, highpass, loudnorm)
    3. Whisper ile transkript
    4. Kelime bazlı hizalama (forced alignment)
    5. pyannote ile konuşmacı ayrımı
    6. Konuşmacı etiketlerini kelimelere ata + temizle + yumuşat
    7. Konuşmacı değişimlerine göre satırla ve yaz
"""

import argparse
import datetime
import gc
import os
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlparse

import torch
import whisperx
from huggingface_hub import snapshot_download

# =========================================================
# 1. AYARLAR
# =========================================================
# Kayda özel üç ayar (ses dosyası, konuşmacı sayısı, terim listesi) bu dosyada
# sabit değil: komut satırından verilir, verilmeyen çalışınca sorulur
# (python main.py --help). Kayıttaki kişilerin isimleri böylece koda hiç girmez.

# Çıktı varsayılan olarak buraya, kaydın adıyla yazılır. Aynı adlı dosya varsa
# üzerine yazılmaz, -2, -3... eki alır; eski çalıştırmayla karşılaştırma bozulmasın.
OUTPUT_DIR = "transkripts/yeni"

LANGUAGE = "tr"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
# Batched pipeline'da her VAD parçası birbirinden bağımsız çözülür; batch_size
# sadece hızı etkiler, çıktıyı değiştirmez.
BATCH_SIZE = 8

# Model yerel CTranslate2 formatında tutulur. Klasör yoksa otomatik indirilir (~3 GB).
MODEL_PATH = "./models/large-v3"
MODEL_REPO = "Systran/faster-whisper-large-v3"

# Konuşmacı sayısı (--konusmaci) en büyük kalite kolu: kesin sayı verilirse pyannote
# tam o sayıya kümeleme yapar ve sonuç belirgin şekilde keskinleşir. Bilinmiyorsa
# boş bırakılır, pyannote kendisi tahmin eder.
#
# Odada 4 kişi olup 1-2'si hiç konuşmuyorsa "4" vermek sessiz kişiyi "icat etmeye"
# zorlar ve konuşan birini ikiye böler. Böyle kayıtlarda "2-4" gibi bir aralık daha
# güvenli.

# --- Ses ön işleme --------------------------------------------------------
# mp4'ün sesi hem ASR'ın hem diarization'ın girdisi; ikisi de 16 kHz mono bekliyor.
# Dönüşümü ffmpeg'e açıkça yaptırmak, üstüne bandı temizlemek (highpass ile mırıltı
# ve masa gürültüsü) ve seviyeyi eşitlemek (loudnorm — kısık konuşan katılımcı VAD
# eşiğinin altında kalmasın) hem kelime hem konuşmacı doğruluğunu artırıyor.
# Gürültü bastırma (afftdn) bilerek yok: temiz kayıtta kelime yiyor.
PREPROCESS_AUDIO = True
PREPROCESS_FILTER = "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11"
PREPROCESS_DIR = "audio_cache"

# --- Tanıma doğruluğu -----------------------------------------------------
# Sık geçen özel isimler ve terimler (--terimler); hotwords olarak her parçaya
# yönelim verilir. Whisper Türkçe özel isimleri ve İngilizce terimleri (Letterboxd ->
# "Leatherbox") sürekli kaçırıyor. Liste her kayıt için ayrı verilir.
#
# Bilerek düz virgüllü liste: cümle biçiminde bir initial_prompt ("Geçen isimler ve
# terimler: ...") denendi ve sessiz bölümlerde modelin çıktı olarak geri kustuğu
# görüldü — 30 saniyelik "Geçen isimler ve terimler." tekrarı. Akıcı cümle
# görünümündeki prompt halüsinasyona davetiye; liste biçimi bu riski azaltıyor,
# kalanını strip_repetitions temizliyor.
#
# Başka kaydın listesi hiç liste olmamasından kötü: terimler sızıyor (listede DEHB
# varken "aklıma o geldi" -> "aklıma DEHB geldi" oldu).
#
# Örnek, bir özel eğitim dersi için:
#   --terimler "özel eğitim, özel gereksinim, yetersizlik, zihin yetersizliği,
#   öğrenme güçlüğü, otizm, DEHB, hiperaktivite, nörolojik, bilişsel, süreğen,
#   Braille, işaret dili, tekerlekli sandalye, kaynaştırma"
#
# Boş liste -> boş string -> faster-whisper hotwords'ü hiç eklemez.

# Doğruluk için hız feda ediliyor: geniş ışın (beam) ve yüksek best_of, düşük
# olasılıklı çözümlerin elenmesi için sabır (patience).
BEAM_SIZE = 10
BEST_OF = 10
PATIENCE = 2.0

# Whisper'ın sessizlikte takılıp aynı kelimeyi tekrarlama ("üçüncü, üçüncü,
# üçüncü...") huyuna karşı hafif ceza. Yükseltmek gerçek tekrarları da bozar.
REPETITION_PENALTY = 1.05
# Tekrar sarmalına girmiş parçanın yeniden, daha yüksek sıcaklıkla çözülmesi için
# eşik. Varsayılan 2.4 Türkçe'de geç tetikleniyor.
COMPRESSION_RATIO_THRESHOLD = 2.2

# Rakamları bastır: model "5-6" yerine "beş altı" yazar. Hizalama modeli rakam
# karakterlerini sese oturtamadığı için rakamlı kelimeler kelime zamanı üretmiyor,
# o da doğrudan konuşmacı atamasını bozuyor. Okunabilirlik uğruna kapatılabilir.
SUPPRESS_NUMERALS = True

# Whisper sessizlik/gürültü üstünde eğitim verisinden ezberlediği YouTube
# kalıplarını uyduruyor ("İzlediğiniz için teşekkür ederim", "Altyazı M.K."). Bu
# kalıplardan birine UYAN CÜMLENİN TAMAMI atılır — kelime kelime değil, çünkü
# uydurma hep bütün bir cümle olarak geliyor. Kalıpları dar tut: "teşekkür ederim"
# tek başına gerçek konuşmada da geçiyor, sadece video kalıbıyla birlikte eleniyor.
HALLUCINATION_PATTERNS = [
    r"izlediğiniz için teşekkür",
    r"bir sonraki videoda görüşmek üzere",
    r"kanalıma? abone ol",
    r"altyazı m\.?k\.?",
    r"altyazı ve[ ']?ya çeviri",
    r"bu videoyu beğendiyseniz",
]

# --- Keskinlik ayarları ---------------------------------------------------
# pyannote binarize eşikleri (histerezis): skor ONSET'i geçince konuşma başlar,
# OFFSET'in altına düşünce biter. ONSET düşürmek konuşma başlangıcını daha erken
# yakalar, OFFSET DÜŞÜRMEK ise konuşmanın erken kesilmesini engeller
# (yükseltmek değil — eski yorum bunu ters yazıyordu).
VAD_ONSET = 0.400
VAD_OFFSET = 0.300

# Etiketsiz kelimeleri en yakın konuşmacıya ata. True ise çıktıda [UNKNOWN] olmaz,
# ama sessizlik/gürültü de bir konuşmacıya yazılır. Kısa boşluklar çoksa True dene.
FILL_NEAREST = False

# Kelimenin konuşmacısı, kelime aralığıyla diarization turlarının kesişimine bakılarak
# seçilir. İki turun sınırına denk gelen kelimede kazanan çoğu zaman kıl payı önde:
# önde olan iki konuşmacının kesişimleri bu kadar (saniye) yakınsa karar bir önceki
# kelimenin konuşmacısına bırakılır. Sınırdaki kelimeyi keyfi olarak komşuya
# kaptırmak yerine konuşmayı sürdürmek daha sık doğru çıkıyor.
# 0 yaparsan saf en-büyük-kesişim (whisperx'in davranışı) olur.
WORD_TIE_MARGIN = 0.10

# Aynı kelimenin ya da aynı kelime öbeğinin üst üste bu sayıdan fazla tekrarı
# halüsinasyon sayılıp kırpılır. 0 yaparsan kırpma kapanır.
MAX_WORD_REPEAT = 3
# Tekrar taramasında bakılacak en uzun öbek (kelime). Whisper sessizlikte tek
# kelimeye değil, çoğu zaman 3-5 kelimelik bir kalıba takılıyor.
MAX_REPEAT_PHRASE = 6

# --- Cümle içi konuşmacı sıçraması ----------------------------------------
# Diarization sınırları kelime sınırlarına oturmadığı için cümlenin ortasında
# konuşmacı değişebiliyor ("Ben de veri" / "analistiyim."). Cümle benzeri
# birimler içinde, arada nefes bile olmadan gelen küçük azınlık bloklarını
# birimin baskın konuşmacısına devrediyoruz.
SENTENCE_SNAP = True
# Bir cümle biriminin bittiğini kabul ettiğimiz sessizlik (noktalama da bitirir).
SENTENCE_GAP = 0.80
# Değişim noktasının iki yanından birinde bu kadar duraklama varsa devir yapılmaz:
# gerçekten söz alınmış olabilir.
SNAP_MIN_PAUSE = 0.25
# Azınlık blok, cümlenin bu oranından fazlasını kaplıyorsa dokunulmaz. Birimler
# zaten cümle sonu noktalamasında bölündüğü için buradaki her değişim tanım gereği
# cümle ORTASINDA; asıl koruma oran değil, SNAP_MIN_PAUSE duraklama şartı. 0.50 =
# "süre olarak azınlıkta kalan taraf". Düşürmek fonksiyonu etkisizleştirir: gerçek
# sınır hataları cümlenin %40'ını rahat kaplıyor.
SNAP_MINORITY_RATIO = 0.50

# Bu süreden kısa süren ve iki yanı aynı konuşmacı olan etiket sıçramaları
# gürültü sayılıp komşuya devredilir. 0 yaparsan yumuşatma kapanır.
MIN_TURN_DURATION = 0.60

# Kısa bir bloğun "etiket titremesi" sayılabilmesi için komşu konuşmaya en fazla
# bu kadar uzak olması gerekir. Daha uzaktaki kısa sözler sessizliğin ortasında
# gerçekten alınmış söz kabul edilip korunur. Yükseltmek gerçek araya girmeleri
# yutmaya başlar — MERGE_GAP'ten ayrı tutulmasının sebebi bu.
ADJACENT_GAP = 1.00

# Aynı konuşmacının ardışık kelimeleri arasındaki bu süreden kısa boşluklar
# yeni satır açmaz, tek satırda birleştirilir. Sadece okunabilirliği etkiler:
# yükseltmek satırları uzatır ama satır içine giren sessizliği de artırır.
MERGE_GAP = 1.50
# --------------------------------------------------------------------------

UNKNOWN = "UNKNOWN"
SENTENCE_END_RE = re.compile(r"[.!?…]+[\"'”’)\]]*$")


# =========================================================
# 2. YARDIMCI FONKSİYONLAR
# =========================================================
def load_env(path=".env"):
    """
    .env dosyasını ortam değişkenlerine yükler. python-dotenv kurulu değil,
    ihtiyacımız da bu kadarla sınırlı: KEY=value, # yorum, opsiyonel tırnak.
    Zaten tanımlı ortam değişkenlerinin üzerine yazmaz.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def format_time(seconds):
    if seconds is None:
        seconds = 0.0
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def free_vram():
    """
    Aşaması biten modeli bellekten düşürür. VRAM burada darboğaz.

    Çağırmadan önce modelin değişkeni `del` ile silinmeli. Modeli argüman olarak
    almak işe yaramıyor: fonksiyon içindeki `del` sadece kendi yerel adını siler,
    main()'deki değişken modeli tutmaya devam eder (Whisper 4 GB'ı böyle bırakmıyordu).
    """
    gc.collect()
    torch.cuda.empty_cache()


def ensure_model():
    if os.path.isdir(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "model.bin")):
        return
    print(f"Model bulunamadı. '{MODEL_REPO}' indiriliyor (~3 GB), lütfen bekleyin...")
    snapshot_download(repo_id=MODEL_REPO, local_dir=MODEL_PATH)
    print("İndirme tamamlandı.")


def prepare_audio(path):
    """
    Sesi 16 kHz mono WAV'a çevirip bandı temizler ve seviyeyi eşitler.
    Sonucu PREPROCESS_DIR altında saklar; aynı kayıt ikinci kez işlenmez.
    ffmpeg yoksa sessizce ham dosyaya döner.
    """
    if not PREPROCESS_AUDIO:
        return path
    if shutil.which("ffmpeg") is None:
        print("UYARI: ffmpeg bulunamadı, ses ön işleme atlanıyor.")
        return path

    os.makedirs(PREPROCESS_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(PREPROCESS_DIR, f"{name}.16k.wav")
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(path):
        print(f"Hazırlanmış ses kullanılıyor: {out}")
        return out

    print("Ses hazırlanıyor (16 kHz mono, highpass + loudnorm)...")
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-vn", "-af", PREPROCESS_FILTER,
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out


def flatten_words(result):
    """
    Segmentleri düz bir kelime listesine indirger: [start, end, speaker, text].

    Segment bazlı çalışmak, tek segment içinde konuşmacı değiştiğinde bunu
    kaçırır; asıl keskinlik kelime seviyesinde gezmekten geliyor. Hizalanamamış
    (kelime bilgisi olmayan) segmentler tek parça olarak listeye düşer.
    """
    words = []
    for segment in result["segments"]:
        seg_words = segment.get("words", [])

        if not seg_words:
            text = segment.get("text", "").strip()
            if text:
                words.append(
                    [segment.get("start"), segment.get("end"),
                     segment.get("speaker", UNKNOWN), text]
                )
            continue

        for word in seg_words:
            text = word.get("word", "").strip()
            if not text:
                continue
            words.append(
                [word.get("start"), word.get("end"),
                 word.get("speaker", UNKNOWN), text]
            )

    # Zamansız kelimeler (nadir) bir öncekinin bitişine yapıştırılır ki
    # sonraki adımlar zaman aritmetiğinde None ile uğraşmasın.
    for i, w in enumerate(words):
        if w[0] is None or w[1] is None:
            prev_end = words[i - 1][1] if i > 0 else 0.0
            w[0] = prev_end if w[0] is None else w[0]
            w[1] = w[0] if w[1] is None else w[1]

    return words


def lower_tr(text):
    """
    Türkçe'ye uygun küçültme. Python'un casefold'u 'İ' için i + U+0307 (birleşen
    nokta) üretiyor, o yüzden düz 'izlediğiniz' kalıbı 'İzlediğiniz' ile eşleşmiyor.
    bench.py'daki normalize_words ile aynı dönüşüm.
    """
    return text.replace("I", "ı").replace("İ", "i").casefold().replace("̇", "")


def _repeat_key(word):
    return lower_tr(word[3].strip(" ,.!?;:…\"'"))


def strip_repetitions(words):
    """
    Whisper'ın sessizlik/gürültü üstünde takılıp ürettiği tekrar sarmallarını kırpar.

    İki biçimi de var: tek kelime ("üçüncü, üçüncü, üçüncü...") ve öbek ("Geçen
    isimler ve terimler." x8 — hotwords/prompt sızıntısı). O yüzden 1'den
    MAX_REPEAT_PHRASE'e kadar her öbek uzunluğu için art arda tekrar aranır;
    MAX_WORD_REPEAT kopyaya izin verilip fazlası atılır. Uzun öbekten kısaya
    doğru gidiyoruz ki "a b a b a b" önce öbek olarak yakalansın.
    """
    if MAX_WORD_REPEAT <= 0:
        return words

    for size in range(MAX_REPEAT_PHRASE, 0, -1):
        words = _strip_phrase_repeats(words, size)
    return words


def _strip_phrase_repeats(words, size):
    """Art arda gelen 'size' kelimelik özdeş öbeklerden ilk MAX_WORD_REPEAT tanesini tutar."""
    keys = [_repeat_key(w) for w in words]
    kept = []
    i = 0
    while i < len(words):
        block = keys[i:i + size]
        if len(block) < size or not any(block):
            kept.append(words[i])
            i += 1
            continue

        # Bu öbek buradan itibaren kaç kez üst üste tekrar ediyor?
        count = 0
        while keys[i + count * size:i + (count + 1) * size] == block:
            count += 1

        limit = min(count, MAX_WORD_REPEAT) if count > MAX_WORD_REPEAT else count
        kept.extend(words[i:i + limit * size])
        i += count * size
    return kept


def _overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(turns, words, fill_nearest=FILL_NEAREST):
    """
    Her kelimeye, diarization turlarıyla kesişimi en büyük olan konuşmacıyı yazar.

    whisperx'in kendi atamasından iki farkı var:
      - segment değil, düz kelime listesi üstünde çalışır (hizalanamamış segmentler
        de tek parça olarak buradan etiketlenir),
      - berabere kalan sınır kelimelerinde (WORD_TIE_MARGIN) bir önceki kelimenin
        konuşmacısını sürdürür. İki tur arasına düşen kısa kelime ("ben", "ya", "de")
        aksi halde birkaç ms fark yüzünden komşuya geçiyor ve cümlenin ortasında
        sahte bir konuşmacı değişimi doğuruyordu.

    turns: (start, end, speaker) üçlüleri. Kelime listesini yerinde günceller.
    """
    if not turns:
        return words

    turns = sorted(turns, key=lambda t: t[0])
    mids = [((t[0] + t[1]) / 2.0, t[2]) for t in turns]

    previous = None
    for word in words:
        start, end = word[0], word[1]
        ranked = _rank_speakers(turns, start, end)

        if ranked:
            speaker = ranked[0][0]
            # Kıl payı fark: süreklilik kazansın.
            if previous is not None and WORD_TIE_MARGIN > 0:
                for other, score in ranked[1:]:
                    if other == previous and ranked[0][1] - score < WORD_TIE_MARGIN:
                        speaker = previous
                        break
        elif fill_nearest:
            mid = (start + end) / 2.0
            speaker = min(mids, key=lambda m: abs(m[0] - mid))[1]
        else:
            speaker = UNKNOWN

        word[2] = speaker
        if speaker != UNKNOWN:
            previous = speaker

    return words


def _rank_speakers(turns, start, end):
    """Konuşmacıları verilen aralıkla toplam kesişimlerine göre büyükten küçüğe sıralar."""
    totals = {}
    for t_start, t_end, speaker in turns:
        if t_start >= end:
            break
        gain = _overlap(start, end, t_start, t_end)
        if gain > 0:
            totals[speaker] = totals.get(speaker, 0.0) + gain
    return sorted(totals.items(), key=lambda kv: kv[1], reverse=True)


def drop_hallucinations(words):
    """
    HALLUCINATION_PATTERNS'e uyan cümle birimlerini tamamen atar.

    Sessiz bölümde uydurulan YouTube kalıpları hem transkripti kirletiyor hem de
    sessizliği "konuşma" diye bir konuşmacıya yazdırıp konuşma sürelerini şişiriyor.
    """
    if not HALLUCINATION_PATTERNS:
        return words

    patterns = [re.compile(p, re.IGNORECASE) for p in HALLUCINATION_PATTERNS]
    kept = []
    for unit in split_units(words):
        text = lower_tr(" ".join(w[3] for w in unit))
        if any(p.search(text) for p in patterns):
            print(f"  Halüsinasyon atıldı [{format_time(unit[0][0])}]: {' '.join(w[3] for w in unit)}")
            continue
        kept.extend(unit)
    return kept


def group_runs(words):
    """
    Ardışık aynı konuşmacıya ait kelimeleri tek bloğa toplar.

    Bloklar kelimelerin kendisini ('items') taşır; etiket düzeltmesi kelime
    üstünde yapılsın diye. Bloklar birleştirilmiş halde döndürülürse satırlama
    aşaması aradaki sessizliği göremez ve MERGE_GAP işlevsiz kalır.
    """
    runs = []
    for word in words:
        speaker = word[2]
        if runs and runs[-1]["speaker"] == speaker:
            runs[-1]["end"] = word[1]
            runs[-1]["items"].append(word)
        else:
            runs.append({"start": word[0], "end": word[1], "speaker": speaker, "items": [word]})
    return runs


def split_units(words):
    """
    Kelimeleri cümle benzeri birimlere böler: sonu noktalama ile biten kelimeden
    veya SENTENCE_GAP'ten uzun sessizlikten sonra yeni birim başlar.
    Birim listesini (kelime listeleri olarak) döndürür.
    """
    units = []
    current = []
    for i, word in enumerate(words):
        if current:
            gap = word[0] - current[-1][1]
            if gap > SENTENCE_GAP or SENTENCE_END_RE.search(current[-1][3]):
                units.append(current)
                current = []
        current.append(word)
    if current:
        units.append(current)
    return units


def snap_to_sentences(words):
    """
    Cümle içinde, arada duraklama olmadan beliren küçük azınlık bloklarını cümlenin
    baskın konuşmacısına devreder.

    Amaç, diarization sınırının kelime sınırına oturmamasından doğan "cümle ortasında
    konuşmacı değişti" hatası. Devir üç şarta birden bağlı: blok cümlenin küçük bir
    parçası (SNAP_MINORITY_RATIO), cümle içinde birden fazla konuşmacı var ve
    değişim noktasının İKİ yanında da SNAP_MIN_PAUSE'tan uzun bir duraklama yok.
    Duraklama varsa gerçekten söz alınmış olabilir — smooth_speakers'taki
    ADJACENT_GAP korumasının aynı mantığı.
    """
    if not SENTENCE_SNAP:
        return words

    for unit in split_units(words):
        runs = group_runs(unit)
        if len(runs) < 2:
            continue

        totals = {}
        for run in runs:
            totals[run["speaker"]] = totals.get(run["speaker"], 0.0) + (run["end"] - run["start"])
        unit_duration = sum(totals.values())
        if unit_duration <= 0:
            continue
        major = max(totals.items(), key=lambda kv: kv[1])[0]

        for i, run in enumerate(runs):
            if run["speaker"] == major:
                continue
            if (run["end"] - run["start"]) / unit_duration >= SNAP_MINORITY_RATIO:
                continue
            pause_before = run["start"] - runs[i - 1]["end"] if i > 0 else 0.0
            pause_after = runs[i + 1]["start"] - run["end"] if i + 1 < len(runs) else 0.0
            if max(pause_before, pause_after) >= SNAP_MIN_PAUSE:
                continue
            for word in run["items"]:
                word[2] = major

    return words


def smooth_speakers(words):
    """
    Tek kelimelik / çok kısa konuşmacı sıçramalarını temizler.

    pyannote, araya giren "hı hı", öksürük veya örtüşen konuşmada bir iki kelimeyi
    yanlış konuşmacıya yazabiliyor. İki yanı aynı konuşmacı olan kısa blokları
    o konuşmacıya devrediyoruz.

    Devretmenin şartı sadece kısalık değil, KOMŞULUK da: blok iki yanındaki
    konuşmaya ADJACENT_GAP'ten uzaksa bu bir etiket titremesi değil, sessizliğin
    ortasında gerçekten alınmış kısa bir sözdür — dokunmuyoruz. Aksi halde
    tam da yakalamak istediğimiz araya girmeleri siliyor oluruz.

    Kelime listesini (etiketleri düzeltilmiş olarak) döndürür; blokları
    birleştirmek satırlama aşamasının işi.
    """
    if MIN_TURN_DURATION <= 0:
        return words

    changed = True
    while changed:
        changed = False
        runs = group_runs(words)
        for i in range(1, len(runs) - 1):
            prev_run, run, next_run = runs[i - 1], runs[i], runs[i + 1]
            if prev_run["speaker"] != next_run["speaker"]:
                continue
            if (run["start"] - prev_run["end"] > ADJACENT_GAP
                    or next_run["start"] - run["end"] > ADJACENT_GAP):
                continue
            duration = run["end"] - run["start"]
            if run["speaker"] == UNKNOWN or duration < MIN_TURN_DURATION:
                for word in run["items"]:
                    word[2] = prev_run["speaker"]
                changed = True

    return words


def build_lines(words):
    """Konuşmacı değişimlerine göre satırlar; kısa nefes molalarını birleştirir."""
    lines = []
    for start, end, speaker, text in words:
        if lines and lines[-1][2] == speaker and (start or 0) - (lines[-1][1] or 0) <= MERGE_GAP:
            prev_start, _, prev_speaker, prev_text = lines[-1]
            lines[-1] = (prev_start, end, prev_speaker, f"{prev_text} {text}")
        else:
            lines.append((start, end, speaker, text))
    return lines


def write_lines(lines, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for start, end, speaker, text in lines:
            line = f"[{format_time(start)} --> {format_time(end)}] [{speaker}]  {text.strip()}\n"
            print(line, end="")
            f.write(line)
            f.flush()


# =========================================================
# 3. KAYDA ÖZEL AYARLAR (komut satırı + soru)
# =========================================================
def parse_audio(text):
    """
    Ses dosyası yolunu temizler ve varlığını denetler. Terminale sürükleyip
    bırakılan yol tırnaklı, 'file://' önekli ya da boşlukları '\\ ' ile kaçırılmış
    gelebiliyor.
    """
    path = text.strip().strip("'\"")
    if path.startswith("file://"):
        path = unquote(urlparse(path).path)
    path = os.path.expanduser(path.replace("\\ ", " "))
    if not path:
        raise ValueError("Dosya yolu boş.")
    if not os.path.isfile(path):
        raise ValueError(f"'{path}' bulunamadı.")
    return path


def parse_speakers(text):
    """
    Konuşmacı sayısını (num, min, max) üçlüsüne çevirir:
    '3' -> (3, None, None), '2-4' -> (None, 2, 4), '' ya da '?' -> (None, None, None).
    """
    text = text.strip()
    if text in ("", "?"):
        return None, None, None
    low, sep, high = text.partition("-")
    try:
        if not sep:
            num = int(text)
            if num >= 1:
                return num, None, None
        else:
            low, high = int(low), int(high)
            if 1 <= low <= high:
                return None, low, high
    except ValueError:
        pass
    raise ValueError(f"'{text}' anlaşılmadı: 3 gibi bir sayı, 2-4 gibi bir aralık ya da boş yaz.")


def parse_terms(text):
    """Virgülle ayrılmış isim/terim listesini temizler."""
    return [term.strip() for term in text.split(",") if term.strip()]


def describe_speakers(speakers):
    num, low, high = speakers
    if num:
        return str(num)
    if low:
        return f"{low}-{high}"
    return "bilinmiyor (pyannote tahmin edecek)"


def default_output(audio_path):
    """OUTPUT_DIR/<kayıt adı>.txt; varsa üzerine yazmak yerine -2, -3... eki."""
    name = os.path.splitext(os.path.basename(audio_path))[0]
    path = os.path.join(OUTPUT_DIR, f"{name}.txt")
    number = 2
    while os.path.exists(path):
        path = os.path.join(OUTPUT_DIR, f"{name}-{number}.txt")
        number += 1
    return path


def _arg_type(parse):
    """parse_* fonksiyonlarının hatasını argparse'ın kendi hata mesajına çevirir."""
    def convert(text):
        try:
            return parse(text)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error))
    return convert


def parse_args():
    parser = argparse.ArgumentParser(
        description="Türkçe kaydı yazıya döker ve her kelimeye konuşmacı etiketi verir. "
                    "Verilmeyen ayarlar çalışınca sorulur.",
    )
    parser.add_argument("ses", nargs="?", type=_arg_type(parse_audio),
                        help="ses ya da video dosyası")
    parser.add_argument("--konusmaci", type=_arg_type(parse_speakers),
                        help="konuşan kişi sayısı: 3, aralık 2-4 ya da ? (bilinmiyor)")
    parser.add_argument("--terimler", type=parse_terms,
                        help='kayıtta geçen isim ve terimler, virgülle: "Ayşe, DEHB, GitHub"')
    parser.add_argument("--cikti",
                        help=f"çıktı dosyası (varsayılan: {OUTPUT_DIR}/<kayıt adı>.txt)")
    return parser.parse_args()


def ask(prompt, parse):
    """Geçerli bir cevap gelene kadar sorar."""
    while True:
        try:
            return parse(input(prompt))
        except ValueError as error:
            print(f"  {error}")


def resolve_settings(args):
    """
    Kayda özel ayarları toplar; komut satırında verilmeyeni terminalde sorar.

    Terminal yoksa (başka bir programın içinden çalışırken) sormaz, çünkü input()
    orada takılı kalır: ses dosyası zorunludur, konuşmacı sayısı "bilinmiyor",
    terim listesi boş kabul edilir.
    """
    audio, speakers, terms = args.ses, args.konusmaci, args.terimler
    if sys.stdin.isatty():
        try:
            if audio is None:
                audio = ask("Ses dosyası (sürükleyip bırakabilirsin): ", parse_audio)
            if speakers is None:
                speakers = ask("Kaç kişi konuşuyor? (3 / aralık 2-4 / boş = bilinmiyor): ",
                               parse_speakers)
            if terms is None:
                terms = parse_terms(input("Kayıtta geçen isim ve terimler, virgülle (boş = yok): "))
        except (EOFError, KeyboardInterrupt):
            print("\nİptal edildi.")
            return None

    if audio is None:
        print("HATA: ses dosyası verilmedi. Kullanım: python main.py <kayıt> (ayrıntı: --help)")
        return None

    return {
        "audio": audio,
        "speakers": speakers or (None, None, None),
        "terms": terms or [],
        "output": args.cikti or default_output(audio),
    }


# =========================================================
# 4. ÇALIŞTIRMA
# =========================================================
def main():
    args = parse_args()
    load_env()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("HATA: HF_TOKEN bulunamadı.")
        print("  .env.example dosyasını .env olarak kopyala ve token'ı içine yaz:")
        print("      cp .env.example .env")
        print("  pyannote diarization modelleri için HF hesabından erişim onayı da gerekli.")
        return

    settings = resolve_settings(args)
    if settings is None:
        return
    num_speakers, min_speakers, max_speakers = settings["speakers"]
    output_file = settings["output"]
    print(f"Kayıt      : {settings['audio']}")
    print(f"Konuşmacı  : {describe_speakers(settings['speakers'])}")
    print(f"Terimler   : {', '.join(settings['terms']) or 'yok'}")
    print(f"Çıktı      : {output_file}")

    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ensure_model()

    audio_path = prepare_audio(settings["audio"])

    print("Whisper modeli yükleniyor...")
    asr_options = {
        "beam_size": BEAM_SIZE,
        "best_of": BEST_OF,
        "patience": PATIENCE,
        "repetition_penalty": REPETITION_PENALTY,
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "condition_on_previous_text": False,
        "compression_ratio_threshold": COMPRESSION_RATIO_THRESHOLD,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "suppress_numerals": SUPPRESS_NUMERALS,
        "initial_prompt": None,
        "hotwords": ", ".join(settings["terms"]),
    }

    model = whisperx.load_model(
        MODEL_PATH,
        DEVICE,
        compute_type=COMPUTE_TYPE,
        language=LANGUAGE,
        vad_options={"vad_onset": VAD_ONSET, "vad_offset": VAD_OFFSET},
        asr_options=asr_options,
    )

    print("Ses dosyası yükleniyor...")
    audio = whisperx.load_audio(audio_path)

    print("Transkript oluşturuluyor...")
    result = model.transcribe(audio, batch_size=BATCH_SIZE)
    del model
    free_vram()

    print("Kelime bazlı hizalama yapılıyor...")
    model_a, metadata = whisperx.load_align_model(language_code=LANGUAGE, device=DEVICE)
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False
    )
    del model_a
    free_vram()

    print("Konuşmacı ayrımı yapılıyor...")
    diarize_model = whisperx.diarize.DiarizationPipeline(token=hf_token, device=DEVICE)
    diarize_segments = diarize_model(
        audio,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    del diarize_model
    free_vram()

    turns = [
        (row.start, row.end, row.speaker)
        for row in diarize_segments.itertuples()
    ]
    print(f"  {len(turns)} tur, {len({t[2] for t in turns})} konuşmacı bulundu.")

    print(f"Sonuçlar '{output_file}' dosyasına yazılıyor...")
    words = flatten_words(result)
    words = strip_repetitions(words)
    words = drop_hallucinations(words)
    words = assign_speakers(turns, words)
    words = snap_to_sentences(words)
    words = smooth_speakers(words)
    write_lines(build_lines(words), output_file)

    print("\n--- İşlem başarıyla tamamlandı ---")


if __name__ == "__main__":
    main()

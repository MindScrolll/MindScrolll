

import re
from zemberek import TurkishMorphology

print(">> Zemberek yükleniyor (bir kereye mahsus, birkaç saniye sürer)...")
morphology = TurkishMorphology.create_with_defaults()


kok_sozlugu = {}

def kelime_koku_bul(kelime: str) -> str:
    """Tek bir kelimenin kökünü Zemberek ile bulur, önbellekten okur/yazar."""
    if kelime in kok_sozlugu:
        return kok_sozlugu[kelime]

    try:
        analiz = morphology.analyze(kelime)
        if analiz.analysis_results:
            kok = str(analiz.analysis_results[0].get_stem())
        else:
            # Zemberek kökünü bulamadıysa (argo, kısaltma, yazım hatası
            # gibi durumlarda sık olur), kelimeyi olduğu gibi bırakıyoruz.
            # Bunu yapmazsak veri kaybederiz, modelin göreceği bilgi azalır.
            kok = kelime
    except Exception:
        kok = kelime

    kok_sozlugu[kelime] = kok
    return kok


def temizle_ve_kokle(metin: str) -> str:
    """
    Bir tweet metnini tam pipeline'dan geçirir:
    1. Küçük harfe çevir, URL/mention/hashtag/noktalama temizle
    2. Kelimelere ayır
    3. Her kelimenin Zemberek ile kökünü bul
    4. Kökleri tekrar birleştirip tek bir temiz metin döndür
    """
    # --- Adım 1: temel temizlik ---
    metin = str(metin).lower()
    metin = re.sub(r"http\S+|www\S+", "", metin)
    metin = re.sub(r"@\w+", "", metin)
    metin = re.sub(r"#", "", metin)
    metin = re.sub(r"[^\wçğıöşüÇĞİÖŞÜ\s]", "", metin)
    metin = re.sub(r"\s+", " ", metin).strip()

    # --- Adım 2-3-4: kelime kelime köklendirme ---
    kelimeler = metin.split()
    kokler = [kelime_koku_bul(k) for k in kelimeler]
    return " ".join(kokler)


if __name__ == "__main__":
    # Hızlı bir test
    ornek = "allahım çıldıracağım yemin ederim çıldıracağım sinirimden yerden çıkaramıyorum"
    print("Önce :", ornek)
    print("Sonra:", temizle_ve_kokle(ornek))
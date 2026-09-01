
from zemberek import TurkishMorphology

print(">> Zemberek yükleniyor...")
morphology = TurkishMorphology.create_with_defaults()

test_cumle = "allahım çıldıracağım yemin ederim çıldıracağım sinirimden yerden çıkaramıyorum"

# Cümleyi kelimelere ayır
kelimeler = test_cumle.split()

print(f"\n>> Cümle {len(kelimeler)} kelimeye ayrıldı: {kelimeler}\n")

print(">> Her kelimenin kökü analiz ediliyor:")
for kelime in kelimeler:
    analiz = morphology.analyze(kelime)
    if analiz.analysis_results:  # eğer analiz sonucu bulunduysa
        kok = analiz.analysis_results[0].get_stem()
        print(f"  '{kelime}' -> kök: '{kok}'")
    else:
        print(f"  '{kelime}' -> analiz edilemedi (muhtemelen argo/kısaltma/yazım hatası)")
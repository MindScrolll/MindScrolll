

import pandas as pd
from on_isleme import temizle_ve_kokle

url = "https://huggingface.co/datasets/anilguven/turkish_tweet_emotion_dataset/resolve/main/Turkish_Tweet_Dataset.csv"

print(">> Veri seti indiriliyor...")
df = pd.read_csv(
    url,
    engine="python",
    on_bad_lines="skip",
    header=None,
    names=["text", "label"]
)
print(f">> {len(df)} satır yüklendi.")

print(">> Zemberek ile ön işleme başlıyor, bu biraz sürebilir, lütfen bekle...")
print(">> (İlerlemeyi görebilmen için her 500 satırda bir bilgi vereceğim)")

islenmis_metinler = []
for i, metin in enumerate(df["text"]):
    islenmis_metinler.append(temizle_ve_kokle(metin))
    if (i + 1) % 500 == 0:
        print(f"   ... {i + 1}/{len(df)} tweet işlendi")

df["islenmis_text"] = islenmis_metinler

print(">> Ön işleme tamamlandı. Örnek karşılaştırma:")
print(df[["text", "islenmis_text", "label"]].head())

# Sonucu kaydet, bir daha bu işlemi tekrarlamamıza gerek kalmasın
df.to_csv("islenmis_veri.csv", index=False, encoding="utf-8-sig")
print("\n>> Kaydedildi: islenmis_veri.csv")
print(f">> Zemberek'in bulduğu toplam benzersiz kelime sayısı (sözlük boyutu): {len(kok_sozlugu := __import__('on_isleme').kok_sozlugu)}")
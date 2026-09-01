import pandas as pd

url = "https://huggingface.co/datasets/anilguven/turkish_tweet_emotion_dataset/resolve/main/Turkish_Tweet_Dataset.csv"

df = pd.read_csv(
    url,
    engine="python",
    on_bad_lines="skip",   # bozuk/kaçışlanmamış satırları atla, script çökmesin
    header=None,           # dosyada gerçek başlık satırı yok
    names=["text", "label"]  # sütunlara kendimiz isim veriyoruz
)

print("=" * 50)
print("Sütun isimleri:", df.columns.tolist())
print("=" * 50)
print()

print("Kaç satır yüklendi:", len(df))
print("(Rapor 4000 diyor; 4000'e çok yakın çıkması bekleniyor)")
print()

print("İlk 5 satır:")
print(df.head())
print()

print("Sınıf dağılımı (etiket bazında kaç tweet var):")
print(df["label"].value_counts())
print()

print("Veri setindeki BENZERSİZ (tekil) etiketler:")
print(df["label"].unique())
print()

print("Örnek metinler (her etiketten bir tane göstermeye çalışalım):")
for etiket in df["label"].unique():
    ornek = df[df["label"] == etiket].iloc[0]["text"]
    print(f"[{etiket}] {ornek}")
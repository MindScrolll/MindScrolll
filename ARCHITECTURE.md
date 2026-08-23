# MindScroll — Edge AI Mimarisi ve Model Tasarım Kararları

**İş Paketi:** İP1 — Teknik Rapor ve UI/UX Prototipleme
**Sorumlu:** Kaptan / AI Uzmanı
**Tarih:** 24 Ağustos 2026

Bu doküman, İP1 kapsamında netleştirilen Edge AI mimarisini ve model
tasarım kararlarını özetler.

## 1. Neden Edge AI (cihaz üzerinde çıkarım)?

- Kullanıcı verisi (sosyal medya akış metinleri, kaydırma davranışı)
  hiçbir zaman sunucuya ham olarak gönderilmez; model çıkarımı
  doğrudan mobil cihaz üzerinde (TFLite) çalışır.
- Bu, hem KVKK/mahremiyet açısından hem de gecikme (latency) açısından
  bulut tabanlı API çağrısına göre üstündür (bkz. rapor, Kaynak [12]).
- Sunucu tarafı (`backend/`) yalnızca anonimleştirilmiş "Duygu Ayak
  İzi" özet metriklerini (negatiflik skoru, kaydırma süresi) alır —
  ham metin asla backend'e gitmez.

## 2. Model mimarisi kararı: neden hibrit CNN-LSTM?

| Katman | Görevi |
|---|---|
| Embedding | Kelimeleri yoğun vektör uzayına taşır |
| Conv1D + MaxPooling1D | Yerel öznitelikleri / n-gram ilişkilerini hızlı filtreler |
| Bidirectional LSTM | Ardışık paylaşımlar arasındaki zamansal/bağlamsal duygu dizilimini yakalar |
| Dense (softmax, 5 sınıf) | Nihai duygu sınıfını üretir: öfke, korku, mutluluk, üzüntü, şaşkınlık |

**Düzeltme notu:** İlk taslak (`model/cnn_lstm_model.py`, önceki commit)
tek nöronlu `sigmoid` çıktı kullanıyordu — bu, ikili sınıflandırmaya
uygundur ama seçtiğimiz veri seti (`anilguven/turkish_tweet_emotion_dataset`)
5 sınıflıdır. Mimari, `Dense(5, activation='softmax')` ve
`categorical_crossentropy` kaybına güncellenerek rapor Bölüm 3.2 ile
tutarlı hale getirildi.

## 3. Türkçe'ye özgü ön işleme: Zemberek

Türkçe sondan eklemeli bir dil olduğu için ham tokenizasyon yetersiz
kalır. Ön işleme (lemmatizasyon) sunucu/eğitim tarafında **Zemberek**
kütüphanesi ile yapılacak; üretilen hafifletilmiş tokenizer/sözlük
yapısı mobil tarafa (TFLite ile birlikte) aktarılacak. Böylece
Zemberek'in kendisi (Java tabanlı, ağır) cihaza gömülmez — sadece
çıktısı gömülür. Bu karar, mobil performans (Edge AI) hedefiyle
doğrudan ilgilidir.

## 4. Sonraki adım (İP2)

- `anilguven/turkish_tweet_emotion_dataset` indirilip Zemberek ile
  ön işlenecek.
- Model eğitilecek, K-Fold çapraz doğrulama ve dropout ile overfitting
  önlemleri uygulanacak.
- Eğitilen model `convert_to_tflite()` ile TFLite formatına
  dönüştürülüp `mobile/` tarafına entegre edilecek.

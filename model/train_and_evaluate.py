

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from cnn_lstm_model import build_cnn_lstm_model, convert_to_tflite, VOCAB_SIZE, MAX_LENGTH


print(">> İşlenmiş veri okunuyor...")
df = pd.read_csv("islenmis_veri.csv")
print(f">> {len(df)} satır yüklendi.")
print(df[["text", "islenmis_text", "label"]].head())

print(f">> Temizlik öncesi satır sayısı: {len(df)}")
bos_satir_sayisi = df["islenmis_text"].isna().sum()
print(f">> Boş/eksik 'islenmis_text' satırı: {bos_satir_sayisi}")

df = df.dropna(subset=["islenmis_text", "label"])
df = df[df["islenmis_text"].astype(str).str.strip() != ""]

print(f">> Temizlik sonrası satır sayısı: {len(df)}")



etiket_sirasi = {
    "kizgin": 0,   # öfke
    "korku": 1,    # korku
    "mutlu": 2,    # mutluluk
    "uzgun": 3,    # üzüntü
    "surpriz": 4,  # şaşkınlık
}
DUYGU_ISIMLERI = ["öfke", "korku", "mutluluk", "üzüntü", "şaşkınlık"]

y_raw = df["label"].map(etiket_sirasi).values


if pd.isna(y_raw).any():
    print("!! UYARI: Bazı etiketler eşleştirilemedi, veri setinde beklenmeyen bir etiket olabilir.")
    print(df[df["label"].map(etiket_sirasi).isna()]["label"].unique())

y = to_categorical(y_raw, num_classes=5)


print(f"\n>> Tokenizer eğitiliyor (VOCAB_SIZE={VOCAB_SIZE}, MAX_LENGTH={MAX_LENGTH})...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df["islenmis_text"].astype(str))

sequences = tokenizer.texts_to_sequences(df["islenmis_text"].astype(str))
X = pad_sequences(sequences, maxlen=MAX_LENGTH, padding="post", truncating="post")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_raw
)
print(f">> Eğitim seti: {len(X_train)} örnek, Test seti: {len(X_test)} örnek")


print("\n>> K-Fold çapraz doğrulama başlıyor (5 kat)...")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

best_model = None
best_val_acc = 0.0

for fold_no, (train_idx, val_idx) in enumerate(kfold.split(X_train), start=1):
    print(f"\n--- Fold {fold_no}/5 ---")
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = build_cnn_lstm_model()

   
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    val_acc = max(history.history["val_accuracy"])
    fold_accuracies.append(val_acc)
    print(f">> Fold {fold_no} doğrulama accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = model

print(f"\n>> K-Fold ortalama accuracy: {np.mean(fold_accuracies):.4f} (+-{np.std(fold_accuracies):.4f})")


print("\n>> Test seti üzerinde final değerlendirme...")
y_pred_probs = best_model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

test_accuracy = accuracy_score(y_true, y_pred)
test_f1 = f1_score(y_true, y_pred, average="weighted")

print(f"\n{'='*50}")
print(f"GERÇEK TEST SONUÇLARI (raporda kullanacağın sayılar bunlar)")
print(f"{'='*50}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1 Score (weighted): {test_f1:.4f}")
print(f"\nDetaylı sınıf bazlı rapor:")
print(classification_report(y_true, y_pred, target_names=DUYGU_ISIMLERI))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=DUYGU_ISIMLERI, yticklabels=DUYGU_ISIMLERI)
plt.title("MindScroll - Duygu Sınıflandırma Confusion Matrix")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print(">> confusion_matrix.png kaydedildi.")


print("\n>> Model TFLite formatına dönüştürülüyor...")
tflite_path = convert_to_tflite(best_model, "mindscroll_model.tflite")

import os
model_size_kb = os.path.getsize(tflite_path) / 1024
print(f">> TFLite model boyutu: {model_size_kb:.2f} KB ({model_size_kb/1024:.3f} MB)")
print(">> Raporundaki '~1-2 MB' hedefiyle bu gerçek sayıyı karşılaştır!")



with open("model_sonuclari.txt", "w", encoding="utf-8") as f:
    f.write("MindScroll Model Eğitim Sonuçları\n")
    f.write("=" * 40 + "\n")
    f.write(f"Kullanılan veri: anilguven/turkish_tweet_emotion_dataset (4000 tweet, 5 sınıf, dengeli)\n")
    f.write(f"Ön işleme: Zemberek ile kelime köklendirme (sözlük boyutu: {len(pd.read_csv('islenmis_veri.csv'))} satır işlendi)\n")
    f.write(f"K-Fold ortalama accuracy: {np.mean(fold_accuracies):.4f} (+-{np.std(fold_accuracies):.4f})\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test F1 Score (weighted): {test_f1:.4f}\n")
    f.write(f"TFLite model boyutu: {model_size_kb:.2f} KB\n")

print("\n>> Tüm sonuçlar model_sonuclari.txt dosyasına kaydedildi.")
print(">> ARTIK RAPORDA KULLANACAĞIN GERÇEK SAYILAR ELİNDE!")
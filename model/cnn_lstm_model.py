"""
MindScroll: Hibrit CNN-LSTM Duygu Analizi Modeli
Kaptan / AI Uzmanı katkısı - İP1: Edge AI mimarisinin netleştirilmesi
ve model mimarisi tasarımı.

Metin içerisindeki yerel özellikler CNN ile, zamansal/bağlamsal dizilim
LSTM ile analiz edilir. Çıktı katmanı, rapordaki 5 sınıflı duygu
etiketlemesine (öfke, korku, mutluluk, üzüntü, şaşkınlık) uygun olacak
şekilde softmax olarak tasarlanmıştır (önceki sürümdeki ikili sigmoid
çıktı, veri setiyle [anilguven/turkish_tweet_emotion_dataset] uyumsuzdu).
"""

import tensorflow as tf

# Rapor Tablo 2 / Bölüm 3.2 ile birebir uyumlu sabitler
NUM_CLASSES = 5  # öfke, korku, mutluluk, üzüntü, şaşkınlık
VOCAB_SIZE = 10000
MAX_LENGTH = 100
EMBEDDING_DIM = 64


def build_cnn_lstm_model(
    vocab_size: int = VOCAB_SIZE,
    max_length: int = MAX_LENGTH,
    num_classes: int = NUM_CLASSES,
) -> tf.keras.Model:
    """
    MindScroll: Hibrit CNN-LSTM Duygu Analizi Modeli

    - Embedding: kelime gömme
    - Conv1D + MaxPooling1D: yerel öznitelik / n-gram çıkarımı (CNN katmanı)
    - Bidirectional LSTM: ardışık/bağlamsal duygu dizilimini analiz eder
    - Dropout: aşırı öğrenmeyi (overfitting) önlemek için
    - Dense(num_classes, softmax): çok sınıflı duygu çıktısı
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_length
        ),

        # --- CNN katmanı: yerel özellik / n-gram çıkarımı ---
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.3),

        # --- LSTM katmanı: ardışık/bağlamsal duygu dizilimi ---
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)
        ),

        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),

        # --- Çıktı katmanı: 5 sınıflı duygu sınıflandırması ---
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"],
    )
    return model


def convert_to_tflite(keras_model: tf.keras.Model, output_path: str = "mindscroll_model.tflite") -> str:
    """Eğitilen modeli Edge AI (cihaz üzerinde) çalışacak TFLite formatına dönüştürür."""
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # boyut/nicemleme optimizasyonu
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    return output_path


if __name__ == "__main__":
    print("CNN-LSTM hibrit model (5 sınıflı, Edge AI mimarisi) oluşturuldu.")
    model = build_cnn_lstm_model()
    model.summary()
    print("Model, Edge AI için TFLite formatına dönüştürülmeye hazır.")

    # TODO (İP2): Zemberek ile ön işlenmiş anilguven/turkish_tweet_emotion_dataset
    # buraya bağlanacak; model.fit(X_train, y_train, validation_split=0.2, ...) çağrılacak.

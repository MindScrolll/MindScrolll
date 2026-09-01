

import tensorflow as tf


NUM_CLASSES = 5  # öfke, korku, mutluluk, üzüntü, şaşkınlık
VOCAB_SIZE = 10000
MAX_LENGTH = 100
EMBEDDING_DIM = 64


def build_cnn_lstm_model(
    vocab_size: int = VOCAB_SIZE,
    max_length: int = MAX_LENGTH,
    num_classes: int = NUM_CLASSES,
) -> tf.keras.Model:
   
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
    
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # boyut/nicemleme optimizasyonu

    # LSTM katmanının TFLite'a düzgün aktarılabilmesi için gerekli ayarlar
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,   # standart, hafif TFLite operasyonları
        tf.lite.OpsSet.SELECT_TF_OPS,     # LSTM gibi karmaşık katmanlar için TF operasyonları
    ]
    converter._experimental_lower_tensor_list_ops = False

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

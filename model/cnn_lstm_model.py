import tensorflow as tf

def build_cnn_lstm_model(vocab_size=10000, max_length=100):
    """
    MindScroll: Hibrit CNN-LSTM Duygu Analizi Modeli
    Metin içerisindeki yerel özellikleri CNN ile, zamansal/bağlamsal dizilimi LSTM ile analiz eder.
    """
    model = tf.keras.Sequential([
        # Kelime gömme (Embedding) katmanı
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length),
        
        # CNN Katmanı: Yerel özellikleri ve n-gram ilişkilerini filtreler
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        # LSTM Katmanı: Ardışık duygu bağlamını analiz eder
        tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        
        # Çıktı katmanı: İkili (negatif/pozitif) veya çoklu duygu sınıflandırması
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("CNN-LSTM hibrit model iskeleti oluşturuldu.")
    model = build_cnn_lstm_model()
    model.summary()
    print("Model, Edge AI için TFLite formatına dönüştürülmeye hazır.")
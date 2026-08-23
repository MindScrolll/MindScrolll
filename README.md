# MindScroll - Dijital Refah Asistanı 🚀

MindScroll, "doomscrolling" problemine karşı yapay zekâ tabanlı, proaktif bir müdahale mekanizması sunan dijital esenlik ekosistemidir. TEKNOFEST 2026 NSosyal İnovasyon Yarışması kapsamında geliştirilmektedir.

## 🛠️ Teknolojik Altyapı

* **Mobil İstemci (Frontend):** Flutter (Dart)
* **Arka Plan Servisleri (Backend):** FastAPI (Python), PostgreSQL
* **Yapay Zeka & Veri Bilimi:** CNN-LSTM Hibrit Duygu Analizi Modeli (TensorFlow Lite, Edge AI)

## 📂 Proje Mimarisi ve Kurulum

Projemiz, çevik (agile) geliştirme prensiplerine uygun olarak üç ana modüle ayrılmıştır:

### 1. Backend (Arka Plan Servisleri)
FastAPI ve PostgreSQL altyapısını içerir.
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
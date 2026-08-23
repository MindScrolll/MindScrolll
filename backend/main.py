from fastapi import FastAPI

app = FastAPI(title="MindScroll API", description="Dijital Esenlik ve Duygu Analizi Servisi", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "MindScroll Backend Servisi Çalışıyor 🚀"}
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from database import Base

class DuyguAyakIzi(Base):
    __tablename__ = "duygu_ayak_izi"

    id = Column(Integer, primary_key=True, index=True)
    kullanici_id = Column(String, index=True)
    icerik_tipi = Column(String) # Metin, Görsel, vs.
    negatiflik_skoru = Column(Float) # 0.0 ile 1.0 arası
    maruz_kalinma_suresi = Column(Integer) # Saniye cinsinden
    olusturulma_tarihi = Column(DateTime(timezone=True), server_default=func.now())
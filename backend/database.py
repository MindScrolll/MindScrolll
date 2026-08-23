from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# PostgreSQL bağlantı dizesi (Geliştirme aşaması için varsayılan değerler)
SQLALCHEMY_DATABASE_URL = "postgresql://kullanici:sifre@localhost/mindscroll_db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Veritabanı oturumu (session) oluşturucu
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
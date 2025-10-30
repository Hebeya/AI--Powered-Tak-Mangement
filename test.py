from sqlalchemy import create_engine

try:
    engine = create_engine("mysql+pymysql://root:HS%402223@localhost:3306/task_management")
    with engine.connect() as conn:
        print("✅ Database connection successful!")
except Exception as e:
    print("❌ Error:", e)

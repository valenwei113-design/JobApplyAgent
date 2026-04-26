import os
import pandas as pd
import psycopg2

# 连接 Postgres（先连默认 postgres database）
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    user="postgres",
    password="difyai123456",
    database="postgres"
)
conn.autocommit = True
cur = conn.cursor()

# 创建新 database
cur.execute("SELECT 1 FROM pg_database WHERE datname='jobsdb'")
if not cur.fetchone():
    cur.execute("CREATE DATABASE jobsdb")
    print("Created database: jobsdb")
else:
    print("Database jobsdb already exists")

cur.close()
conn.close()

# 连接 jobsdb
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    user="postgres",
    password="difyai123456",
    database="jobsdb"
)
cur = conn.cursor()

# ── 表1：job_applications ──
cur.execute("""
    CREATE TABLE IF NOT EXISTS job_applications (
        id SERIAL PRIMARY KEY,
        company TEXT,
        position TEXT,
        applied_date TEXT,
        location TEXT,
        link TEXT,
        feedback TEXT
    )
""")

df1 = pd.read_csv(os.path.expanduser("~/Desktop/Job Track Agent/job_applications.csv"))
df1.columns = ['company', 'position', 'applied_date', 'location', 'link', 'feedback']

cur.execute("DELETE FROM job_applications")  # 避免重复导入
for _, row in df1.iterrows():
    cur.execute("""
        INSERT INTO job_applications (company, position, applied_date, location, link, feedback)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, tuple(row))

print(f"job_applications: 导入 {len(df1)} 条")

# ── 表2：work_permits ──
cur.execute("""
    CREATE TABLE IF NOT EXISTS work_permits (
        id SERIAL PRIMARY KEY,
        country TEXT,
        visa TEXT,
        annual_salary TEXT,
        permanent_residence TEXT
    )
""")

df2 = pd.read_csv(os.path.expanduser("~/Desktop/Job Track Agent/work_permits.csv"), usecols=[0,1,2,4])
df2.columns = ['country', 'visa', 'annual_salary', 'permanent_residence']

cur.execute("DELETE FROM work_permits")
for _, row in df2.iterrows():
    cur.execute("""
        INSERT INTO work_permits (country, visa, annual_salary, permanent_residence)
        VALUES (%s, %s, %s, %s)
    """, tuple(row))

print(f"work_permits: 导入 {len(df2)} 条")

conn.commit()
cur.close()
conn.close()
print("全部完成！")

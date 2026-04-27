from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import psycopg2
import psycopg2.extras
import re
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from openai import OpenAI
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

SQL_SYSTEM_PROMPT = """你是一个求职数据分析助手，专门帮助用户查询和分析他们的求职申请记录数据库。

## 职责范围
你只能回答与以下内容相关的问题：
- 用户的求职申请记录（公司、职位、申请日期、地点、工作类型、反馈结果）
- 数据统计与分析（投递数量、地点分布、通过率、时间趋势等）
- 工作许可信息（work_permits 表：国家、签证类型、薪资门槛、永居年限）

## 拒绝规则
如果用户的问题与上述职责范围无关（例如：写代码、翻译、天气、通用知识、聊天等），
必须严格执行以下要求：
- 不得生成任何 SQL 语句
- 不得调用自身知识直接回答问题
- 不得尝试用任何方式帮助用户完成与求职数据无关的请求
- 只输出以下固定回复，不做任何补充：
"抱歉，我只能帮你分析求职申请数据。请提问与你的投递记录相关的问题，例如：'我投了多少家公司？' 或 '哪个地点投递最多？'"

## 数据库结构

表1：job_applications
- id：主键
- company：公司名称
- position：职位名称
- applied_date：投递日期（DATE 类型，格式 YYYY-MM-DD，年份为 2026）
- location：国家/地区（如 "Norway"、"Netherlands"）
- link：职位链接
- feedback：反馈结果（NULL=待回复，"Fail"=拒绝，"Offer"=录用，"Interview"=面试，"Online Assessment"=线上笔试）
- work_type：工作类型（Remote / Onsite / Hybrid）
- user_id：用户 ID

表2：work_permits
- country：国家
- visa：签证/工作许可类型
- annual_salary：年薪门槛（文本）
- permanent_residence：永居申请年限

## 字段映射
- "地点" / "location" / "country" → job_applications 表的 location 字段
- "没有反馈" / "pending" / "待回复" → feedback IS NULL

## SQL 生成规则
确认问题与求职数据相关后，生成标准 PostgreSQL SELECT 语句：
- 只生成 SELECT 语句，不生成 INSERT / UPDATE / DELETE
- 查询 job_applications 时必须加上 WHERE user_id = {user_id}
- 涉及 location 字段时，必须附加过滤条件：location IS NOT NULL AND location != '' AND location != 'NaN'
- 涉及"最多/最少/前N名/排名"等问题时，必须使用 GROUP BY + ORDER BY + LIMIT
- 只输出原始 SQL 语句本身，不加任何解释、不加 markdown、不加代码块"""

EXPLAIN_SYSTEM_PROMPT = """你是一个求职数据分析助手，根据数据库查询结果用自然语言回答用户的问题。

要求：
- 用中文回答
- 语言简洁清晰，直接给出结论
- 如果数据为空，告知用户暂无相关记录
- 不要重复展示原始数据，用自然语言总结
- 回答控制在3-5句话以内"""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "difyai123456",
    "database": "jobsdb"
}

SECRET_KEY = os.environ["SECRET_KEY"]
ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer = HTTPBearer()

BLOCKED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE
)

# ── Models ──

class QueryRequest(BaseModel):
    sql: str

class ApplicationRequest(BaseModel):
    company: str
    position: str
    applied_date: str | None = None
    location: str | None = None
    link: str | None = None
    feedback: str | None = None
    work_type: str | None = None

class AuthRequest(BaseModel):
    email: str
    password: str

class ResetPasswordRequest(BaseModel):
    new_password: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []

# ── Auth helpers ──

def get_db():
    return psycopg2.connect(**DB_CONFIG)

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)

def create_token(user_id: int, is_admin: bool = False) -> str:
    payload = {
        "sub": str(user_id),
        "adm": is_admin,
        "exp": datetime.utcnow() + timedelta(days=TOKEN_EXPIRE_DAYS)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer)) -> int:
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

def get_admin_user(credentials: HTTPAuthorizationCredentials = Depends(bearer)) -> int:
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        if not payload.get("adm"):
            raise HTTPException(status_code=403, detail="Admin access required")
        return int(payload["sub"])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ── Auth endpoints ──

@app.post("/auth/register")
def register(req: AuthRequest):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM users WHERE email=%s", (req.email,))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered")
        cur.execute(
            "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id, is_admin",
            (req.email, hash_password(req.password))
        )
        row = cur.fetchone()
        conn.commit()
        return {"token": create_token(row[0], row[1]), "email": req.email, "is_admin": row[1]}
    finally:
        cur.close(); conn.close()

@app.post("/auth/login")
def login(req: AuthRequest):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, password_hash, is_admin FROM users WHERE email=%s", (req.email,))
        row = cur.fetchone()
        if not row or not verify_password(req.password, row[1]):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        return {"token": create_token(row[0], row[2]), "email": req.email, "is_admin": row[2]}
    finally:
        cur.close(); conn.close()

# ── Application endpoints (auth required) ──

@app.get("/applications")
def get_applications(user_id: int = Depends(get_current_user)):
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT id, company, position, applied_date, location, link, feedback, work_type
        FROM job_applications
        WHERE user_id=%s
        ORDER BY applied_date DESC NULLS LAST, id DESC
    """, (user_id,))
    rows = [dict(r) for r in cur.fetchall()]
    cur.close(); conn.close()
    for r in rows:
        if r['applied_date']:
            r['applied_date'] = r['applied_date'].isoformat()
    return rows

@app.post("/applications")
def add_application(req: ApplicationRequest, user_id: int = Depends(get_current_user)):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO job_applications (company, position, applied_date, location, link, feedback, work_type, user_id)
            VALUES (%s, %s, %s::date, %s, %s, %s, %s, %s)
        """, (req.company, req.position, req.applied_date or None,
              req.location, req.link, req.feedback, req.work_type, user_id))
        conn.commit()
        cur.close(); conn.close()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/applications/{app_id}")
def update_application(app_id: int, req: ApplicationRequest, user_id: int = Depends(get_current_user)):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            UPDATE job_applications
            SET company=%s, position=%s, applied_date=%s::date,
                location=%s, link=%s, feedback=%s, work_type=%s
            WHERE id=%s AND user_id=%s
        """, (req.company, req.position, req.applied_date or None,
              req.location, req.link, req.feedback, req.work_type, app_id, user_id))
        conn.commit()
        cur.close(); conn.close()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Stats endpoints (auth required) ──

@app.get("/stats/summary")
def stats_summary(user_id: int = Depends(get_current_user)):
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE feedback IS NULL) as pending,
            COUNT(DISTINCT location) as countries
        FROM job_applications WHERE user_id=%s
    """, (user_id,))
    row = dict(cur.fetchone())
    cur.close(); conn.close()
    return row

@app.get("/stats/countries")
def stats_countries(user_id: int = Depends(get_current_user)):
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT location, COUNT(*) as count
        FROM job_applications
        WHERE user_id=%s AND location IS NOT NULL AND location != 'NaN'
        GROUP BY location ORDER BY count DESC LIMIT 5
    """, (user_id,))
    rows = [dict(r) for r in cur.fetchall()]
    cur.close(); conn.close()
    return rows

@app.get("/stats/worktype")
def stats_worktype(user_id: int = Depends(get_current_user)):
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE work_type = 'Remote') as remote,
            COUNT(*) FILTER (WHERE work_type = 'Onsite') as onsite
        FROM job_applications WHERE user_id=%s
    """, (user_id,))
    row = dict(cur.fetchone())
    cur.close(); conn.close()
    return row

# ── Admin endpoints ──

@app.get("/admin/users")
def admin_list_users(admin_id: int = Depends(get_admin_user)):
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT u.id, u.email, u.is_admin, u.created_at,
               COUNT(a.id) as app_count
        FROM users u
        LEFT JOIN job_applications a ON a.user_id = u.id
        GROUP BY u.id, u.email, u.is_admin, u.created_at
        ORDER BY u.id
    """)
    rows = [dict(r) for r in cur.fetchall()]
    cur.close(); conn.close()
    for r in rows:
        if r['created_at']:
            r['created_at'] = r['created_at'].isoformat()
    return rows

@app.delete("/admin/users/{uid}")
def admin_delete_user(uid: int, admin_id: int = Depends(get_admin_user)):
    if uid == admin_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM job_applications WHERE user_id=%s", (uid,))
        cur.execute("DELETE FROM users WHERE id=%s", (uid,))
        conn.commit()
        return {"success": True}
    finally:
        cur.close(); conn.close()

@app.patch("/admin/users/{uid}/toggle-admin")
def admin_toggle_admin(uid: int, admin_id: int = Depends(get_admin_user)):
    if uid == admin_id:
        raise HTTPException(status_code=400, detail="Cannot change your own admin status")
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("UPDATE users SET is_admin = NOT is_admin WHERE id=%s RETURNING is_admin", (uid,))
        new_status = cur.fetchone()[0]
        conn.commit()
        return {"is_admin": new_status}
    finally:
        cur.close(); conn.close()

@app.patch("/admin/users/{uid}/reset-password")
def admin_reset_password(uid: int, req: ResetPasswordRequest, admin_id: int = Depends(get_admin_user)):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("UPDATE users SET password_hash=%s WHERE id=%s", (hash_password(req.new_password), uid))
        conn.commit()
        return {"success": True}
    finally:
        cur.close(); conn.close()

# ── Chat endpoint ──

@app.post("/chat")
def chat(req: ChatRequest, user_id: int = Depends(get_current_user)):
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    # Step 1: NL → SQL
    system_prompt = SQL_SYSTEM_PROMPT.replace("{user_id}", str(user_id))
    history = [{"role": m.role, "content": m.content} for m in (req.history or [])]
    messages = [
        {"role": "system", "content": system_prompt},
        *history[-6:],
        {"role": "user", "content": req.message}
    ]
    sql_resp = client.chat.completions.create(
        model="deepseek-chat", messages=messages, temperature=0
    )
    sql_or_reject = sql_resp.choices[0].message.content.strip()

    # 拒绝语直接返回
    if not sql_or_reject.upper().lstrip().startswith("SELECT"):
        return {"answer": sql_or_reject, "sql": None}

    # Step 2: 执行 SQL（安全检查）
    if BLOCKED.search(sql_or_reject):
        return {"answer": "生成的查询包含不允许的操作，已拦截。", "sql": None}
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql_or_reject)
        rows = [dict(r) for r in cur.fetchall()]
        cur.close(); conn.close()
        for r in rows:
            for k, v in r.items():
                if hasattr(v, 'isoformat'):
                    r[k] = v.isoformat()
    except Exception as e:
        return {"answer": f"查询出错：{str(e)}", "sql": sql_or_reject}

    # Step 3: 结果 → 自然语言
    explain_messages = [
        {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
        {"role": "user", "content": f"问题：{req.message}\n\n查询结果：{rows}"}
    ]
    explain_resp = client.chat.completions.create(
        model="deepseek-chat", messages=explain_messages, temperature=0.3
    )
    answer = explain_resp.choices[0].message.content.strip()
    return {"answer": answer, "sql": sql_or_reject}

# ── Public endpoints ──

@app.post("/query")
async def run_query(req: QueryRequest):
    if BLOCKED.search(req.sql):
        raise HTTPException(status_code=400, detail="Only SELECT queries are allowed.")
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(req.sql)
        rows = cur.fetchall()
        cur.close(); conn.close()
        return {"result": [dict(r) for r in rows], "count": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

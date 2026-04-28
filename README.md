# JJobApply Agent 

> 🌐 [中文](#中文) | [English](#english)

---

<a name="中文"></a>
# 中文

## 一、项目概述

**项目名称**：Job Track Agent

**项目定位**：基于自然语言的求职投递情况数据分析工具，支持多用户，可小范围商用

**核心功能**：用中文或英文提问，自动查询数据库并给出结构化回答；实时数据可视化看板；图片/截图 AI 自动识别并填写申请记录；在线新增、编辑、删除申请记录；搜索与排序；分页浏览；邀请码注册体系；管理员后台

---

## 二、技术栈

| 层级 | 技术 |
|------|------|
| 大模型（对话） | DeepSeek V3（via API） |
| 大模型（图像识别） | Claude Haiku 4.5（via Anthropic API） |
| 后端 | Python FastAPI + uvicorn |
| 数据库 | PostgreSQL |
| 前端 | 纯 HTML + Chart.js |
| 版本管理 | GitHub |

---

## 三、系统架构

```
用户浏览器
    │
    ├── 左侧面板（HTML + Chart.js）
    │       └── GET /stats/*  →  实时图表数据
    │
    ├── 主内容区（申请记录列表 + 搜索/排序/分页 + 在线表单）
    │       └── GET/POST/PUT/DELETE /applications
    │
    ├── AI 对话面板（自定义对话 UI，点击展开）
    │       │
    │       └── POST /chat（JWT 鉴权）
    │               │
    │               ├── DeepSeek API：自然语言 → SQL（注入 user_id）
    │               ├── PostgreSQL 执行查询
    │               └── DeepSeek API：查询结果 → 自然语言回复
    │
    └── 自动识别申请记录（上传图片或 Ctrl+V 粘贴截图）
            │
            └── POST /applications/parse-image（JWT 鉴权）
                    │
                    └── Claude Haiku 4.5 Vision：图片 → 结构化 JSON 字段
```

---

## 四、数据库结构

### 表1：job_applications
| 字段 | 类型 | 说明 |
|------|------|------|
| id | SERIAL | 主键 |
| company | TEXT | 公司名称 |
| position | TEXT | 职位名称 |
| applied_date | DATE | 投递日期 |
| location | TEXT | 国家 / 地区 |
| link | TEXT | 职位链接 |
| feedback | TEXT | 反馈结果（NULL=待回复，Fail=拒绝，Offer=录用，Interview=面试，Online Assessment=笔试） |
| work_type | TEXT | 工作类型（Remote / Onsite / Hybrid） |
| user_id | INTEGER | 关联用户 |

### 表2：users
| 字段 | 类型 | 说明 |
|------|------|------|
| id | SERIAL | 主键 |
| email | TEXT | 邮箱（唯一） |
| password_hash | TEXT | bcrypt 哈希 |
| is_admin | BOOLEAN | 是否管理员 |
| created_at | TIMESTAMPTZ | 注册时间 |

### 表3：invite_codes
| 字段 | 类型 | 说明 |
|------|------|------|
| id | SERIAL | 主键 |
| code | TEXT | 邀请码（唯一） |
| created_by | INTEGER | 生成者（管理员） |
| used_by | INTEGER | 使用者 |
| is_active | BOOLEAN | 是否有效 |
| created_at | TIMESTAMPTZ | 生成时间 |
| used_at | TIMESTAMPTZ | 使用时间 |

### 表4：chat_usage
| 字段 | 类型 | 说明 |
|------|------|------|
| user_id | INTEGER | 用户 ID |
| date | DATE | 日期 |
| count | INTEGER | 当日调用次数 |

### 表5：work_permits
| 字段 | 类型 | 说明 |
|------|------|------|
| country | TEXT | 国家 |
| visa | TEXT | 签证/工作许可类型 |
| annual_salary | TEXT | 年薪门槛 |
| permanent_residence | TEXT | 永居申请年限 |

---

## 五、API 接口

### 公开接口（无需 token）
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /auth/register | 注册（需邀请码），返回 JWT token |
| POST | /auth/login | 登录，返回 JWT token |
| GET | /health | 健康检查 |

### 业务接口（需要 Bearer token）
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /applications | 获取当前用户的申请记录 |
| POST | /applications | 新增申请记录 |
| PUT | /applications/{id} | 编辑申请记录 |
| DELETE | /applications/{id} | 删除申请记录 |
| POST | /applications/parse-image | 上传图片，AI 识别并返回申请字段 JSON |
| POST | /chat | AI 对话（每日限 50 次，每分钟限 30 次） |
| GET | /stats/summary | 总数、地点数 |
| GET | /stats/countries | Top 5 投递地点 |
| GET | /stats/worktype | 工作类型分布（Remote / Onsite / Hybrid） |

### 管理员接口（需要 Admin token）
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /admin/users | 查看所有用户 |
| DELETE | /admin/users/{id} | 删除用户及其所有记录 |
| PATCH | /admin/users/{id}/toggle-admin | 切换管理员权限 |
| PATCH | /admin/users/{id}/reset-password | 重置密码 |
| GET | /admin/invite-codes | 查看邀请码列表 |
| POST | /admin/invite-codes | 生成邀请码 |
| DELETE | /admin/invite-codes/{id} | 撤销邀请码 |

---

## 六、前端功能

- **登录/注册页**：首次访问显示认证界面；注册需填写邀请码；登录后 token 存入 localStorage，30 天有效
- **左侧栏**：logo、当前登录邮箱 + 退出按钮、Ask AI 按钮、管理员入口（仅管理员可见）、总投递数 / 地点数统计卡、Work Type 环形图（含 Hybrid）、Top Locations 柱状图（前 5）
- **主内容区**：**自动识别申请记录**（点击按钮或 Ctrl+V 粘贴截图，AI 自动提取公司/职位/地点/链接等字段）、手动新增申请记录、搜索框（按公司名/职位名实时过滤）、申请时间排序（点击列标题切换升/降序）、分页（每页 10 条）、点击记录编辑、每条记录可删除
- **AI 对话面板**：支持多轮对话，内置示例问题，每日限 50 次，拒绝回答与求职无关的问题
- **管理员后台**：用户管理（删除、切换权限、重置密码）+ 邀请码管理（生成、复制、撤销）

---

## 七、安全

- JWT token 鉴权（SECRET_KEY 存于 .env，不进 git）
- DeepSeek / Anthropic API Key 存于 .env，不进 git
- 邀请码注册控制，一码一次
- SQL 安全检查：仅允许 SELECT，强制 user_id 过滤，禁止访问非授权表
- /chat 每用户每日 50 次、每分钟 30 次双重限流
- CORS 白名单控制
- 全局错误日志写入 logs/error.log
- 每日凌晨 2 点自动备份数据库，保留 7 天

---

## 八、项目文件结构

```
~/jobtrack/
├── db_api.py          # FastAPI 后端
├── job-agent.html     # 前端页面
├── schema.sql         # 数据库建表语句
├── requirements.txt   # Python 依赖
├── .env.example       # 环境变量模板
├── backup.sh          # 数据库备份脚本
├── import_jobs.py     # 历史数据导入脚本
├── .env               # 密钥配置（不进 git）
├── logs/              # 运行日志
└── backups/           # 数据库备份文件
```

---

## 九、部署

### 环境要求
- Python 3.10+
- PostgreSQL

### 步骤

```bash
# 1. 克隆项目
git clone https://github.com/valenwei113-design/Job-Track.git
cd Job-Track

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY、ANTHROPIC_API_KEY、SECRET_KEY 和数据库信息

# 4. 建表
psql -U postgres -d jobsdb -f schema.sql

# 5. 创建日志目录
mkdir -p logs

# 6. 启动服务
uvicorn db_api:app --host 0.0.0.0 --port 8000
```

### 本地开发

```bash
# 带热重载启动
uvicorn db_api:app --host 0.0.0.0 --port 8000 --reload

# 前端直接用浏览器打开
open job-agent.html
```

---

<a name="english"></a>
# English

## 1. Overview

**Project Name**: Job Track Agent

**Purpose**: A natural-language job application tracking and analytics tool with multi-user support, suitable for small-scale deployment.

**Key Features**: Query your application data in plain Chinese or English; real-time data visualization dashboard; AI-powered image/screenshot parsing to auto-fill application records; create, edit, and delete records online; search, sort, and paginate; invite-code registration system; admin panel.

---

## 2. Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM (Chat) | DeepSeek V3 (via API) |
| LLM (Image Recognition) | Claude Haiku 4.5 (via Anthropic API) |
| Backend | Python FastAPI + uvicorn |
| Database | PostgreSQL |
| Frontend | Vanilla HTML + Chart.js |
| Version Control | GitHub |

---

## 3. Architecture

```
User Browser
    │
    ├── Left Panel (HTML + Chart.js)
    │       └── GET /stats/*  →  live chart data
    │
    ├── Main Area (application list + search/sort/pagination + form)
    │       └── GET/POST/PUT/DELETE /applications
    │
    ├── AI Chat Panel (collapsible)
    │       │
    │       └── POST /chat (JWT auth)
    │               │
    │               ├── DeepSeek API: natural language → SQL (user_id injected)
    │               ├── PostgreSQL executes query
    │               └── DeepSeek API: results → natural language reply
    │
    └── Auto-Parse (upload image or Ctrl+V paste screenshot)
            │
            └── POST /applications/parse-image (JWT auth)
                    │
                    └── Claude Haiku 4.5 Vision: image → structured JSON fields
```

---

## 4. Database Schema

### Table 1: job_applications
| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| company | TEXT | Company name |
| position | TEXT | Job title |
| applied_date | DATE | Application date |
| location | TEXT | Country / Region |
| link | TEXT | Job posting URL |
| feedback | TEXT | Status (NULL=Pending, Fail, Offer, Interview, Online Assessment) |
| work_type | TEXT | Remote / Onsite / Hybrid |
| user_id | INTEGER | Owner (foreign key) |

### Table 2: users
| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| email | TEXT | Email (unique) |
| password_hash | TEXT | bcrypt hash |
| is_admin | BOOLEAN | Admin flag |
| created_at | TIMESTAMPTZ | Registration time |

### Table 3: invite_codes
| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| code | TEXT | Invite code (unique) |
| created_by | INTEGER | Issuing admin |
| used_by | INTEGER | Redeemer |
| is_active | BOOLEAN | Still valid |
| created_at | TIMESTAMPTZ | Created at |
| used_at | TIMESTAMPTZ | Redeemed at |

### Table 4: chat_usage
| Column | Type | Description |
|--------|------|-------------|
| user_id | INTEGER | User ID |
| date | DATE | Date |
| count | INTEGER | Daily usage count |

### Table 5: work_permits
| Column | Type | Description |
|--------|------|-------------|
| country | TEXT | Country |
| visa | TEXT | Visa / work permit type |
| annual_salary | TEXT | Minimum salary threshold |
| permanent_residence | TEXT | Years to permanent residency |

---

## 5. API Reference

### Public Endpoints (no token required)
| Method | Path | Description |
|--------|------|-------------|
| POST | /auth/register | Register with invite code, returns JWT |
| POST | /auth/login | Login, returns JWT |
| GET | /health | Health check |

### Authenticated Endpoints (Bearer token required)
| Method | Path | Description |
|--------|------|-------------|
| GET | /applications | List current user's applications |
| POST | /applications | Create application |
| PUT | /applications/{id} | Update application |
| DELETE | /applications/{id} | Delete application |
| POST | /applications/parse-image | Upload image, AI extracts fields as JSON |
| POST | /chat | AI chat (50/day, 30/min per user) |
| GET | /stats/summary | Total applications & locations |
| GET | /stats/countries | Top 5 locations |
| GET | /stats/worktype | Work type distribution |

### Admin Endpoints (Admin token required)
| Method | Path | Description |
|--------|------|-------------|
| GET | /admin/users | List all users |
| DELETE | /admin/users/{id} | Delete user and all their data |
| PATCH | /admin/users/{id}/toggle-admin | Toggle admin role |
| PATCH | /admin/users/{id}/reset-password | Reset user password |
| GET | /admin/invite-codes | List invite codes |
| POST | /admin/invite-codes | Generate invite code |
| DELETE | /admin/invite-codes/{id} | Revoke invite code |

---

## 6. Frontend Features

- **Auth Page**: Shown on first visit; registration requires an invite code; JWT stored in localStorage for 30 days
- **Left Sidebar**: Logo, logged-in email + sign-out, Ask AI button, admin entry (admin only), total applications / locations stats, Work Type donut chart, Top Locations bar chart (top 5)
- **Main Area**: **Auto-parse** (click button or Ctrl+V paste a screenshot — AI extracts company, position, location, link, etc.), manual add, search (by company/position), sort by date, pagination (10 per page), inline edit, per-record delete
- **AI Chat Panel**: Multi-turn conversation, built-in example questions, 50 queries/day limit, rejects off-topic questions
- **Admin Panel**: User management (delete, toggle admin, reset password) + invite code management (generate, copy, revoke)

---

## 7. Security

- JWT authentication (SECRET_KEY in .env, excluded from git)
- DeepSeek / Anthropic API keys in .env, excluded from git
- Invite-code gated registration, one use per code
- SQL safety: SELECT only, mandatory user_id filter, blocked unauthorized tables
- Chat rate-limited: 50/day and 30/min per user
- CORS allowlist
- Global error logging to logs/error.log
- Automated daily DB backup at 2 AM, retained for 7 days

---

## 8. File Structure

```
~/jobtrack/
├── db_api.py          # FastAPI backend
├── job-agent.html     # Frontend (single-page)
├── schema.sql         # Database schema
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variable template
├── backup.sh          # Database backup script
├── import_jobs.py     # Historical data import script
├── .env               # Secrets (excluded from git)
├── logs/              # Runtime logs
└── backups/           # Database backup files
```

---

## 9. Deployment

### Requirements
- Python 3.10+
- PostgreSQL

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/valenwei113-design/Job-Track.git
cd Job-Track

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — fill in DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, SECRET_KEY, and DB credentials

# 4. Initialize database
psql -U postgres -d jobsdb -f schema.sql

# 5. Create log directory
mkdir -p logs

# 6. Start server
uvicorn db_api:app --host 0.0.0.0 --port 8000
```

### Local Development

```bash
# Start with hot reload
uvicorn db_api:app --host 0.0.0.0 --port 8000 --reload

# Open frontend directly in browser
open job-agent.html
```

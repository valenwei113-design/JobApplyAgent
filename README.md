# Job Track Agent

## 一、项目概述

**项目名称**：Job Track Agent

**项目定位**：基于自然语言的求职投递情况数据分析 Agent，本地部署，可嵌入任意网页

**核心功能**：用中文或英文提问，自动查询数据库并给出结构化回答，配合实时数据可视化看板；支持在线新增和编辑申请记录

---

## 二、技术栈

| 层级 | 技术 |
|------|------|
| AI 编排平台 | Dify 1.13.3（自托管，Docker 部署） |
| 大模型 | DeepSeek V3（via API） |
| 工作流模式 | Dify Chatflow（固定流程，强制查库） |
| 数据库 | PostgreSQL 15（Dify 内置实例复用） |
| 数据库 API | Python FastAPI + uvicorn |
| 前端 | 纯 HTML + Chart.js + Dify Embed iframe |
| 版本管理 | GitHub |

---

## 三、系统架构

```
用户浏览器
    │
    ├── 左侧面板（HTML + Chart.js）
    │       └── GET http://localhost:8000/stats/*  →  实时图表数据
    │
    ├── 中间主内容区（申请记录列表 + 在线表单）
    │       └── GET/POST/PUT http://localhost:8000/applications
    │
    └── AI 对话面板（Dify Embed iframe，点击展开覆盖主内容区）
            │
            └── Dify Chatflow 工作流
                    │
                    ├── [LLM 1] DeepSeek：自然语言 → SQL
                    │
                    ├── [HTTP Request] → FastAPI :8000/query
                    │                       └── PostgreSQL (jobsdb)
                    │
                    └── [LLM 2] DeepSeek：查询结果 → 自然语言解释
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
| feedback | TEXT | 反馈结果（NULL = 待回复，Fail = 拒绝） |
| work_type | TEXT | 工作类型（Remote / Onsite） |

### 表2：work_permits

| 字段 | 类型 | 说明 |
|------|------|------|
| country | TEXT | 国家 |
| visa | TEXT | 签证/工作许可类型 |
| annual_salary | TEXT | 年薪门槛 |
| permanent_residence | TEXT | 永久居留申请年限 |

---

## 五、API 接口

### 认证接口（无需 token）
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /auth/register | 注册账号，返回 JWT token |
| POST | /auth/login | 登录，返回 JWT token |
| POST | /query | 执行 SELECT 查询（供 Dify 调用） |
| GET | /health | 健康检查 |

### 业务接口（需要 Bearer token）
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /applications | 获取当前用户的申请记录 |
| POST | /applications | 新增申请记录 |
| PUT | /applications/{id} | 编辑申请记录 |
| GET | /stats/summary | 总数、国家数 |
| GET | /stats/countries | Top 10 投递国家 |
| GET | /stats/worktype | 工作类型分布（Remote / Onsite） |

---

## 六、前端功能

- **登录/注册页**：首次访问显示认证界面，登录后 token 存入 localStorage，30 天有效
- **左侧栏**：JobTrack AI logo、当前登录邮箱 + 退出按钮、Ask AI 按钮、总投递数 / 国家数统计卡、Work Type 环形图、Top Countries 柱状图（前 10）
- **主内容区**：仅显示当前账号的申请记录（公司、职位、地点、工作类型、链接、申请时间、反馈），点击记录可编辑，顶部"新增申请记录"按钮（默认填入当天日期）
- **AI 对话面板**：点击 Ask AI 展开，覆盖主内容区；内嵌 Dify Chatflow iframe
- **设计风格**：亮色主题，紫色（#6366f1）主色调，No Response 徽章灰色，Rejected 红色

---

## 七、遇到的问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 浏览器访问 localhost 显示空白 | API 容器在 PostgreSQL 前启动，数据库迁移失败 | 重启 api 容器，reload nginx 刷新 DNS |
| DeepSeek 不稳定调用工具 | Agent 模式依赖模型自主判断 | 改用 Chatflow 工作流，强制每次执行数据库查询 |
| passlib + bcrypt 500 错误 | bcrypt >= 4.1 移除了 `__about__` 属性 | 降级至 bcrypt==4.0.1 |
| 空白 feedback 被存为 "NaN" | pandas 读取 CSV 时将空单元格转为字符串 | 导入后 UPDATE 将 'NaN' 改为 NULL |
| LLM 生成 SQL 错误 | 字段语义不清晰 | 在系统提示词中明确字段含义和映射关系 |
| 图表无法加载 | file:// 协议跨域限制 | 改用 `python3 -m http.server 9090` 本地服务 |
| applied_date 无法范围查询 | 日期存为文本格式 | 重写导入脚本，转换为 DATE 类型 |

---

## 八、项目文件结构

```
job-search-track-agent/
├── db_api.py          # FastAPI 数据库服务（查询接口 + 统计接口）
├── job-agent.html     # 前端页面（申请记录列表 + 在线表单 + AI 对话）
├── import_jobs.py     # 历史数据导入脚本（CSV → PostgreSQL，一次性使用）
└── .gitignore
```

---

## 九、本地启动方式

```bash
# 0. 安装依赖（首次）
pip3 install fastapi uvicorn psycopg2-binary passlib "bcrypt==4.0.1" python-jose openai python-dotenv

# 1. 进入项目目录
cd ~/jobtrack

# 2. 启动 PostgreSQL（Dify Docker，如果数据库在 Dify 容器里）
cd ~/dify/docker && docker compose up -d && cd ~/jobtrack

# 3. 启动 FastAPI
python3 -m uvicorn db_api:app --host 0.0.0.0 --port 8000 &

# 4. 启动前端服务
python3 -m http.server 9090

# 5. 打开页面
open http://localhost:9090/job-agent.html
```

---

## 十、关键数据

- 总投递记录：**268 条**
- 覆盖国家/地区：**42 个**
- 工作许可数据：**6 个国家**

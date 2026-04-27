# Job Track Agent

## 一、项目概述

**项目名称**：Job Track Agent

**项目定位**：基于自然语言的求职投递情况数据分析工具，支持多用户，可小范围商用

**核心功能**：用中文或英文提问，自动查询数据库并给出结构化回答；实时数据可视化看板；在线新增、编辑、删除申请记录；搜索与排序；分页浏览；邀请码注册体系；管理员后台

---

## 二、技术栈

| 层级 | 技术 |
|------|------|
| 大模型 | DeepSeek V3（via API，直接调用） |
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
    └── AI 对话面板（自定义对话 UI，点击展开）
            │
            └── POST /chat（JWT 鉴权）
                    │
                    ├── DeepSeek API：自然语言 → SQL（注入 user_id）
                    ├── PostgreSQL 执行查询
                    └── DeepSeek API：查询结果 → 自然语言回复
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
- **主内容区**：搜索框（按公司名/职位名实时过滤）、申请时间排序（点击列标题切换升/降序）、分页（每页 20 条）、点击记录编辑、每条记录可删除
- **AI 对话面板**：支持多轮对话，内置示例问题，每日限 50 次，拒绝回答与求职无关的问题
- **管理员后台**：用户管理（删除、切换权限、重置密码）+ 邀请码管理（生成、复制、撤销）

---

## 七、安全

- JWT token 鉴权（SECRET_KEY 存于 .env，不进 git）
- DeepSeek API Key 存于 .env，不进 git
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
# 编辑 .env，填入真实的 key 和数据库信息

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

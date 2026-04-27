# Job Track Agent

## 一、项目概述

**项目名称**：Job Track Agent

**项目定位**：基于自然语言的求职投递情况数据分析工具，支持多用户，可小范围商用

**核心功能**：用中文或英文提问，自动查询数据库并给出结构化回答；实时数据可视化看板；在线新增和编辑申请记录；邀请码注册体系；管理员后台

---

## 二、技术栈

| 层级 | 技术 |
|------|------|
| 大模型 | DeepSeek V3（via API，直接调用） |
| 后端 | Python FastAPI + uvicorn |
| 进程管理 | supervisor（自动重启，开机自启） |
| 数据库 | PostgreSQL 15（Docker 容器） |
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
    ├── 主内容区（申请记录列表 + 在线表单）
    │       └── GET/POST/PUT /applications
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
| user_id | INTEGER | 关联用户，加索引 |

### 表2：users
| 字段 | 类型 | 说明 |
|------|------|------|
| id | SERIAL | 主键 |
| email | TEXT | 邮箱（唯一） |
| password_hash | TEXT | bcrypt 哈希 |
| is_admin | BOOLEAN | 是否管理员 |
| created_at | TIMESTAMP | 注册时间 |

### 表3：invite_codes
| 字段 | 类型 | 说明 |
|------|------|------|
| id | SERIAL | 主键 |
| code | TEXT | 邀请码（唯一） |
| created_by | INTEGER | 生成者（管理员） |
| used_by | INTEGER | 使用者 |
| is_active | BOOLEAN | 是否有效 |

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
| POST | /chat | AI 对话（每日限 50 次） |
| GET | /stats/summary | 总数、地点数 |
| GET | /stats/countries | Top 5 投递地点 |
| GET | /stats/worktype | 工作类型分布 |

### 管理员接口（需要 Admin token）
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /admin/users | 查看所有用户 |
| DELETE | /admin/users/{id} | 删除用户 |
| PATCH | /admin/users/{id}/toggle-admin | 切换管理员权限 |
| PATCH | /admin/users/{id}/reset-password | 重置密码 |
| GET | /admin/invite-codes | 查看邀请码列表 |
| POST | /admin/invite-codes | 生成邀请码 |
| DELETE | /admin/invite-codes/{id} | 撤销邀请码 |

---

## 六、前端功能

- **登录/注册页**：首次访问显示认证界面；注册需填写邀请码；登录后 token 存入 localStorage，30 天有效
- **左侧栏**：JobTrack AI logo、当前登录邮箱 + 退出按钮、Ask AI 按钮、管理员入口（仅管理员可见）、总投递数 / 地点数统计卡、Work Type 环形图、Top Locations 柱状图（前 5）
- **主内容区**：仅显示当前账号的申请记录，点击可编辑，工作类型和反馈支持自定义输入
- **AI 对话面板**：自定义对话 UI，支持多轮对话，每日限 50 次，拒绝回答与求职无关的问题
- **管理员后台**：用户管理（删除、切换权限、重置密码）+ 邀请码管理（生成、复制、撤销）

---

## 七、安全与运维

- JWT token 鉴权（SECRET_KEY 存于 .env，不进 git）
- DeepSeek API Key 存于 .env，不进 git
- 邀请码注册控制，一码一次
- /chat 每用户每日 50 次调用限制
- supervisor 守护进程，自动重启，开机自启
- 每日凌晨 2 点自动备份数据库，保留 7 天
- 全局错误日志写入 logs/error.log

---

## 八、遇到的问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| passlib + bcrypt 500 错误 | bcrypt >= 4.1 移除了 `__about__` 属性 | 降级至 bcrypt==4.0.1 |
| Dify iframe 无法做用户数据隔离 | Dify 不感知外部登录体系 | 改为直接调用 DeepSeek API，user_id 由后端注入 |
| 空白 feedback 被存为 "NaN" | pandas 读取 CSV 时将空单元格转为字符串 | 导入后 UPDATE 将 'NaN' 改为 NULL |
| applied_date 无法范围查询 | 日期存为文本格式 | 重写导入脚本，转换为 DATE 类型 |
| 图表无法加载 | file:// 协议跨域限制 | 改用 `python3 -m http.server 9090` 本地服务 |

---

## 九、项目文件结构

```
~/jobtrack/
├── db_api.py          # FastAPI 后端（认证、业务、AI 对话、管理员接口）
├── job-agent.html     # 前端页面
├── backup.sh          # 数据库备份脚本
├── import_jobs.py     # 历史数据导入脚本（一次性）
├── .env               # 密钥配置（不进 git）
├── .gitignore
├── logs/              # 运行日志
└── backups/           # 数据库备份文件
```

---

## 十、本地启动方式

```bash
# 0. 安装依赖（首次）
pip3 install fastapi uvicorn psycopg2-binary passlib "bcrypt==4.0.1" python-jose openai python-dotenv

# 1. 进入项目目录
cd ~/jobtrack

# 2. 启动 PostgreSQL（Docker）
cd ~/dify/docker && docker compose up -d && cd ~/jobtrack

# 3. 启动 FastAPI（supervisor 管理，开机自启）
brew services start supervisor

# 4. 启动前端服务
python3 -m http.server 9090

# 5. 打开页面
open http://localhost:9090/job-agent.html
```

---

## 十一、关键数据

- 总投递记录：**268 条**
- 覆盖地点：**42 个**
- 工作许可数据：**6 个国家**

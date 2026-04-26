# Job Track Agent

## 一、项目概述

**项目名称**：Job Search Track Agent
**项目定位**：基于自然语言的求职数据分析 Agent，本地部署，可嵌入任意网页
**核心功能**：用中文或英文提问，自动查询数据库并给出结构化回答，配合实时数据可视化看板

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
    └── 右侧聊天（Dify Embed iframe）
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

**数据来源**：Apple Numbers 文件（Task Track.numbers），导出 CSV 后用 Python 脚本导入 PostgreSQL。

### 表1：job_applications（268 条记录）

| 字段 | 类型 | 说明 |
|------|------|------|
| company | TEXT | 公司名称 |
| position | TEXT | 职位名称 |
| applied_date | TEXT | 投递日期（如 Jan 13） |
| location | TEXT | 国家或工作方式（如 Norway、Remote） |
| link | TEXT | 职位链接 |
| feedback | TEXT | 反馈结果（NULL = 待回复，Fail = 拒绝） |

### 表2：work_permits（6 条记录）

| 字段 | 类型 | 说明 |
|------|------|------|
| country | TEXT | 国家 |
| visa | TEXT | 签证/工作许可类型 |
| annual_salary | TEXT | 年薪门槛 |
| permanent_residence | TEXT | 永久居留申请年限 |

---

## 五、实施步骤

### Step 1：部署 Dify
- 克隆官方仓库，使用 `docker compose` 启动
- 排查并修复容器启动顺序问题（API 先于 PostgreSQL 启动导致数据库迁移失败）
- 修复 nginx DNS 缓存问题（容器重启后 IP 变化）

### Step 2：接入大模型
- 尝试 Anthropic Claude（包月订阅不含 API 权限，放弃）
- 改用 DeepSeek V3，通过 platform.deepseek.com 获取 API Key
- 在 Dify 模型供应商中配置，设为默认推理模型

### Step 3：数据导入
- 原始数据为 Apple Numbers 格式，导出为 CSV
- 编写 Python 脚本（import_jobs.py）导入 PostgreSQL
- 修复 pandas 将空白单元格导入为字符串 `"NaN"` 的问题，统一改为 NULL

### Step 4：构建数据库 API
- 用 FastAPI 编写 `/query` 接口，供 Dify 调用
- 加入 SQL 白名单校验，禁止 INSERT/UPDATE/DELETE 等写操作
- 新增 `/stats/summary`、`/stats/countries`、`/stats/feedback` 统计接口

### Step 5：搭建 Dify Chatflow 工作流
- 初期使用 Agent 模式，但 DeepSeek 有时自主决定不调用工具，结果不稳定
- 改用 **Chatflow 工作流模式**，硬编码三步流程：
  1. LLM 节点：将用户问题转换为 SQL
  2. HTTP Request 节点：调用 FastAPI 执行 SQL
  3. LLM 节点：将查询结果转化为自然语言回答
- 优化第一个 LLM 的系统提示词，明确字段含义（如 location = 国家）

### Step 6：前端页面
- 基于 Dify Embed iframe 构建聊天界面
- 左侧边栏：实时统计数字 + 饼图（反馈状态）+ 柱状图（国家分布 Top 8）
- 设计风格：亮色主题，参考 Linear / Notion 风格

---

## 六、遇到的问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 浏览器访问 localhost 显示空白 | API 容器在 PostgreSQL 前启动，数据库迁移失败后 gunicorn 仍启动但无法响应 | 重启 api 容器触发重新迁移，reload nginx 刷新 DNS |
| DeepSeek 不稳定调用工具 | Agent 模式依赖模型自主判断，DeepSeek 有时跳过工具调用 | 改用 Chatflow 工作流，强制每次执行数据库查询 |
| 空白 feedback 被存为 "NaN" | pandas 读取 CSV 时将空单元格转为字符串 NaN | 导入后执行 UPDATE 将 'NaN' 改为 NULL |
| LLM 生成 SQL 错误 | 字段语义不清晰，如"国家"未映射到 location 列 | 在系统提示词中明确字段含义和映射关系 |
| HTTP Request 变量引用无效 | 手动输入 `{{#变量#}}` 未被解析 | 改用 Dify 变量选择器（输入 `/` 选择）插入变量 |

---

## 七、当前能力演示

| 问题示例 | 实际查询 |
|----------|----------|
| 我一共投了多少家公司？ | `SELECT COUNT(*) FROM job_applications` |
| 哪个国家投的最多？ | `SELECT location, COUNT(*) GROUP BY location ORDER BY count DESC` |
| 有多少家还没给反馈？ | `SELECT COUNT(*) WHERE feedback IS NULL` |
| 哪个国家最容易拿工作签证？ | `SELECT country, permanent_residence FROM work_permits ORDER BY permanent_residence` |

---

## 八、项目文件结构

```
job-search-track-agent/
├── db_api.py          # FastAPI 数据库服务（查询接口 + 统计接口）
├── job-agent.html     # 前端页面（聊天 + 实时图表）
├── import_jobs.py     # 数据导入脚本（CSV → PostgreSQL）
└── .gitignore

Desktop/Job Track Agent/           # 数据文件目录（不上传 GitHub）
├── job_applications.csv           # 求职投递记录
├── work_permits.csv               # 工作许可数据
└── 更新数据库.command              # 一键同步脚本
```

---

## 九、下一步计划

- [ ] 部署到云服务器（公网可访问）
- [x] 数据库一键更新（双击 `更新数据库.command` 同步最新 CSV）
- [ ] 支持更复杂的查询（多表 JOIN、时间范围筛选）
- [ ] 增加投递趋势折线图（按月统计）

---

## 十、关键数据

- 总投递记录：**268 条**
- 覆盖国家/地区：**42 个**
- 待回复：**197 家**
- 已拒绝：**71 家**
- 工作许可数据：**6 个国家**

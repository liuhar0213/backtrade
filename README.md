# Backtrade v5 - 量化交易特征系统

> **v5.0** | 简化JSON格式 | 166特征 | 生产就绪 ✅

[![CI](https://github.com/liuhar0213/backtrade/actions/workflows/ci.yml/badge.svg)](https://github.com/liuhar0213/backtrade/actions/workflows/ci.yml)

## 🚀 快速开始

```bash
# 一键运行
python start_stage_abc_v5.py --symbol BTCUSDT_15 --quick

# 查看结果
ls results/v5_stage_abc/BTCUSDT_15/
```

## 📊 系统概览

```
166个特征 = Stage A (80) + Stage B (56) + Stage C (30)
           ↓            ↓            ↓
         趋势/波段    行为/反转    流动性/风控/稳态
```

### 特征分层

| 阶段 | Layer | 特征数 | 配置文件 |
|------|-------|--------|----------|
| **Stage A** | Trend/Structure | 60 | 高级趋势分析×1, 阿佩尔×1, Vegas×1 |
| | Rhythm/Swing | 20 | 高级波段分析×1 |
| **Stage B** | Action/Reversal | 56 | 反转分析×1, 蜡烛图×1, 超短线×1, PA×1 |
| **Stage C** | Liquidity | 10 | 量价分析×1 |
| | Risk | 10 | 资金管理×1 |
| | Stability | 10 | 系统稳态×1 |

## 📁 项目结构

```
backtrade/
├── .claude/
│   └── commands/           # Claude快捷命令
│       ├── v5-docs.md      # /v5-docs
│       ├── v5-run.md       # /v5-run
│       └── v5-status.md    # /v5-status
├── configs/
│   └── books_v5/           # ⭐ 11个JSON配置
├── core/
│   └── v5/                 # ⭐ 计算引擎
│       ├── compute_feature_v5.py           # Stage A (80)
│       ├── compute_feature_v5_stage_b.py   # Stage B (56)
│       ├── compute_feature_v5_stage_c.py   # Stage C (30)
│       └── fusion_stage_abc.py             # 融合器
├── data/                   # 市场数据
├── docs/                   # 📚 文档中心
│   ├── README.md           # 文档索引
│   ├── v5_knowledge_base.md # ⭐ 完整知识库
│   └── QUICK_REFERENCE.md  # 快速参考
├── results/                # 输出结果
├── start_stage_abc_v5.py   # ⭐ 主程序入口
└── README.md               # 本文档
```

## 🎯 核心命令

```bash
# 基础运行（推荐）
python start_stage_abc_v5.py --symbol BTCUSDT_15 --quick

# 完整验证（含HSIC检查）
python start_stage_abc_v5.py --symbol BTCUSDT_15

# 指定阶段
python start_stage_abc_v5.py --symbol BTCUSDT_15 --stages A B

# 不保存结果
python start_stage_abc_v5.py --symbol BTCUSDT_15 --no-save
```

## 📖 文档资源

- **[完整知识库](docs/v5_knowledge_base.md)** - 系统架构、JSON规范、问题排查
- **[快速参考](docs/QUICK_REFERENCE.md)** - 3秒速查表、常见问题

## CI / 测试

- 持续集成: GitHub Actions workflow `.github/workflows/ci.yml` 在 `push` / `pull_request` 到 `master` 时触发。
- Workflow 步骤:
  - 安装 Python 3.11
  - 安装 `requirements.txt`（如果存在）
  - 运行 smoke 脚本: `python scripts/run_orchestrator_init.py`
  - 运行单元测试: `pytest -q`

本地快速运行：

```powershell
# 在仓库根运行 smoke 脚本
$env:PYTHONPATH='.'; python scripts/run_orchestrator_init.py

# 运行 pytest（只会运行 `tests/` 下的测试）
$env:PYTHONPATH='.'; python -m pytest -q
```
- **[文档索引](docs/README.md)** - 项目导航、文件说明

或使用Claude命令：
- `/v5-docs` - 查看完整文档
- `/v5-run` - 运行系统
- `/v5-status` - 检查状态

## 🔧 v5新特性

### JSON格式简化

**旧格式（v4.x）:**
```json
{
  "chapter": "章节名",
  "feature_id": "特征名",
  "formula": "公式",
  "deviation": "robust_dev(...)",
  "entropy_weight": "1 - H_t/H_max",
  "meta_params": {"alpha": 0.9, "p": 0.7, "kappa": 0.25},
  "relations": {...}
}
```

**新格式（v5.0）:**
```json
{
  "feature_id": "特征名",
  "formula": "公式描述"
}
```

✅ **优势**:
- 配置文件减少70%
- 更易维护
- 默认参数自动注入

### 关键修复

- ✅ Windows编码兼容（Unicode→ASCII）
- ✅ Index重复问题（reset_index）
- ✅ Layer路由支持斜杠（"Action/Reversal"）
- ✅ 默认meta_params自动注入
- ✅ 按位运算符修复

## 📊 运行结果示例

```
[1/6] 加载数据: 22697根K线
[2/6] 初始化融合器: Stage A, B, C
[3/6] 加载JSON规范: 11本书, 166特征
[4/6] 构建特征矩阵
  [1/166] trend_energy_flow... [OK]
  [2/166] channel_dev_z... [OK]
  ...
  [166/166] steady_state_flag... [OK]

  [Time] 构建耗时: 9.66秒

[5/6] 质量验证
  能量平衡: [FAIL] (需调优)
  相关性: [FAIL] (需调优)

[6/6] 保存结果
  特征矩阵: D_hat_stage_abc.csv (22697×166)
  特征列表: feature_names.txt
  层级索引: layer_blocks.json
  质量报告: quality_report.txt

[OK] 所有文件已保存
```

## 🐛 故障排查

| 问题 | 解决方案 |
|------|---------|
| 编码错误 | ✅ 已修复（v5.0） |
| Index重复 | ✅ 已修复（v5.0） |
| 能量不平衡 | 调整 `lambda_energy`: 0.15 → 0.08 |
| 特征全0/NaN | 检查数据列名（必须小写） |

更多问题查看: [v5_knowledge_base.md](docs/v5_knowledge_base.md#-常见问题排查)

## 🎓 下一步

1. **参数调优** - 提升能量平衡和相关性指标
2. **特征验证** - 分析特征分布和统计特性
3. **模型训练** - 使用特征矩阵训练LightGBM/XGBoost
4. **分层训练** - 利用layer_blocks.json进行分层建模

## 📞 支持

- **文档**: `docs/` 目录
- **Claude命令**: `/v5-docs`, `/v5-run`, `/v5-status`
- **项目路径**: `C:\Users\27654\Desktop\交易\backtrade`

---

**版本**: v5.0
**状态**: ✅ 生产就绪
**最后更新**: 2025-10-29
**特征总数**: 166 (80+56+30)

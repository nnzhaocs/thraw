## 目录结构

```
THAW/
  README.md                本文件
  requirements.txt
  semview/                 测量核心：migration、estimate、union、online 服务、
                           procrustes、runlog、tests
  thaw_pg/                 PostgreSQL 实现（load、backfill、retriever、step2、
                           sql_check、encoders）
  thaw_audit/              claim 账本工具（仅代码；其数据已移除，见下文）
  analysis/                各实验工具：n_ladder、chain、scaling_mechanism、
                           estimate_check、operator_safety、margin_cert、
                           drift_sweep、tab_form、gen2_union 等
  _local/                  运维脚本：盒子访问、A6 启动器、A7 登记批处理、
                           Windows DLL 防护包装器
  beir/                    BEIR 布局的 MS MARCO 语料（3.2 GB）
  cache/                   按内容哈希键控的编码矩阵（17 GB）：500k 的
                           MiniLM / bge-small / bge-base / bge-m3 / Qwen3、
                           8.8M MiniLM 空间、500k 的 f16 变体
  step1_data/              step1 向量（950 MB）
  dump/                    leg0..leg3 的各代 query dump（110 MB）
  pool.jsonl               gen2/pool 轮次使用的 passage 池
  results/runs/            77 个 run 目录，每个含 command.txt、env.json、
                           log.txt、结果 .json 及其 sha256/MANIFEST
  external/
    wiki_dpr_clean/        DPR 估计读取的 wiki_dpr parquet 分片
    models/huggingface/    模型权重（MiniLM、bge-small/base/m3、
                           Qwen3-Embedding-0.6B、DPR question/ctx、e5、mpnet）
    models/modelscope/     bge-m3 与 Qwen3-Embedding-0.6B 的快照路径
    db/thaw.dump           已加载数据库的 pg_dump -Fc 转储（50 万文档、
                           两个模型的向量），恢复：pg_restore -d thaw
```

## 如何运行

```bash
pip install -r requirements.txt
export USE_TF=0 TRANSFORMERS_NO_TF=1 HF_HUB_OFFLINE=1   # Windows 防护，见 semview/__init__.py
python -m semview.tests
```




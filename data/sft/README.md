# data/sft

本目录保存 TravelGym canonical SFT 语料、分类审核结果和固定 split。它是“Teacher 轨迹 → 清洗 → SFT”的数据落点，不是 GRPO rollout 缓存目录。

当前 checked-in 版本包括 795 条合并 canonical 记录，固定 SFT split 为 490 条 train 和 10 条 held-out validation。数据来源、分类、mask 和任务隔离见 [数据采集](../../docs/data_collection.md) 与 [SFT](../../docs/sft.md)。

重要产物：

- travel_sft_public.json：历史公开输入；
- travel_sft_qwen35_merged.jsonl：合并后的 canonical 语料；
- travel_sft_qwen35_split/：训练和验证输入及 split manifest；
- audit/manifest/class 文件：清洗结果和可追溯统计。

原始 Teacher cache 和私有答案只应保存在 outputs 的受控目录。

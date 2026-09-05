# GRPO data

The GRPO Parquet files are stored here:

- data/grpo/travel*_multiturn_onechoice/
- data/grpo/alltrain_multiturn/
- data/grpo/alltest_multiturn/

The task-pool manifests remain under:

data/task_pools/

The manifests contain repository-relative source_path values pointing to data/grpo/.
The GRPO launcher consumes these manifests directly.
Legacy data/* paths are retained as compatibility symlinks.
The Parquet files are not duplicated.

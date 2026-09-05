# Collection

This package owns the offline data path from Teacher/evaluator output to
canonical TravelGym corpora and task-level splits.

Key modules:

- `teacher_collection.py`, `summarize_teacher_collection.py`: Teacher cache
  bookkeeping and summaries;
- `clean_travel_trajectories.py`, `travel_canonical.py`,
  `travel_task_resolver.py`: public-protocol replay, canonicalization, and
  private task recovery;
- `prepare_travel_sft.py`, `merge_travel_sft.py`, `task_pools.py`:
  canonical SFT preparation and disjoint pool construction;
- `merge_customize.py`, `sanitize_travel_labels.py`,
  `travel_multiturn_w_tool.py`: data-preprocessing utilities.

Run the CLIs as modules, for example:

```bash
python -m travel_grpo.collection.task_pools --help
python -m travel_grpo.collection.merge_travel_sft --help
```

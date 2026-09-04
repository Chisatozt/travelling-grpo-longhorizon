# TravelGym Contribution Guide

All environment changes in this repository target `gyms/TravelGym`.

## Required invariants

1. Search returns the complete `task["all_options"][aspect]` list.
2. Action only elicits or records natural-language evidence; it never filters
   candidates or exposes preference IDs.
3. Answer submits exactly one ID visible in the current Search result.
4. Public observations contain only the six control fields and public
   conversation history described in `docs/travel_public_protocol.md`.
5. Reward remains terminal-only (`travelgym-terminal-v2`); diagnostics stay out
   of Actor observations and tool feedback.

## Validation

Run the offline contract tests before opening a change:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

When changing data generation, regenerate only the TravelGym parquet variants
and verify that `reward_model.env_name` is `TravelGym` for every row.

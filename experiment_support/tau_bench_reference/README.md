# Pinned business reference source

This is the small Python source snapshot used by the native-tool preparation and differential validation workflow. `tau_bench/envs/tool.py` and the retail/airline tool implementations supply the independent `invoke` results against complete pinned databases.

The complete source databases are external. To use this tree as `--tau-root`, provide the matching `tau_bench/envs/retail/data/{users,orders,products}.json` and `tau_bench/envs/airline/data/{users,reservations,flights}.json`. Preserve `LICENSE`. The reference code does not supply model inference or reproduce natural-language/user-confirmation checks.

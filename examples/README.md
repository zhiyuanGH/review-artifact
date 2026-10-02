# Frozen workflow input specifications

These small JSON files are source examples: workflow definitions, explicit dependency constraints and a runtime configuration. They contain no recorded outcomes, latency observations, experimental result rows, initial/expected states or databases. Full cohorts and state/reference snapshots remain external to the source-only upload.

`selected10_workflows.json` contains the ten selected executable workflows with source indices **120, 7, 33, 100, 186, 213, 239, 48, 152, 179**. They came from the 240 nonempty members of the compact280 population, sorted by `(business-node count, source selection_index, workflow_id)` and selected at ranks `floor(i*(N-1)/9+0.5)` for `i=0..9`. Forty zero-node workflows were excluded. The selected workflows contain 35 business nodes and 35 inserted checker nodes; this rank sample does not establish broad representativeness.

`workflow_120.json` is a small measured derived workflow. `workflow_179.json` also shows the explicit conflict dependency added before execution: the address-update checker must finish before changing the same order's items. These are the frozen actions and DAGs used by the current measurements. Business arguments and benchmark synthetic identifiers have been preserved.

`runtime_ort_p1.json` is the corresponding corrected ORT p=1 configuration for workflow 120. It retains the 64-token budget, 240-second timeout, complete-prompt setting and native-tool binding, while using `adb` on `PATH`, `DEVICE_A_SERIAL` as a public placeholder and a fresh example remote directory. Replace the serial and supply the matching phone runtime/model installation before use. It is an inspection example, not a complete prepared-run manifest.

Unrelated generation notes, the private cloud endpoint and original machine/source provenance were removed from the published selection document. The retained `wfgen.v1` field is a runtime schema compatibility identifier.

These examples do not include the complete original database, expected final states, raw results, launch logs, or models. Their embedded reference-informed action values mean this is fixed-action replay, rather than a prompt-only planning or independent semantic-checker evaluation. See `../experiment_support/README.md` for dependencies and measurement scope.

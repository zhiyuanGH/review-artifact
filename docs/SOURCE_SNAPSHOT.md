# Source snapshot scope

The local smoke-test sample catalogs are generic examples, not the fixed
ten-workflow paper cohort. The latter is identified in the paper measurement guide and the external-input
mapping in `paper_artifacts/`.

This snapshot collects the current framework working tree, the native mobile
tool/measurement sources used for the paper, and plotting/analysis source. Experimental inputs and outputs are outside
this source-only upload.
Original source repositories and measurement records are left unchanged.

Packaging changes are limited to anonymization and portability: removing author
links and host paths, configuring device serials, resolving files relative to the
package, and making the copied Python package importable independently of the
clone directory name. The Python import namespace is `MOBench`; entrypoint
filenames and workflow schema identifiers retain their compatibility names.
The copied generated Python runner loads its embedded JSON with `json.loads`
so JSON `null`/`true` values are valid at runtime, and its default config resolves
relative to the installed MOBench package. This corrects template execution without
changing the workflow graph or benchmark parameters.

Scheduling algorithms, benchmark parameters, generation budgets, and failure
accounting are preserved.

The native tool implementation and JSON protocol can be built from source.
Hardware inference still needs the supported runtime, model files, and a user's
own device configuration. Experimental data, figure snapshots, binaries, and raw traces are outside this
source-only snapshot. Minimal workflow definitions and configuration examples
are included to explain the interfaces and exercise mocked execution.

`MANIFEST.sha256` records the packaged files. It checks this upload snapshot,
not the original raw-measurement archive. No original Git history, controller
launch state, reviewer documents, credentials, or personal Git configuration
is included.

# Phone-native business tools

The C++17 implementation exposes fourteen retail/airline tool entrypoints through a JSON stdin/stdout protocol. It performs shared read locking, exclusive mutation locking, and atomic complete-state writes. Errors return failure and do not commit state changes.

The request contains `entrypoint`, `parameters`, and optional `upstream`. Supply the state envelope explicitly with `--state STATE_JSON`:

```sh
wfgen-tools get_user_details --state STATE_JSON
```

`src/main.cpp` and the scheduling/reference semantics retain the measured implementation. `vendor/JSON_LICENSE` and `vendor/TAU_BENCH_LICENSE` preserve upstream notices. Build helpers, validators, integration runners and the condition catalog are under `scripts/` and `configs/`.

See `../README.md` for measured scope, execution flags, device profile parameters, timing limits and external datasets. No compiled binary, model file, source database or measurement output is included here.

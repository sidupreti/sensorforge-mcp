# Changelog

All notable changes to this project are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [semantic versioning](https://semver.org/).

## [0.1.0] - 2026-10-30

First release.

### Added

- MCP server over stdio, started with `uvx sensorforge-mcp`, for the hosted SensorForge Optimizer.
- Tools: `list_chips`, `check_fit`, `lossless`, `check_data`, `optimize`, `wait_for_run`, `run_status`, `download`,
  `get_report`, `list_runs`, `stop_run`, `account`.
- Prompt `fit_model_to_chip`: check fit, lossless, check the data, optimize, wait, download.
- Local checks before any upload: file type (PyTorch and Keras files refused with how to export), size limits, zip
  contents, and the data against the model's input and output shapes.
- Streamed uploads with progress notifications; `wait_for_run` reports progress while it waits.
- Downloads unpacked into `<model folder>/sensorforge/<run id>/`, refusing paths outside the folder, links and
  overwrites; nothing downloaded is executed.
- `server.json` for the MCP Registry.

[0.1.0]: https://github.com/sidupreti/sensorforge-mcp/releases/tag/v0.1.0

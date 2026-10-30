# sensorforge-mcp

<!-- mcp-name: io.github.sidupreti/sensorforge-mcp -->

We shrink models to fit edge devices. This MCP server lets your coding agent (Claude Code, Cursor, VS Code, Claude
Desktop, Windsurf, Cline) send a trained `.tflite` or `.onnx` model to the hosted SensorForge Optimizer and unpack the
result into your project: a TensorFlow Lite Micro model and its C sources that need less SRAM and flash, within the
accuracy limit you set, with every figure measured by running the file.

The server runs on your machine over stdio. It reads the model and data files you name, calls the SensorForge service
with your API key, and writes the results next to your model. The optimizer itself runs on the service.

## Get an API key

Sign in at <https://sensorforge.dev/optimize/app> and create a key under **API keys**. Keys start with `sfk_`. Keep it
out of your repository: pass it as an environment variable, as in the configurations below.

## Install

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) (for `uvx`). Python 3.10 or newer is fetched by
uv if you don't have it.

**Claude Code**

```bash
claude mcp add sensorforge -e SENSORFORGE_API_KEY=sfk_... -- uvx sensorforge-mcp
```

**Claude Desktop** (`claude_desktop_config.json`), **Cursor** (`~/.cursor/mcp.json`, or `.cursor/mcp.json` in a
project), **Windsurf** (`~/.codeium/windsurf/mcp_config.json`) and **Cline** (MCP Servers, Configure,
`cline_mcp_settings.json`) all take the same block:

```json
{
  "mcpServers": {
    "sensorforge": {
      "command": "uvx",
      "args": ["sensorforge-mcp"],
      "env": { "SENSORFORGE_API_KEY": "sfk_..." }
    }
  }
}
```

**VS Code** (`.vscode/mcp.json`; VS Code asks for the key once and stores it):

```json
{
  "inputs": [
    { "type": "promptString", "id": "sensorforge-key", "description": "SensorForge API key", "password": true }
  ],
  "servers": {
    "sensorforge": {
      "type": "stdio",
      "command": "uvx",
      "args": ["sensorforge-mcp"],
      "env": { "SENSORFORGE_API_KEY": "${input:sensorforge-key}" }
    }
  }
}
```

Restart the client (or reload its MCP servers), then ask it to list the SensorForge tools.

## What to ask

- "Will `models/kws.tflite` fit on an nRF54L15? Check it with SensorForge."
- "Run SensorForge lossless on `models/kws.tflite` for the nrf5340 and put the C files in `firmware/model/`."
- "Optimize `models/kws.tflite` for the esp32s3 with `data/kws.npz`: least SRAM, at most 1 point of accuracy lost.
  Check the data first."
- "What's left on my SensorForge account today?"

The server also has a prompt, `fit_model_to_chip`, that walks the agent through every step (in Claude Code:
`/mcp__sensorforge__fit_model_to_chip`). For a repository, drop [`examples/AGENTS.md`](examples/AGENTS.md) into your
project so any agent knows the flow.

## Tools

| Tool | What it does |
|---|---|
| `list_chips` | The chips you can target, with each one's SRAM and flash budget. |
| `check_fit` | Does the model run on a chip, and in how much SRAM and flash? Seconds; no data. |
| `lossless` | Less SRAM and flash, outputs bit-identical to yours; no data. Waits, downloads and unpacks. |
| `check_data` | Checks your data file against the model, locally. Nothing is uploaded. |
| `optimize` | The smallest model within your accuracy limit, retrained on your data. Returns a run id at once (5–60 min). |
| `wait_for_run` | Waits for a run (up to 10 minutes per call) and returns the result; call again while it says still running. |
| `run_status` | A run's state now: place in the queue, running, or the verdict with before → after figures. |
| `download` | Unpacks a finished run into `<model folder>/sensorforge/<run id>/` and says what each file is for. |
| `get_report` | The run's full report (markdown). |
| `list_runs` | Your recent runs. |
| `stop_run` | Stops a queued or running run. |
| `account` | Your plan, this month's usage, and the runs left today. |

What a download holds:

- the model for TensorFlow Lite Micro (`optimized_tflm.tflite` or `model_lossless.tflite`), the file to deploy;
- sometimes the same model as desktop LiteRT loads it, for checking it on a PC;
- `c/`: `model_data.cc` / `model_data.h` (the model as a C array), `model_ops.h` (the operators to register, the
  input and output scale and zero point, the tensor arena size) and `INTEGRATION.md` (how to build it in);
- the report (`report.md` or `lossless.md`).

## Data for optimize

**Images**: a `.zip` of folders, one per class, holding JPEG or PNG files (`person/`, `no_person/`, ...). Folder names
in alphabetical order map to the model's outputs 0, 1, 2, ... To fix the test set yourself, put `train/` and `test/`
folders in the zip, each with the class folders.

**Audio, sensors and anything else**: an `.npz` with `X`, the model's inputs exactly as your front end produces them
(float32, one row per sample, in the model's input shape), and `y`, each sample's class index in the model's output
order. Add `X_test` / `y_test` for a fixed test set. Without `y`, your model's own answers become the labels.

```python
import numpy as np
np.savez("data.npz", X=features.astype("float32"), y=labels.astype("int64"))
```

Detectors, autoencoders and regression models: pass `output="values"`; `y` is not needed. A few hundred samples per
class are enough to measure accuracy; more helps the retraining. `check_data` checks all of this before you upload.

## Limits

- Model files up to 50 MB (`.tflite`, `.onnx`, or a `.zip` of a firmware export). PyTorch and Keras files are not
  accepted: export to `.tflite` or `.onnx` first.
- Data up to 500 MB (`.npz` or `.zip`; a zip may unpack to 2,000 MB).
- Runs per day depend on your plan; `account` shows what is left and when the next run is free.
- Results (report and model) are kept 14 days.
- Uploaded models and data are deleted when the run ends.

## Privacy

- The server reads only the files you name. `check_data` runs on your machine and uploads nothing.
- Your API key is sent only to the SensorForge service, over HTTPS.
- On the service, your uploads and everything made from them are deleted when the run ends; the report and the
  optimized model stay 14 days for you to download, visible to your account only.
- Locally the server keeps a short list of your run ids and model paths in `~/.sensorforge-mcp/runs.json`, so
  downloads land next to the right model. Nothing it downloads is ever run.

## Settings

| Variable | |
|---|---|
| `SENSORFORGE_API_KEY` | Your key (required). |
| `SENSORFORGE_URL` | The service, default `https://sensorforge.dev/optimize`. |
| `SENSORFORGE_MCP_HOME` | Where the run list is kept, default `~/.sensorforge-mcp`. |
| `SENSORFORGE_LOG` | Log level on stderr, default `WARNING`. |

## License

Apache-2.0. Copyright 2026 EdgeForge LLC. See [LICENSE](LICENSE).

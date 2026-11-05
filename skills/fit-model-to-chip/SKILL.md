---
name: fit-model-to-chip
description: Fit a trained .tflite or ONNX model to an edge device with the SensorForge Optimizer. Use when the user asks whether a model fits a chip or board, or wants to reduce its SRAM or flash, quantize it, or deploy it with TensorFlow Lite Micro.
---

Work through these steps with the SensorForge tools, and report the measured figures each one returns:

1. If the user hasn't named a chip, call `list_chips` and ask which one is closest to theirs.
2. Call `check_fit` with the model. It says in seconds whether the model runs on the chip, and how much SRAM and flash it needs.
3. If it doesn't fit, or they want it smaller, call `lossless`. It needs no data and keeps the outputs identical.
4. If that isn't enough and they have data, call `check_data`, which runs locally and uploads nothing. Fix what it reports, then call `optimize` with the accuracy limit they choose.
5. `optimize` returns a run id at once. Call `wait_for_run` until it has a result, which takes 5 to 60 minutes.
6. Call `download` to unpack the result next to the model, and say which file to deploy.

Before an upload, tell the user which files will go to the SensorForge service. Report a rejected or refused run with its reason and what the report suggests doing next. Don't describe a run as a success if it didn't shrink the model: give the before and after figures.

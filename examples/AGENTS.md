## Fitting models to edge devices (SensorForge)

This project uses the SensorForge MCP tools (server `sensorforge`) to make trained models fit our edge devices.
When asked to fit, shrink or deploy a model on a chip, use them in this order:

1. `list_chips` to find the target chip's name (ask if the device is unclear).
2. `check_fit(model_path, [chip])`: read the verdict and the SRAM / flash against the chip's budget.
3. `lossless(model_path, chip)`: less SRAM and flash with outputs identical to the original, no data needed. If this
   already fits with room to spare, stop here.
4. `check_data(model_path, data_path)` and fix every problem it reports. Data is an `.npz` with `X` (the model's
   inputs, float32, one row per sample) and `y` (class indices), or a `.zip` of image folders, one per class.
5. `optimize(model_path, data_path, chip, minimize="ram", accuracy_limit=1.0)`. It returns a run id at once.
6. `wait_for_run(run_id)`, called again each time it says still running (quick runs take about 15 minutes).
7. On PASS, `download(run_id)`. On a refusal, report which limit to loosen instead of guessing.

Rules:

- Pass `.tflite` or `.onnx` files. Export PyTorch (`torch.onnx.export`) or Keras (`tf.lite.TFLiteConverter`) models
  first; never upload `.pt`, `.h5` or `.keras` files.
- Results land in `<model folder>/sensorforge/<run id>/`. Copy what the firmware needs (the `.tflite` or the `c/`
  sources) into the firmware tree, and follow `c/INTEGRATION.md`.
- Report the before and after SRAM, flash and accuracy from the tool output; do not estimate them.
- The API key comes from the `SENSORFORGE_API_KEY` environment variable; never write it into the repository.

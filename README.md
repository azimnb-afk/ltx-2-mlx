# ltx-2-mlx (Local Video Studio runtime export)

This repository is a clean, source-only export of the `ltx-2-mlx` MLX runtime
packages used by [Local Video Studio](https://github.com/azimnb-afk/local-video-studio)
for Mac to run LTX-2.5 (Experimental) generation on Apple Silicon.

It contains exactly two installable packages:

- `packages/ltx-core-mlx` — model library (DiT, VAE, audio, text encoder, conditioning)
- `packages/ltx-pipelines-mlx` — generation pipelines (T2V/I2V, GGUF loading, decode/mux)

Local Video Studio installs both directly from this repository using pip's
Git subdirectory syntax, e.g.:

```bash
pip install "git+https://github.com/azimnb-afk/ltx-2-mlx.git@<commit>#subdirectory=packages/ltx-core-mlx"
pip install "git+https://github.com/azimnb-afk/ltx-2-mlx.git@<commit>#subdirectory=packages/ltx-pipelines-mlx" --no-deps
```

This export intentionally contains no development history, experimental
research scripts, tests with local-machine paths, or other local-machine
paths — it is a single clean commit of the production source tree, matching
the export precedent already used for the Local Video Studio application
itself.

## License

MIT License. See [LICENSE](LICENSE). Original copyright preserved.

## Origin

Ported from and building on [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2/),
via the MLX port originally published as [dgrauet/ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx).

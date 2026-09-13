# HoloLlama

Python wrapper for **llama.cpp** with a live mood-reactive tensor hologram UI, persistent persona memory, and optional web research. Runs on desktop and Termux.

## Features

- Local GGUF inference via `llama-cpp-python`
- NiceGUI interface: chat + terminal-style console + live 3D tensor hologram
- Persona tensor that evolves with conversation and persists across restarts (`holo_memory/`)
- Mood-reactive hologram colors (model self-reports mood as JSON)
- Optional DuckDuckGo research when the query needs current information
- Termux-friendly (CPU builds; smaller quantized models recommended)

## Requirements

- Python 3.10+
- A GGUF model file (place it under `models/`)
- RAM: 8 GB+ desktop recommended; 6 GB+ for small mobile models

## Install (desktop)

```bash
pip install -r requirements.txt
```

GPU (CUDA example):

```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## Run

1. Put a GGUF at `models/your-model.gguf` (or edit `MODEL_PATH` in `app.py`).
2. Start the UI:

```bash
python app.py
```

3. Open http://localhost:8080

Prefix a prompt with `/research` (or include words like `latest` / `current`) to pull web snippets before answering.

## Termux (Android)

Use the F-Droid Termux build.

```bash
pkg update && pkg upgrade -y
pkg install python git cmake clang ninja libandroid-execinfo -y
pip install --upgrade pip
CMAKE_ARGS="-DLLAMA_CMAKE_ARGS=-DGGML_NATIVE=OFF" pip install llama-cpp-python
pip install nicegui duckduckgo-search torch --index-url https://download.pytorch.org/whl/cpu
python app.py
```

Open `http://localhost:8080` in the phone browser. Prefer small Q4/Q5 models (1B–3B class) for usable speed.

## Project layout

```
HoloLlama/
  app.py              # NiceGUI UI (chat, console, live hologram)
  llama_wrapper.py    # llama.cpp wrapper, mood JSON, research, persistence
  requirements.txt
  LICENSE             # MIT
  holo_memory/        # created at runtime (persona tensor + console log)
  models/             # put your .gguf here (not committed)
```

## License

MIT — see [LICENSE](LICENSE).

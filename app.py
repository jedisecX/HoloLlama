import os
import numpy as np
import torch
from nicegui import ui, app
from llama_wrapper import LlamaWrapper

MODEL_PATH = os.environ.get("HOLOLLAMA_MODEL", "models/your-model.gguf")
N_GPU_LAYERS = int(os.environ.get("HOLOLLAMA_GPU_LAYERS", "-1"))

llm = LlamaWrapper(MODEL_PATH, n_gpu_layers=N_GPU_LAYERS)

if "history" not in app.storage.user:
    app.storage.user["history"] = []
if "console_log" not in app.storage.user:
    log_path = "holo_memory/console.log"
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            app.storage.user["console_log"] = f.read()
    else:
        app.storage.user["console_log"] = "=== HoloLlama Persistent Console ===\n"

scene = None
point_cloud = None
current_color = "#00ffff"


def save_console():
    os.makedirs("holo_memory", exist_ok=True)
    with open("holo_memory/console.log", "w", encoding="utf-8") as f:
        f.write(app.storage.user["console_log"])


def update_hologram(mood: str = "neutral"):
    global point_cloud, current_color
    tensor = llm.get_persona_tensors()
    points_3d = torch.pca_lowrank(tensor.unsqueeze(0), q=3)[0].squeeze().numpy()
    points = np.tile(points_3d, (60, 1)) + np.random.normal(0, 0.25, (60, 3))
    current_color = llm.get_mood_color(mood)
    if point_cloud:
        point_cloud.color = current_color
        point_cloud.points = points.tolist()
    else:
        point_cloud = scene.point_cloud(points.tolist(), color=current_color, size=0.07)


def send_message(prompt: str, is_console: bool = False):
    if not prompt.strip():
        return

    app.storage.user["history"].append({"role": "user", "content": prompt})
    if is_console:
        app.storage.user["console_log"] += f"> {prompt}\n"
        console_log.refresh()
        save_console()

    with chat_container:
        ui.chat_message(text=prompt, name="You", avatar="👤")

    with ui.spinner(size="lg"):
        response, mood = llm.chat(prompt)

    app.storage.user["history"].append({"role": "assistant", "content": response})
    if is_console:
        app.storage.user["console_log"] += f"Holo: {response}\nMood: {mood}\n\n"
        console_log.refresh()
        save_console()

    with chat_container:
        ui.chat_message(text=response, name="HoloLlama", avatar="🌀")

    update_hologram(mood)
    ui.notify(f"Mood: {mood.upper()}", color=current_color)


with ui.header().classes("justify-between"):
    ui.label("HoloLlama — Persistent Tensor Hologram").classes("text-2xl font-bold")
    ui.label("Memory: ./holo_memory/").classes("text-sm text-cyan-300")

with ui.row().classes("w-full h-full no-wrap"):
    with ui.column().classes("w-1/2 h-full gap-4 p-4 overflow-auto"):
        chat_container = ui.column().classes("w-full flex-grow")
        for msg in app.storage.user["history"]:
            avatar = "👤" if msg["role"] == "user" else "🌀"
            name = "You" if msg["role"] == "user" else "HoloLlama"
            ui.chat_message(text=msg["content"], name=name, avatar=avatar)

        ui.label("Persistent Console").classes("text-lg mt-6")
        console_log = ui.code(app.storage.user["console_log"]).classes(
            "w-full h-80 font-mono text-xs bg-black text-cyan-300 overflow-auto"
        )

        console_input = ui.input(placeholder="Console prompt...").classes("w-full")
        console_input.on(
            "keydown.enter",
            lambda e: (send_message(console_input.value, True), console_input.set_value("")),
        )

    with ui.column().classes("w-1/2 h-full p-4"):
        scene = ui.scene(width=820, height=620, grid=False).classes("bg-black rounded-3xl shadow-2xl")
        with scene:
            scene.sphere(radius=0.12, color="#112233", opacity=0.4)
            update_hologram()
        ui.button("Refresh Hologram", on_click=lambda: update_hologram()).classes("mt-4")

with ui.footer().classes("bg-transparent p-4"):
    with ui.row().classes("w-full max-w-4xl mx-auto gap-3"):
        main_input = ui.input(placeholder="Message the persistent hologram...").classes("flex-grow")
        main_input.on(
            "keydown.enter",
            lambda: (send_message(main_input.value), main_input.set_value("")),
        )
        ui.button(
            "Send",
            on_click=lambda: (send_message(main_input.value), main_input.set_value("")),
        ).props("icon=send")


def animate():
    if scene and point_cloud:
        scene.run_method("rotate", 0.4, 0.25, 0.1)


ui.timer(0.08, animate)
ui.run(title="HoloLlama Persistent", port=8080, reload=True, storage_secret="holo-secret")

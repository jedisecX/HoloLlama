import os
import json
import re
import torch
from llama_cpp import Llama
from duckduckgo_search import DDGS


class LlamaWrapper:
    def __init__(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 8192):
        self.model_path = model_path
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
            chat_format="chatml",
        )
        self.memory_dir = "holo_memory"
        os.makedirs(self.memory_dir, exist_ok=True)
        self.persona_tensor = None
        self.load_memory()

    def load_memory(self):
        tensor_path = os.path.join(self.memory_dir, "persona_tensor.pt")
        if os.path.exists(tensor_path):
            self.persona_tensor = torch.load(tensor_path, weights_only=True)
            print("Loaded persistent persona tensor")
        else:
            print("First run — creating initial persona tensor")

    def save_memory(self):
        if self.persona_tensor is not None:
            torch.save(self.persona_tensor, os.path.join(self.memory_dir, "persona_tensor.pt"))

    def chat(self, prompt: str, research_mode: bool = True):
        context = ""
        if research_mode or re.search(r"/research|search|latest|current|news", prompt, re.I):
            query = re.sub(r"/research", "", prompt).strip()
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=3))
                context = "\n\n=== WEB RESEARCH ===\n" + "\n\n".join(
                    [
                        f"Source: {r['title']}\n{r['body'][:300]}...\nURL: {r['href']}"
                        for r in results
                    ]
                )
            except Exception as e:
                context = f"\n\n(Web search failed: {e})"

        system = f"You are a helpful assistant with persistent memory. {context}"
        raw = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": prompt
                    + '\n\nReply in JSON: {"content": "answer", "mood": "one-word-mood"}',
                },
            ],
            max_tokens=600,
            temperature=0.35,
        )["choices"][0]["message"]["content"]

        try:
            parsed = json.loads(raw)
            content = parsed.get("content", raw)
            mood = parsed.get("mood", "neutral").lower()
        except Exception:
            content, mood = raw, "neutral"

        tokens = self.llm.tokenizer.encode(content)
        new_emb = torch.tensor(self.llm.embed(tokens)).mean(dim=0)

        if self.persona_tensor is None:
            self.persona_tensor = new_emb
        else:
            self.persona_tensor = 0.85 * self.persona_tensor + 0.15 * new_emb

        self.save_memory()
        return content, mood

    def get_persona_tensors(self):
        if self.persona_tensor is None:
            self.chat("Describe your core identity and purpose.", research_mode=False)
        return self.persona_tensor

    def get_mood_color(self, mood: str) -> str:
        mood_map = {
            "excited": "#ff00ff",
            "happy": "#00ffaa",
            "curious": "#00ffff",
            "calm": "#0088ff",
            "serious": "#ffaa00",
            "concerned": "#ff4400",
            "neutral": "#aaaaaa",
            "creative": "#aa00ff",
            "reflective": "#7744ff",
        }
        return mood_map.get(mood, "#00ffff")

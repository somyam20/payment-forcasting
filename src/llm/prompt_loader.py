import yaml
import os

CONFIG_PATH = os.path.join("config", "prompt_templates.yaml")

class PromptLoader:
    def __init__(self):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)

    def get(self, key: str):
        return self.prompts.get(key, "")
        

prompt_loader = PromptLoader()

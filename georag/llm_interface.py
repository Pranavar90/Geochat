import ollama
from . import config

class LLMInterface:
    def __init__(self, model_name=config.LLM_MODEL_NAME):
        self.model_name = model_name

    def generate_response(self, prompt, stream=False):
        """Generates a response from the LLM."""
        try:
            return ollama.chat(model=self.model_name, messages=[{'role': 'user', 'content': prompt}], stream=stream)
        except Exception as e:
            return f"Error: {e}"

    def check_model(self):
        """Checks if the model is available."""
        try:
            ollama.show(self.model_name)
            return True
        except:
            return False

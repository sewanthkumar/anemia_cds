import os
import json
import base64
import requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """
    Unified LLM interface. Supports Groq (cloud) and Ollama (local).
    Switch backend by calling set_backend() or changing config.
    All methods have identical signatures regardless of backend.
    """

    GROQ_TEXT_MODEL   = 'llama-3.3-70b-versatile'
    GROQ_VISION_MODEL = 'meta-llama/llama-4-scout-17b-16e-instruct'
    OLLAMA_TEXT_MODEL = 'llama3.2'
    OLLAMA_VISION_MODEL = 'llava'
    OLLAMA_BASE_URL   = 'http://localhost:11434'

    def __init__(self, backend: str = 'groq'):
        """
        backend: 'groq' or 'ollama'
        """
        self.backend = backend
        self._groq_client = None
        self._init_backend()

    def _init_backend(self):
        if self.backend == 'groq':
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                raise ValueError('GROQ_API_KEY not found in .env file. Get one free at https://console.groq.com')
            self._groq_client = Groq(api_key=api_key)
        elif self.backend == 'ollama':
            self._verify_ollama()

    def _verify_ollama(self):
        try:
            r = requests.get(f'{self.OLLAMA_BASE_URL}/api/tags', timeout=3)
            if r.status_code != 200:
                raise ConnectionError('Ollama is not running. Start it with: ollama serve')
        except requests.exceptions.ConnectionError:
            raise ConnectionError('Ollama not found at localhost:11434. Run: ollama serve')

    def set_backend(self, backend: str, model: str = None):
        """Switch backend at runtime without restarting."""
        self.backend = backend
        if model:
            if backend == 'groq':
                self.GROQ_TEXT_MODEL = model
            else:
                self.OLLAMA_TEXT_MODEL = model
        self._init_backend()

    def chat(self, prompt: str, system: str = None, temperature: float = 0.2, max_tokens: int = 1000) -> str:
        """Send a text prompt. Returns response string. Same API for Groq and Ollama."""
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})

        if self.backend == 'groq':
            response = self._groq_client.chat.completions.create(
                model=self.GROQ_TEXT_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        elif self.backend == 'ollama':
            payload = {
                'model': self.OLLAMA_TEXT_MODEL,
                'messages': messages,
                'stream': False,
                'options': {'temperature': temperature, 'num_predict': max_tokens}
            }
            r = requests.post(f'{self.OLLAMA_BASE_URL}/api/chat', json=payload, timeout=120)
            return r.json()['message']['content']

    def chat_json(self, prompt: str, system: str = None, temperature: float = 0.1) -> dict:
        """Send prompt, parse response as JSON. Strips markdown fences automatically."""
        raw = self.chat(prompt, system=system, temperature=temperature, max_tokens=1000)
        clean = raw.strip()
        if clean.startswith('```'):
            clean = clean.split('```')[1]
            if clean.startswith('json'):
                clean = clean[4:]
        clean = clean.strip().rstrip('`')
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', clean, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f'Could not parse JSON from LLM response: {raw[:200]}')

    def vision(self, image_path: str, prompt: str) -> str:
        """Send image + prompt. Uses vision-capable model. Returns response string."""
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        ext = image_path.lower().split('.')[-1]
        mime = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png'}.get(ext, 'image/jpeg')

        if self.backend == 'groq':
            response = self._groq_client.chat.completions.create(
                model=self.GROQ_VISION_MODEL,
                messages=[{
                    'role': 'user',
                    'content': [
                        {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{image_data}'}},
                        {'type': 'text', 'text': prompt}
                    ]
                }],
                max_tokens=500
            )
            return response.choices[0].message.content

        elif self.backend == 'ollama':
            payload = {
                'model': self.OLLAMA_VISION_MODEL,
                'prompt': prompt,
                'images': [image_data],
                'stream': False
            }
            r = requests.post(f'{self.OLLAMA_BASE_URL}/api/generate', json=payload, timeout=120)
            return r.json()['response']

    def get_status(self) -> dict:
        """Returns current backend info for display in UI status bar."""
        if self.backend == 'groq':
            return {
                'backend': 'Groq Cloud ☁',
                'text_model': self.GROQ_TEXT_MODEL,
                'vision_model': self.GROQ_VISION_MODEL,
                'status': 'connected'
            }
        else:
            models_r = requests.get(f'{self.OLLAMA_BASE_URL}/api/tags', timeout=3)
            installed = [m['name'] for m in models_r.json().get('models', [])]
            return {
                'backend': 'Ollama Local 🖥',
                'text_model': self.OLLAMA_TEXT_MODEL,
                'vision_model': self.OLLAMA_VISION_MODEL,
                'installed_models': installed,
                'status': 'connected'
            }

from openai import OpenAI

from .base import BaseLLM, LLMS


@LLMS.register("openai")
class OpenAILLM(BaseLLM):
    """需要環境變數 OPENAI_API_KEY"""

    def __init__(self, model: str = "gpt-4o-mini", **_):
        self.client = OpenAI()
        self.model = model

    def generate(self, prompt, system=""):
        resp = self.client.chat.completions.create(
            model=self.model, messages=self._messages(prompt, system))
        return resp.choices[0].message.content

    def stream(self, prompt, system=""):
        resp = self.client.chat.completions.create(
            model=self.model, messages=self._messages(prompt, system), stream=True)
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

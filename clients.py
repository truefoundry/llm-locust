import json
import logging
import random

from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class BaseModelClient:
    """Base Model Client definition"""

    def ping_url(self) -> str:
        """return Health check URL"""
        raise NotImplementedError

    def get_request_params(self) -> tuple[str, dict, dict, dict]:
        """Returns request url, headers, data, prompt dictionary.
        Should not have blocking code"""
        raise NotImplementedError

    def reset_parse_state(self) -> None:
        """Reset any internal parsing state between requests"""
        pass

    def parse_response(self, chunk: bytes) -> list[int]:
        """Parse response bytes into list of tokens"""
        raise NotImplementedError


class OpenAIChatStreamingClient(BaseModelClient):
    """OpenAI Chat Completions compatible streaming client"""

    def __init__(
        self,
        base_url: str,
        prompts: list[str],
        system_prompt: str | None,
        openai_model_name: str,
        tokenizer: PreTrainedTokenizerBase,
        max_tokens: int,
        seed: int = 0,
        ignore_eos: bool = False,
        openai_api_key: str | None = None,
    ):
        self.base_url = base_url
        self.prompts = prompts
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.openai_model_name = openai_model_name
        self.tokenizer = tokenizer
        self.ignore_eos = ignore_eos
        self.openai_api_key = openai_api_key
        self.buffer = ""
        random.seed(seed)

    def ping_url(self) -> str:
        return f"{self.base_url}/health"

    def get_request_params(self) -> tuple[str, dict, dict, dict]:
        prompt = random.choice(self.prompts)
        headers = {
            "Content-Type": "application/json",
        }
        if self.openai_api_key:
            headers.update(
                {
                    "Authorization": f"Bearer {self.openai_api_key}",
                }
            )
        if "chat/completions" in self.base_url:
            url = self.base_url
        else:
            url = f"{self.base_url}/v1/chat/completions"
        data = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt["prompt"]},
            ]
            if self.system_prompt is not None
            else [{"role": "user", "content": prompt["prompt"]}],
            "max_tokens": self.max_tokens,
            "stream": True,
            "ignore_eos": self.ignore_eos,
        }
        if self.openai_model_name is not None:
            data["model"] = self.openai_model_name
        return url, headers, data, prompt

    def reset_parse_state(self) -> None:
        self.buffer = ""

    def parse_response(self, chunk: bytes) -> list[int]:
        self.buffer += chunk.decode("utf-8", errors="replace")
        output = []

        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.strip()

            if not line:
                continue
            if len(line.split(":", 1)) == 2:
                line = line.split(":", 1)[1].strip()
                if line == "[DONE]":
                    continue
                try:
                    data = json.loads(line)
                    choices = data.get("choices")
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    text = delta.get("content")
                    if text is not None:
                        output += self.tokenizer.encode(
                            text, add_special_tokens=False
                        )
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.warning(
                        "Error while parsing chunk: %s, error: %s", line[:200], e
                    )
                    continue

        return output

from openai import OpenAI, AsyncOpenAI

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.openai.base import BaseOpenAIClient


class OpenAIClient(BaseOpenAIClient):
    """
    Client for OpenAI Chat Completions API using the official SDK.
    """

    def __init__(
        self,
        endpoint: str,
        model_name: str,
        system_prompt: str,
        api_key: str,
    ):
    super().__init__(
        endpoint=endpoint,
        model_name=model_name,
        system_prompt=system_prompt,
        api_key=api_key,
        )

    self._client = OpenAI(api_key=api_key)
    self._async_client = AsyncOpenAI(api_key=api_key)

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response from OpenAI's Chat Completions API.
        """

        message_history = [
            {
                "role": "system",
                "content": self._system_prompt,
            }
        ]

        message_history.extend(
            {
                "role": message.role.value,
                "content": message.content,
            }
            for message in messages
        )

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=message_history,
            **kwargs,
        )

        content = response.choices[0].message.content

        print(content)

        return Message(
            role=Role.ASSISTANT,
            content=content,
        )

    async def stream_response(
        self,
        messages: list[Message],
        **kwargs,
    ) -> Message:
        """
        Get a streaming response from OpenAI's Chat Completions API.
        """

        message_history = [
            {
                "role": "system",
                "content": self._system_prompt,
            }
        ]

        message_history.extend(
            {
                "role": message.role.value,
                "content": message.content,
            }
            for message in messages
        )

        stream = await self._async_client.chat.completions.create(
            model=self._model_name,
            messages=message_history,
            stream=True,
            **kwargs,
        )

        full_response = ""

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content

        print()

        return Message(
            role=Role.ASSISTANT,
            content=full_response,
        )

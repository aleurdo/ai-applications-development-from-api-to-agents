from openai import OpenAI, AsyncOpenAI

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.openai.base import BaseOpenAIClient


class OpenAIClient(BaseOpenAIClient):
    """
    Client for OpenAI Chat Completions API using the official SDK.

    This implementation uses the official OpenAI Python library to interact
    with the Chat Completions API, providing both synchronous and streaming
    response capabilities.
    """

    def __init__(
        self,
        endpoint: str,
        model_name: str,
        system_prompt: str,
        api_key: str,
    ):
        """
        Initialize the OpenAI Chat Completions client with SDK.
        """

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

        # Prepare message history with system prompt
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

        # Call client
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=message_history,
            **kwargs,
        )

        # Print response
        content = response.choices[0].message.content
        print(content)

        # Return ASSISTANT message
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

        # Prepare message history with system prompt
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

        # Call client with streaming mode
        stream = await self._async_client.chat.completions.create(
            model=self._model_name,
            messages=message_history,
            stream=True,
            **kwargs,
        )

        # Handle stream with chunks
        full_response = ""

        # Print response to console
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content

        print()

        # Return ASSISTANT message
        return Message(
            role=Role.ASSISTANT,
            content=full_response,
        )

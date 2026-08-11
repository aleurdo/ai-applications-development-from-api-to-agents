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

    Attributes:
        _client (OpenAI): Synchronous OpenAI client instance.
        _async_client (AsyncOpenAI): Asynchronous OpenAI client instance.
        Inherits all other attributes from BaseOpenAIClient.
    """

    def __init__(self, endpoint: str, model_name: str, system_prompt: str, api_key: str):
        """
        Initialize the OpenAI Chat Completions client with SDK.

        Args:
            endpoint (str): The OpenAI API endpoint (for compatibility, not used by SDK).
            model_name (str): The OpenAI model to use (e.g., 'gpt-5').
            system_prompt (str): The system message to guide the model's behavior.
            api_key (str): The OpenAI API key for authentication.
        """
        #TODO:
        # Call to __init__ of super class
        super().__init__( 
            endpoint=endpoint, 
            model_name=model_name,
            system_prompt=system_prompt,
            api_key=api_key, )
        # Add OpenAI and AsyncOpenAI clients https://github.com/openai/openai-python?tab=readme-ov-file#usage
        self._client = OpenAI(api_key=api_key) 
        self._async_client = AsyncOpenAI(api_key=api_key)
        # (In readme you can find samples with both of these clients)
        
        # Useful link with request/response samples https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
        raise NotImplementedError

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response from OpenAI's Chat Completions API.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The AI's response message.

        Note:
            The system prompt is automatically prepended to the messages.
            The response is printed to stdout before being returned.
        """
        #TODO:
        # - Prepare message history with System prompt
        message_history = [
            { "role": "system", 
             "content": self._system_prompt, } ]
        message_history.extend( { "role": message.role.value, 
                                 "content": message.content, } for message in messages )
        # - Call client
        response = self._client.chat.completions.create( model=self._model_name,
                                                        messages=message_history, **kwargs, )
        # - Print response to console
        content = response.choices[0].message.content 
        print(content)
        # - Return ASSISTANT message

        return Message( role=Role.ASSISTANT, content=content, ) 
        async def stream_response(self, messages: list[Message], **kwargs) -> Message: """ Get a streaming response from OpenAI's Chat Completions API. """
        raise NotImplementedError

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response from OpenAI's Chat Completions API.

        The response is streamed token-by-token, with each chunk printed
        immediately as it arrives.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            The system prompt is automatically prepended to the messages.
            Each token is printed to stdout as it arrives for real-time display.
        """
        #TODO:
        # - Prepare message history with System prompt
        message_history = [
            { "role": "system", "content": self._system_prompt, } ] 
        message_history.extend( { "role": message.role.value, 
                                 "content": message.content, } for message in messages )
        # - Call client with streaming mode
        stream = await self._async_client.chat.completions.create( model=self._model_name,
                                                                  messages=message_history, 
                                                                  stream=True, 
                                                                  **kwargs, )
        # - Handle stream with chunks
        
        # - Print response to console
        full_response = "" async for chunk in stream: if chunk.choices and chunk.choices[0].delta.content: 
            content = chunk.choices[0].delta.content 
            print(content, end="", flush=True) 
            full_response += content print()
        # - Return ASSISTANT message
        return Message( role=Role.ASSISTANT, content=full_response, )
        

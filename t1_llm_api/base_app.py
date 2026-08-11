from commons.models.conversation import Conversation
from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


async def start(stream: bool, client: AIClient) -> None:
    """
    Start an interactive chat session with an AI client.

    This function runs a continuous loop that:
    1. Prompts the user for input
    2. Sends the conversation history to the AI
    3. Displays the AI's response
    4. Maintains conversation context

    The loop continues until the user types 'exit'.
    """

    conversation = Conversation()

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        user_message = Message(
            role=Role.USER,
            content=user_input,
        )

        conversation.add_message(user_message)

        if stream:
            response = await client.stream_response(
                conversation.get_messages()
            )
        else:
            response = client.response(
                conversation.get_messages()
            )
            print(f"AI: {response.content}")

        conversation.add_message(response)

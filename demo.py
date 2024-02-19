import chainlit as cl

@cl.step
def tool():
    return "Response from the tool!"

@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: cl.Message):

    # Call the tool
    tool()

    # Send the final answer.
    await cl.Message(content="This is the final answer").send()
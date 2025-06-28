import asyncio
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image as AGImage
from PIL import Image
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from pydantic import BaseModel, Field

load_dotenv(override=True)

model = "gpt-4.1-nano"


class ImageDescription(BaseModel):
    scene: str = Field(description="Scene of the image very briefly")
    mood: str = Field(description="Mood of the image")
    country: str = Field(description="Guessed country in the image")
    year: str = Field(description="Guessed year in the image")


client = OpenAIChatCompletionClient(model=model)

agent = AssistantAgent(
    name="image_describer",
    model_client=client,
    system_message="Describe image",
    output_content_type=ImageDescription)

img = AGImage(Image.open('image.jpeg'))
message = MultiModalMessage(content=["Describe the content of the image", img], source="User")


async def main():
    response = await agent.on_messages([message], cancellation_token=CancellationToken())
    description = response.chat_message.content
    print(f"Scene: {description.scene}")
    print(f"Mood: {description.mood}")
    print(f"Country: {description.country}")
    print(f"Year: {description.year}")


asyncio.run(main())

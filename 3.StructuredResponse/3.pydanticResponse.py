import json

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")


# Schema
class Schema(BaseModel):
    key_themes: list[str] = Field(description="must write down all the key themes discussed in the review in the list")
    summary: str = Field(description="must write down a brief summary of the review")
    sentiment: Literal["Pos", "Neg"] = Field(
        description="must return sentiment of the review , either positive or negative")
    pros: Optional[list[str]] = Field(description="write down all the pros inside a list")
    cons: Optional[list[str]] = Field(description="write down all the cons inside a list")
    name: Optional[str] = Field(description="Write down the name of the reviewer", default=None)


prompt = ("""
The Google Pixel series stands out in the smartphone market by offering a highly responsive, stock Android software experience directly from Google, deeply integrated with advanced Gemini AI features like live translation, smart call screening, and contextual assistance. Its primary strength lies in its exceptional camera system, which leverages industry-leading computational photography to deliver stunning low-light images via Night Sight, sharp long-range zoom, and intuitive editing tools like Magic Eraser. Users love the clean, bloatware-free user interface, vibrant high-refresh-rate displays, and Google's industry-leading commitment to up to seven years of software updates, ensuring long-term device longevity. However, the lineup does face notable drawbacks, as its charging speeds remain relatively slow compared to competing Android flagships, and the battery life under heavy use can be average. Additionally, while the customized Tensor processors excel at handling AI tasks and daily efficiency, they occasionally fall short in raw processing power and thermal management during intensive tasks like high-end gaming when compared to rival chips.""")

structured_model = chat_model.with_structured_output(Schema, strict=True)
response = structured_model.invoke(prompt)
# print(json.dumps(response, indent=4))
print(response)
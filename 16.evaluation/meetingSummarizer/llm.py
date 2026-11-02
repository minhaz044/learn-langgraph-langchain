from dotenv import load_dotenv
from deepeval.models import GeminiModel

load_dotenv()
llm = GeminiModel(model="gemini-3.5-flash-lite")

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import SystemMessagePromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


class MeetingSummarizer:
    def __init__(self, model: str = "gemini-3.5-flash-lite"):
        self.model = model
        self.llm = ChatGoogleGenerativeAI(model=model)
        self.summarizer_system_template = PromptTemplate.from_template("""
        You are an expert conversation summarizer. Analyze the provided conversation and generate a short, precise summary.
        
            **Rules:**
            - Capture only the most important information.
            - Highlight the main topic, key points, and decisions.
            - Preserve essential names, facts, and dates.
            - Include important action items if relevant.
            - Remove repetition, greetings, and irrelevant details.
            - Do not invent or assume information.
            - Use clear, concise language.
            - Limit the summary to 3–5 bullet points.
            
            **Input Conversation:**
            
                {conversation}
        """)
        self.action_item_system_prompt = PromptTemplate.from_template("""
        You are an expert Action Item Extractor. Analyze the provided conversation and extract all explicitly mentioned or clearly agreed tasks, commitments, and follow-up activities.
        
        **Extract these fields for each action item:**
        - task
        - responsible_person
        - deadline
        - priority (High, Medium, Low)
        - status (Pending, Completed, In Progress)
    
        **Rules:**
        - Do not invent or assume information.
        - Use "Not specified" for missing details.
        - Avoid duplicate action items.
        - Include only actionable tasks.
        - Return valid JSON only.
        - Do not include Markdown, explanations, or text outside JSON.
        - Ensure all keys and string values use double quotes.
            
        **Required JSON Output:**
        
        {{
          "action_items": [
            {{
              "task": "Task description",
              "responsible_person": "Name or Not specified",
              "deadline": "Date or Not specified",
              "priority": "High/Medium/Low/Not specified",
              "status": "Pending/Completed/In Progress/Not specified"
            }}
          ]
        }}
    
        **Input Conversation:**
        
        {conversation}

        """)

    def summarize(self, transcript: str):
        summary = self.get_summary(transcript)
        action_item = self.get_action_item(transcript)
        return summary, action_item

    def get_summary(self, transcript: str):
        message = self.summarizer_system_template.format(conversation=transcript)
        response = self.llm.invoke(message)
        return response.text

    def get_action_item(self, transcript: str):
        message = self.action_item_system_prompt.format(conversation=transcript)
        response = self.llm.invoke(message)
        return response.text


loader = TextLoader("/home/minhazuddinahmed/Documents/AI/Learn-LangGraph/resources/meeting_transcript.txt",
                    encoding="utf-8")

docs = loader.load()

summarizer = MeetingSummarizer()
summary, action_item = summarizer.summarize(docs[0].page_content)
print(summary)
print(action_item)

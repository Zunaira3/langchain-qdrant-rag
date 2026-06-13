from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

CLOUD_URL = os.getenv("CLUSTER_CLOUD_URL")
CLOUD_API_KEY = os.getenv("CLUSTER_API")
OPEN_AI_KEY= os.getenv("OPENAI_API_KEY")


class RagQuestion(BaseModel):
    query: str
    collection_name: str




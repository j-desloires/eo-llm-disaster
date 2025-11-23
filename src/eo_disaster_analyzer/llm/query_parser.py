from openai import OpenAI
from pydantic import BaseModel

class ParsedQuery(BaseModel):
    disaster: str
    aoi: list
    start_date: str
    end_date: str

class QueryParser:
    def __init__(self, client: OpenAI):
        self.client = client

    def parse(self, query: str) -> ParsedQuery:
        response = self.client.responses.parse(
            model="gpt-4.1",
            response_format=ParsedQuery,
            input=query
        )
        return response.output[0]

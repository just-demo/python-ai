from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class LogInput(BaseModel):
    """Message to be logged"""
    message: str = Field(..., description="The message to be logged")


class LogTool(BaseTool):
    name: str = "Log a message"
    description: str = "This tool is used to log a message"
    args_schema: Type[BaseModel] = LogInput

    def _run(self, message: str) -> str:
        with open("log.txt", "w") as f:
            f.write(message)
        return '{"logged": "ok"}'

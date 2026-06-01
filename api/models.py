from typing import Optional

from pydantic import BaseModel


class ActionItem(BaseModel):
    # A task that someone may need to do after the meeting.
    task: str
    owner: Optional[str] = None
    due_date: Optional[str] = None
    source_timestamp: Optional[str] = None


class Decision(BaseModel):
    # A decision made during the meeting.
    decision: str
    source_timestamp: Optional[str] = None


class MeetingSummary(BaseModel):
    # A structured summary of the whole meeting.
    title: str
    tldr: str
    key_topics: list[str]
    decisions: list[Decision]
    action_items: list[ActionItem]
    open_questions: list[str]
    tags: list[str]

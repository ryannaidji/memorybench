"""Data models shared by the extractor, the store and the API."""
from enum import Enum

from pydantic import BaseModel, Field


class Category(str, Enum):
    contact_preference = "contact_preference"  # how/when the customer wants to be reached
    plan = "plan"                              # subscription tier / product owned
    issue = "issue"                            # an open or past problem
    preference = "preference"                  # language, format, product preferences
    profile = "profile"                        # stable personal/business facts (role, company size, location)


class Fact(BaseModel):
    category: Category
    value: str = Field(description="Short, self-contained statement of the fact, max ~15 words")


class FactList(BaseModel):
    facts: list[Fact]


class Turn(BaseModel):
    role: str  # "customer" | "agent"
    text: str


class Conversation(BaseModel):
    customer_id: str
    turns: list[Turn]


class StoredFact(Fact):
    id: int
    customer_id: str
    created_at: str


class RecallHit(BaseModel):
    fact: StoredFact
    score: float

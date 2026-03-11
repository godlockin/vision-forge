"""Core data models for the expert system."""

from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, List, Any
from enum import Enum


class Archetype(str, Enum):
    """Expert archetype."""
    STATIC = "static"
    DYNAMIC = "dynamic"


class LoadStrategy(str, Enum):
    """Expert loading strategy."""
    ALWAYS = "always"
    ON_DEMAND = "on_demand"
    DYNAMIC = "dynamic"


class TaskStatus(str, Enum):
    """Task status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Persona(BaseModel):
    """Expert persona configuration."""
    description: str
    background: str
    personality: str


class Capability(BaseModel):
    """Expert capability."""
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class IOSpec(BaseModel):
    """Input/Output specification."""
    input: Dict[str, str]
    output: Dict[str, str]


class DecisionStyle(BaseModel):
    """Decision making style."""
    risk_tolerance: Literal["very_low", "low", "medium", "high"]
    consensus_need: Literal["none", "low", "medium", "high"]


class MemoryConfig(BaseModel):
    """Memory configuration."""
    scope: Literal["task", "session", "persistent"]
    retention: Literal["task", "session", "persistent", "permanent"]


class ExpertConfig(BaseModel):
    """Expert configuration model."""
    id: str
    role: str
    archetype: Archetype
    load_strategy: LoadStrategy = LoadStrategy.ALWAYS
    persona: Persona
    thinking_framework: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    blind_spots: List[str] = Field(default_factory=list)
    superhuman_insights: List[str] = Field(default_factory=list)
    capabilities: List[Capability] = Field(default_factory=list)
    i_o_spec: Optional[IOSpec] = None
    decision_style: Optional[DecisionStyle] = None
    memory: Optional[MemoryConfig] = None

    # Optional fields for specific experts
    trigger_conditions: Optional[List[str]] = None
    veto_power: Optional[Dict[str, Any]] = None
    split_behavior: Optional[Dict[str, Any]] = None


class Task(BaseModel):
    """Task representation."""
    id: str
    type: str
    status: TaskStatus = TaskStatus.PENDING
    data: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: int
    updated_at: int


class Job(BaseModel):
    """Job representation for async operations."""
    id: str
    status: TaskStatus
    action: str
    data: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: int
    updated_at: int

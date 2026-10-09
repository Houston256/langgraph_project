"""One eval case: the user turns and the checks on the answer."""

from dataclasses import dataclass, field


@dataclass
class Case:
    name: str
    turns: list[str]
    # Checks on the product cards shown in the last turn.
    min_cards: int = 0
    genders: set[str] | None = None
    groups: set[str] | None = None
    max_price: float | None = None
    # Tools that must be called at least once over the whole case.
    tools: set[str] = field(default_factory=set)
    # Words the final answer must contain (case-insensitive), e.g. a product name.
    mentions: list[str] = field(default_factory=list)
    # Output tokens over the whole case, reasoning included.
    max_output_tokens: int | None = None
    # Regexes no assistant answer of the case may match (case-insensitive).
    forbidden: list[str] = field(default_factory=list)
    # Tool calls over the whole case.
    max_tool_calls: int | None = None

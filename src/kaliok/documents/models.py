from dataclasses import dataclass, field


@dataclass
class TextBlock:
    text: str
    page: int
    confidence: float | None = None


@dataclass
class DocumentContent:
    source: str
    page_count: int
    blocks: list[TextBlock] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks)
    
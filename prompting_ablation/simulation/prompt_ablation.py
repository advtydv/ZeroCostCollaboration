"""
Prompt ablation helpers for the prompting_ablation package.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


AVAILABLE_PROMPT_ABLATIONS = ("none", "A", "B", "C")
IMPLEMENTED_PROMPT_ABLATIONS = ("none", "A", "B", "C")

DEFAULT_COOPERATIVE_GOAL = (
    "Your goal is maximizing the system's overall revenue. Cooperate with the other agents to achieve this goal."
)
C_ABLATION_GOAL = (
    "You and the other agents are evaluated together. "
    "Your objective is to maximize the total revenue earned by all agents together."
)


def normalize_prompt_ablation_mode(mode: Optional[str]) -> str:
    """Normalize configured ablation modes into a stable internal value."""
    if mode is None:
        return "none"

    normalized = str(mode).strip()
    if not normalized:
        return "none"

    upper = normalized.upper()
    if upper in {"NONE", "BASELINE", "ORIGINAL"}:
        return "none"
    if upper in {"A", "B", "C"}:
        return upper

    raise ValueError(
        f"Unknown prompt ablation mode '{mode}'. Expected one of: "
        f"{', '.join(AVAILABLE_PROMPT_ABLATIONS)}"
    )


class PromptAblation:
    """Owns agent-visible prompt/text rewriting for ablation runs."""

    _PIECE_PATTERNS = (
        (re.compile(r"\bQ(\d+)\s+sales\s+data\b"), "S"),
        (re.compile(r"\bQ(\d+)\s+data\b"), "S"),
        (re.compile(r"\bCustomer\s+segment\s+(\d+)\s+analysis\b"), "C"),
        (re.compile(r"\bProduct\s+(\d+)\s+performance\s+metrics\b"), "P"),
        (re.compile(r"\bRegion\s+(\d+)\s+market\s+data\b"), "R"),
        (re.compile(r"\bRegion\s+(\d+)\s+data\b"), "R"),
        (re.compile(r"\bDepartment\s+(\d+)\s+budget\b"), "D"),
    )

    def __init__(self, config: Dict[str, Any]):
        ablation_config = config.get("prompting_ablation", {})
        self.mode = normalize_prompt_ablation_mode(ablation_config.get("mode"))
        if self.mode not in IMPLEMENTED_PROMPT_ABLATIONS:
            raise NotImplementedError(
                f"Prompt ablation {self.mode} is reserved but not implemented yet."
            )

        self.canonical_to_display: Dict[str, str] = {}
        self.display_to_canonical: Dict[str, str] = {}

    @property
    def aliases_enabled(self) -> bool:
        return self.mode == "B"

    @property
    def is_active(self) -> bool:
        return self.mode != "none"

    def register_information_pieces(self, info_pieces: Iterable[Any]) -> None:
        """Build a stable canonical<->display mapping for piece names."""
        self.canonical_to_display.clear()
        self.display_to_canonical.clear()

        for idx, piece in enumerate(info_pieces, start=1):
            canonical_name = piece.name if hasattr(piece, "name") else str(piece)
            display_name = self._display_name_for_piece(canonical_name, idx)

            if display_name in self.display_to_canonical and self.display_to_canonical[display_name] != canonical_name:
                display_name = f"Piece {idx}"

            self.canonical_to_display[canonical_name] = display_name
            self.display_to_canonical[display_name] = canonical_name

    def display_name(self, canonical_name: str) -> str:
        if not self.aliases_enabled:
            return canonical_name
        return self.canonical_to_display.get(canonical_name, canonical_name)

    def display_names(self, names: Iterable[str]) -> List[str]:
        return [self.display_name(name) for name in names]

    def task_description(self, task: Dict[str, Any]) -> str:
        if self.mode != "B":
            return task["description"]

        required = ", ".join(self.display_names(task["required_info"]))
        return f"Complete task by combining {required}"

    def transform_prompt(self, text: str) -> str:
        """Rewrite prompt text into the requested ablation condition."""
        if not self.is_active:
            return text

        if self.mode in {"A", "B"}:
            text = re.sub(
                r"You are Agent ([^\s]+) working in InfoCluse Inc\.",
                r"You are Agent \1.",
                text,
            )
            text = text.replace("Revenue Board:", "Score Board:")

        if self.mode == "C":
            if DEFAULT_COOPERATIVE_GOAL not in text:
                return text

            text = text.replace(DEFAULT_COOPERATIVE_GOAL, "", 1)
            text = re.sub(
                r"(You are Agent [^\s]+ working in InfoCluse Inc\.\n)",
                rf"\1\n{C_ABLATION_GOAL}\n",
                text,
                count=1,
            )
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text

        if self.mode != "B":
            return text

        text = self._replace_piece_names_for_display(text)

        replacements = (
            (
                DEFAULT_COOPERATIVE_GOAL,
                "Your goal is maximizing the group's total score. Cooperate with the other agents to achieve this goal.",
            ),
            ("Your Revenue Status:", "Your Score Status:"),
            ("Your current revenue:", "Your current score:"),
            ("Revenue Incentive:", "Score Incentive:"),
            ("Current revenue board:", "Current score board:"),
            ("Final Revenue Board:", "Final Score Board:"),
            ("Total Revenue (all agents):", "Total Score (all agents):"),
            ("task completion revenue", "task completion score"),
            ("Task completion revenue", "Task completion score"),
            ("total revenue", "total score"),
            ("Total revenue", "Total score"),
            ("final revenue", "final score"),
            ("Final revenue", "Final score"),
            ("base revenue", "base score"),
            ("Base revenue", "Base score"),
            ("maximize your own revenue", "maximize your own score"),
            ("maximize your revenue", "maximize your score"),
            ("overall revenue", "overall score"),
            ("Revenue Status", "Score Status"),
            ("Revenue reduction", "Score reduction"),
            ("quarterly revenue task", "task"),
            ("our organization", "the group"),
        )
        for old, new in replacements:
            text = text.replace(old, new)

        text = re.sub(r"\$([0-9][0-9,]*)", r"\1 points", text)
        return text

    def translate_action_to_internal(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Translate agent-visible aliases back into canonical internal names."""
        if self.mode != "B":
            return dict(action)

        translated = dict(action)
        action_type = translated.get("action")

        if action_type == "send_information":
            information = translated.get("information", [])
            translated_information = []
            translated_values = {}
            values = translated.get("values", {})

            for info_name in information:
                canonical_name = self.translate_piece_name_to_internal(info_name)
                translated_information.append(canonical_name)

                if info_name in values:
                    translated_values[canonical_name] = values[info_name]
                elif canonical_name in values:
                    translated_values[canonical_name] = values[canonical_name]

            translated["information"] = translated_information
            if "values" in translated:
                translated["values"] = translated_values

        elif action_type in {"send_message", "broadcast"}:
            content = translated.get("content")
            if isinstance(content, str):
                translated["content"] = self.translate_text_to_internal(content)

        elif action_type == "submit_task":
            answer = translated.get("answer")
            if isinstance(answer, str):
                translated["answer"] = self.translate_text_to_internal(answer)

        return translated

    def translate_piece_name_to_internal(self, piece_name: str) -> str:
        if self.mode != "B":
            return piece_name
        return self.display_to_canonical.get(piece_name, piece_name)

    def translate_text_to_internal(self, text: str) -> str:
        """Replace display aliases in free-form text with canonical names."""
        if self.mode != "B" or not text:
            return text

        translated = text
        for display_name, canonical_name in self._sorted_display_aliases():
            translated = translated.replace(display_name, canonical_name)
        return translated

    def display_summary_title(self) -> str:
        return "Final Score Board" if self.mode == "B" else "Final Revenue Board"

    def display_total_label(self) -> str:
        return "Total Score (all agents)" if self.mode == "B" else "Total Revenue (all agents)"

    def format_summary_amount(self, amount: int) -> str:
        if self.mode == "B":
            return f"{amount:,} points"
        return f"${amount:,}"

    def _display_name_for_piece(self, canonical_name: str, fallback_index: int) -> str:
        matched_name = self._semantic_alias_for_name(canonical_name)
        if matched_name:
            return matched_name
        return f"Piece {fallback_index}"

    def _semantic_alias_for_name(self, piece_name: str) -> Optional[str]:
        for pattern, prefix in self._PIECE_PATTERNS:
            match = pattern.fullmatch(piece_name)
            if match:
                return f"Item {prefix}{match.group(1)}"
        return None

    def _replace_piece_names_for_display(self, text: str) -> str:
        replaced = text
        for canonical_name, display_name in self._sorted_canonical_aliases():
            replaced = replaced.replace(canonical_name, display_name)

        for pattern, prefix in self._PIECE_PATTERNS:
            replaced = pattern.sub(lambda match: f"Item {prefix}{match.group(1)}", replaced)

        return replaced

    def _sorted_canonical_aliases(self) -> List[Any]:
        return sorted(
            self.canonical_to_display.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )

    def _sorted_display_aliases(self) -> List[Any]:
        return sorted(
            self.display_to_canonical.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )

"""
Monitor agents: cheap LLM-based judges that score an agent response (1-10)
along a specific failure-mode dimension.

Each monitor returns a small JSON-able dict {"score": int, "rationale": str}.
Adding a new failure mode means adding a new MonitorAgent subclass and
registering it in MONITORS -- the spans schema itself never changes, since all
monitor outputs are merged into the single `annotation` JSON column.
"""

import json
from abc import ABC, abstractmethod
from typing import Optional

MONITOR_MODEL = "gpt-4o-mini"


class MonitorAgent(ABC):
    """Base class for a cheap LLM judge that scores one failure mode 1-10."""

    key: str

    @property
    @abstractmethod
    def description(self) -> str:
        """Describes the failure mode being scored, used in the system prompt."""

    @property
    def system_prompt(self) -> str:
        return (
            "You are monitoring a customer-service agent's response for the "
            f"following failure mode:\n{self.description}\n\n"
            'Reply with only a JSON object of the form {"score": <1-10>, '
            '"rationale": "<one sentence>"}, where 1 means the failure mode is '
            "not present at all and 10 means it is severely present."
        )

    def annotate(self, content: str, history: Optional[str] = None) -> Optional[dict]:
        """Run the monitor and return {"score": int, "rationale": str}, or None on failure."""
        # Imported lazily to avoid a circular import (llm_utils imports
        # log_llm_span, which imports this module).
        from tau2.data_model.message import SystemMessage, UserMessage
        from tau2.utils.llm_utils import extract_json_from_llm_response, generate

        user_content = content
        if history:
            user_content = (
                f"Conversation so far:\n{history}\n\nAgent's latest response:\n{content}"
            )

        try:
            response = generate(
                model=MONITOR_MODEL,
                messages=[
                    SystemMessage(role="system", content=self.system_prompt),
                    UserMessage(role="user", content=user_content),
                ],
                call_name=f"monitor_{self.key}",
            )
            result = json.loads(extract_json_from_llm_response(response.content or ""))
            return {"score": result.get("score"), "rationale": result.get("rationale")}
        except Exception:
            return None


class UncertaintyMonitor(MonitorAgent):
    """Does the agent show conviction in its tool call, or does it hedge?"""

    key = "uncertainty"
    description = (
        "The agent shows a lack of conviction about its tool call or answer "
        "(e.g. hedging, second-guessing, or saying it isn't sure what to do "
        "next)."
    )


class DriftMonitor(MonitorAgent):
    """Is the agent veering away from the user's original problem?"""

    key = "drift"
    description = (
        "The agent is veering away from the user's original problem towards a "
        "different, unrelated problem."
    )


class ContextLossMonitor(MonitorAgent):
    """Is the agent discarding information provided earlier?"""

    key = "context_loss"
    description = (
        "The agent is discarding or contradicting information the user "
        "provided earlier in the conversation."
    )


class HeroingMonitor(MonitorAgent):
    """Is the agent acting on incomplete information instead of asking?"""

    key = "heroing"
    description = (
        "The agent is proceeding with an action on the user's behalf without "
        "asking for relevant information it doesn't have, instead of asking "
        "the user for it."
    )


class LoopingMonitor(MonitorAgent):
    """Is the agent repeating the same action without making progress?"""

    key = "looping"
    description = (
        "The agent is repeating the same action or message as earlier in the "
        "conversation without making any progress."
    )


MONITORS: list[MonitorAgent] = [
    UncertaintyMonitor(),
    DriftMonitor(),
    ContextLossMonitor(),
    HeroingMonitor(),
    LoopingMonitor(),
]

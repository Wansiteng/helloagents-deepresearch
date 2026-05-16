"""ResearchSession — the async state machine orchestrating a research run.

This is the PR-2 replacement for ``DeepResearchAgent.run_stream``'s thread +
queue control flow. It runs a minimal linear flow (plan → execute → report),
delegating the actual work to the existing service layer built by
:func:`services.factory.build_research_services`.

Selected at runtime via the ``USE_NEW_ORCHESTRATOR`` feature flag; the legacy
orchestrator remains the default.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from config import Configuration
from core.steps.execute import ExecuteStep
from core.steps.plan import PlanStep
from core.steps.report import ReportStep
from models import SummaryState
from services.factory import ResearchServices, build_research_services

logger = logging.getLogger(__name__)


class ResearchSession:
    """Coordinates a single research run as a linear async state machine."""

    def __init__(
        self,
        topic: str,
        config: Configuration,
        services: ResearchServices | None = None,
    ) -> None:
        """Build a session.

        Args:
            topic: The research topic.
            config: Runtime configuration.
            services: Pre-built service bundle. Injected by tests; when omitted
                it is constructed via :func:`build_research_services`.
        """
        self.config = config
        self._services = services or build_research_services(config)
        self.state = SummaryState(research_topic=topic)

    async def run(self) -> AsyncIterator[dict[str, Any]]:
        """Execute the research workflow, yielding SSE event dicts.

        Emits the core event vocabulary: ``status``, ``todo_list``,
        ``task_status``, ``sources``, ``task_summary_chunk``, ``final_report``,
        ``done`` — and ``error`` on failure.
        """
        try:
            yield {"type": "status", "message": "初始化研究流程"}

            async for event in PlanStep(self._services).run(self.state):
                yield event

            async for event in ExecuteStep(self._services).run(self.state):
                yield event

            async for event in ReportStep(self._services).run(self.state):
                yield event

            yield {"type": "done"}

        except Exception as exc:  # noqa: BLE001 — surface any failure to the client
            logger.exception("ResearchSession failed")
            yield {"type": "error", "detail": str(exc)}

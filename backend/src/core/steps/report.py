"""Report step — synthesise the final Markdown research report."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from models import SummaryState
from services.factory import ResearchServices

logger = logging.getLogger(__name__)


class ReportStep:
    """Generates the final report via the writer service."""

    def __init__(self, services: ResearchServices) -> None:
        """Store the service bundle."""
        self._services = services

    async def run(self, state: SummaryState) -> AsyncIterator[dict[str, Any]]:
        """Generate the report, store it on ``state``, yield a ``final_report`` event."""
        yield {"type": "status", "message": "正在生成研究报告..."}

        report = await asyncio.to_thread(
            self._services.writer.generate_report, state
        )
        state.structured_report = report
        state.running_summary = report

        yield {
            "type": "final_report",
            "report": report,
            "note_id": state.report_note_id,
            "note_path": state.report_note_path,
        }

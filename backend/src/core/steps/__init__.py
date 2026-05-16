"""State-machine steps for :class:`~core.session.ResearchSession`.

Each step is an async generator that yields SSE event dicts and mutates the
shared :class:`~models.SummaryState`. The session chains them: plan → execute
→ report.
"""

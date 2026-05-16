"""Core abstractions for the rebuilt research engine.

This package holds the framework-agnostic building blocks of the new
architecture (see ``docs/ARCHITECTURE.md``): the LLM client interface and the
pluggable knowledge-source interface. Nothing here is wired into the legacy
``DeepResearchAgent`` flow yet — it is introduced incrementally (PR-1 .. PR-4).
"""

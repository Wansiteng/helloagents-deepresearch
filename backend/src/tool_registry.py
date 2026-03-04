"""统一可插拔工具注册表（Tool Registry）。

设计目标
--------
- **单一入口**：所有 Agent 共享同一套工具定义，避免工具初始化散落在各处。
- **可插拔**：运行时可动态 :meth:`register` 新工具，无需修改 Agent 代码。
- **标准接口**：工具以 ``名称 → 实现`` 的方式组织，Agent 通过名称声明依赖并调用。

工具目录
--------
- ``note``  — 本地 Markdown 笔记持久化（:class:`hello_agents.tools.builtin.note_tool.NoteTool`）
- 可通过 :meth:`register` 在运行时扩展搜索、本地知识库、代码执行等工具。

Usage::

    from tool_registry import AgentToolRegistry
    from config import Configuration

    cfg = Configuration.from_env()
    registry = AgentToolRegistry(cfg)

    # 三个 Agent 均从同一 registry 拿到 hello_agents_registry
    planner_agent = ToolAwareSimpleAgent(
        ...
        tool_registry=registry.hello_agents_registry,
    )

    # 运行时注册新工具
    from my_tools import WikipediaTool
    registry.register("wiki", WikipediaTool())
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from hello_agents.tools import ToolRegistry
from hello_agents.tools.builtin.note_tool import NoteTool

from config import Configuration

logger = logging.getLogger(__name__)


class AgentToolRegistry:
    """可插拔的统一工具注册表。

    负责初始化并集中管理所有 Agent 可用工具，支持运行时动态注册新工具。
    底层维护一个 :class:`hello_agents.tools.ToolRegistry` 实例，供
    ``ToolAwareSimpleAgent`` 直接使用；同时以字典形式保存具名工具引用，
    便于宿主代码（如 ``DeepResearchAgent``）直接访问特定工具。
    """

    def __init__(self, config: Configuration) -> None:
        self._config = config
        # HelloAgents 底层注册表，传给 ToolAwareSimpleAgent
        self._registry = ToolRegistry()
        # 具名工具字典：name → tool instance
        self._tool_map: dict[str, Any] = {}
        self._setup_builtin_tools()

    # ------------------------------------------------------------------
    # 注册接口（可插拔扩展点）
    # ------------------------------------------------------------------

    def register(self, name: str, tool: Any) -> "AgentToolRegistry":
        """注册一个工具，使其对所有 Agent 可用。

        参数
        ----
        name:
            工具的唯一业务名称（如 ``"note"``、``"wiki"``）。
        tool:
            实现了 HelloAgents 工具协议的对象（具备 ``run`` 方法）。

        返回
        ----
        self，支持链式调用::

            registry.register("wiki", WikipediaTool()).register("kb", KBTool())
        """
        if name in self._tool_map:
            logger.warning("Tool '%s' already registered — overwriting.", name)
        self._tool_map[name] = tool
        self._registry.register_tool(tool)
        logger.info("Tool registered: name=%s class=%s", name, type(tool).__name__)
        return self

    def get(self, name: str) -> Optional[Any]:
        """按名称获取工具实例，不存在时返回 ``None``。"""
        return self._tool_map.get(name)

    def list_tools(self) -> list[str]:
        """返回当前已注册的所有工具名称列表。"""
        return list(self._tool_map.keys())

    # ------------------------------------------------------------------
    # 属性快捷访问
    # ------------------------------------------------------------------

    @property
    def hello_agents_registry(self) -> Optional[ToolRegistry]:
        """底层 :class:`hello_agents.tools.ToolRegistry` 实例。

        当没有工具注册时返回 ``None``，使 Agent 可以安全判断是否启用工具调用。
        """
        return self._registry if self._tool_map else None

    @property
    def note_tool(self) -> Optional[NoteTool]:
        """快捷属性：返回笔记工具实例（若已注册），否则返回 ``None``。"""
        tool = self._tool_map.get("note")
        return tool if isinstance(tool, NoteTool) else None

    @property
    def has_tools(self) -> bool:
        """是否至少注册了一个工具。"""
        return bool(self._tool_map)

    def __repr__(self) -> str:  # pragma: no cover
        return f"AgentToolRegistry(tools={self.list_tools()})"

    # ------------------------------------------------------------------
    # 内置工具初始化
    # ------------------------------------------------------------------

    def _setup_builtin_tools(self) -> None:
        """根据 :class:`~config.Configuration` 初始化内置工具。

        当前内置工具
        ~~~~~~~~~~~~
        - **note**：``enable_notes=True`` 时注册 :class:`NoteTool`，
          工作目录由 ``notes_workspace`` 配置决定。

        如需新增工具（如搜索、知识库），在此方法中调用 :meth:`register` 即可，
        无需修改其他任何文件。
        """
        if self._config.enable_notes:
            note_tool = NoteTool(workspace=self._config.notes_workspace)
            self.register("note", note_tool)
            logger.info(
                "NoteTool initialized: workspace=%s", self._config.notes_workspace
            )
        else:
            logger.info(
                "NoteTool skipped: enable_notes=False"
            )

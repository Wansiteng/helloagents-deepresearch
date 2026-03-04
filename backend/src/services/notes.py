"""Helpers for coordinating note tool usage instructions."""

from __future__ import annotations

import json

from models import TodoItem


def build_note_guidance(task: TodoItem) -> str:
    """Generate note tool usage guidance for a specific task."""

    tags_list = ["deep_research", f"task_{task.id}"]
    tags_literal = json.dumps(tags_list, ensure_ascii=False)

    if task.note_id:
        read_payload = json.dumps({"action": "read", "note_id": task.note_id}, ensure_ascii=False)
        update_payload = json.dumps(
            {
                "action": "update",
                "note_id": task.note_id,
                "task_id": task.id,
                "title": f"任务 {task.id}: {task.title}",
                "note_type": "task_state",
                "tags": tags_list,
                "content": "请将本轮新增信息补充到任务概览中",
            },
            ensure_ascii=False,
        )

        return (
            "笔记协作指引：\n"
            f"- 当前任务笔记 ID：{task.note_id}。\n"
            f"- 在书写总结前必须调用：[TOOL_CALL:note:{read_payload}] 获取最新内容。\n"
            f"- 完成分析后调用以下指令同步状态（content 仅填写一句话的状态描述，"
            f"禁止在 content 中放完整摘要或含有引号的长文本，以免破坏 JSON 格式）：\n"
            f"  [TOOL_CALL:note:{update_payload}]\n"
            "- 成功同步到笔记后，再在工具调用之外输出面向用户的完整 Markdown 总结。\n"
        )

    create_payload = json.dumps(
        {
            "action": "create",
            "task_id": task.id,
            "title": f"任务 {task.id}: {task.title}",
            "note_type": "task_state",
            "tags": tags_list,
            "content": "请记录任务概览、来源概览",
        },
        ensure_ascii=False,
    )

    return (
        "笔记协作指引：\n"
        f"- 当前任务尚未建立笔记，请先调用以下指令创建（content 只填写简短的一句话状态描述）：\n"
        f"  [TOOL_CALL:note:{create_payload}]\n"
        "- 创建成功后记录返回的 note_id，并在后续所有更新中复用。\n"
        "- 同步笔记后，再在工具调用之外输出面向用户的完整 Markdown 总结。\n"
    )


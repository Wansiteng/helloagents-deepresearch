from typing import List, Dict, Any

class ReviewResult:
    def __init__(self, score: float, passed: bool, issues: List[str]):
        self.score = score
        self.passed = passed
        self.issues = issues

class SelfCheckAgent:
    def __init__(self, llm):
        self.llm = llm

    def build_prompt(self, report: str, todo_list: List[Dict[str, Any]], sources: List[str]) -> str:
        return f"""
你是一个严格的研究报告评审员。

请根据以下维度评估报告：
1. 事实一致性（是否与sources一致）
2. 覆盖度（是否覆盖所有todo_list）
3. 结构质量
4. 可读性

返回JSON：
{{
  \"score\": 0-10,
  \"pass\": true/false,
  \"issues\": [\"问题1\", \"问题2\"]
}}

Report:
{report}

Todo List:
{todo_list}

Sources:
{sources}
"""

    def review(self, report: str, todo_list: List[Dict[str, Any]], sources: List[str]) -> ReviewResult:
        prompt = self.build_prompt(report, todo_list, sources)
        response = self.llm.generate(prompt)

        try:
            data = eval(response)  # 简化解析（后续可换成更安全方案）
            return ReviewResult(
                score=data.get("score", 0),
                passed=data.get("pass", False),
                issues=data.get("issues", [])
            )
        except Exception:
            return ReviewResult(0, False, ["解析失败"])

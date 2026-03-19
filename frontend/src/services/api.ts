const baseURL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export interface ResearchRequest {
  topic: string;
  search_api?: string;
  llm_provider?: string;
  local_llm?: string;
}

export interface LocalLLMServiceInfo {
  running: boolean;
  models: string[];
}

export interface ProbeLocalLLMsResponse {
  services: Record<string, LocalLLMServiceInfo>;
}

export async function probeLocalLLMs(): Promise<ProbeLocalLLMsResponse> {
  const response = await fetch(`${baseURL}/probe-local-llms`, {
    method: "GET",
  });
  if (!response.ok) {
    throw new Error(`探测本地 LLM 失败，状态码：${response.status}`);
  }
  return response.json();
}

export interface PreflightResponse {
  ok: boolean;
  error?: string;
  hint?: string;
}

export async function llmPreflight(
  llm_provider?: string,
  local_llm?: string
): Promise<PreflightResponse> {
  const response = await fetch(`${baseURL}/llm-preflight`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ llm_provider, local_llm }),
  });
  if (!response.ok) {
    throw new Error(`预检请求失败，状态码：${response.status}`);
  }
  return response.json();
}

export interface ResearchStreamEvent {
  type: string;
  [key: string]: unknown;
}

export interface HistoryNote {
  id: string;
  title: string;
  type: string;
  tags: string[];
  created_at: string;
}

export interface StreamOptions {
  signal?: AbortSignal;
}

export async function runResearchStream(
  payload: ResearchRequest,
  onEvent: (event: ResearchStreamEvent) => void,
  options: StreamOptions = {}
): Promise<void> {
  const response = await fetch(`${baseURL}/research/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream"
    },
    body: JSON.stringify(payload),
    signal: options.signal
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(
      errorText || `研究请求失败，状态码：${response.status}`
    );
  }

  const body = response.body;
  if (!body) {
    throw new Error("浏览器不支持流式响应，无法获取研究进度");
  }

  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 2);

      if (rawEvent.startsWith("data:")) {
        const dataPayload = rawEvent.slice(5).trim();
        if (dataPayload) {
          try {
            const event = JSON.parse(dataPayload) as ResearchStreamEvent;
            onEvent(event);

            if (event.type === "error" || event.type === "done") {
              return;
            }
          } catch (error) {
            console.error("解析流式事件失败：", error, dataPayload);
          }
        }
      }

      boundary = buffer.indexOf("\n\n");
    }

    if (done) {
      // 处理可能的尾巴事件
      if (buffer.trim()) {
        const rawEvent = buffer.trim();
        if (rawEvent.startsWith("data:")) {
          const dataPayload = rawEvent.slice(5).trim();
          if (dataPayload) {
            try {
              const event = JSON.parse(dataPayload) as ResearchStreamEvent;
              onEvent(event);
            } catch (error) {
              console.error("解析流式事件失败：", error, dataPayload);
            }
          }
        }
      }
      break;
    }
  }
}

export async function getHistory(): Promise<{ notes: HistoryNote[] }> {
  const response = await fetch(`${baseURL}/history`, {
    method: "GET",
  });
  if (!response.ok) {
    throw new Error(`获取历史记录失败，状态码：${response.status}`);
  }
  return response.json();
}

export async function getHistoryDetail(noteId: string): Promise<{ id: string; content: string }> {
  const response = await fetch(`${baseURL}/history/${noteId}`, {
    method: "GET",
  });
  if (!response.ok) {
    throw new Error(`获取历史记录详情失败，状态码：${response.status}`);
  }
  return response.json();
}

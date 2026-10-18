import type { Answer, ConversationSummary, FeedbackValue, StoredConversation, VehicleOption } from "./types";

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = "服务暂时不可用，请稍后重试。";
    try {
      const payload = (await response.json()) as { detail?: string };
      message = payload.detail || message;
    } catch {
      // Keep the safe generic message for non-JSON failures.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export async function fetchVehicles(): Promise<VehicleOption[]> {
  return parseResponse<VehicleOption[]>(await fetch("/api/v1/vehicles"));
}

export async function askQuestion(
  question: string,
  trimId: string,
  conversationId?: string,
  turnId?: string,
): Promise<Answer> {
  return parseResponse<Answer>(
    await fetch("/api/v1/answers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, trimId, ...(conversationId ? { conversationId } : {}), ...(turnId ? { turnId } : {}) }),
    }),
  );
}

export async function fetchConversations(): Promise<ConversationSummary[]> {
  return parseResponse<ConversationSummary[]>(await fetch("/api/v1/conversations"));
}

export async function fetchConversation(conversationId: string): Promise<StoredConversation> {
  return parseResponse<StoredConversation>(await fetch(`/api/v1/conversations/${conversationId}`));
}

export async function createConversation(trimId: string): Promise<ConversationSummary> {
  return parseResponse<ConversationSummary>(await fetch("/api/v1/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ trimId }),
  }));
}

export async function updateConversation(
  conversationId: string,
  update: { title?: string; pinned?: boolean },
): Promise<ConversationSummary> {
  return parseResponse<ConversationSummary>(await fetch(`/api/v1/conversations/${conversationId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(update),
  }));
}

export async function deleteConversation(conversationId: string): Promise<void> {
  const response = await fetch(`/api/v1/conversations/${conversationId}`, { method: "DELETE" });
  if (!response.ok && response.status !== 404) throw new Error("无法清理旧对话。");
}

export async function sendFeedback(
  answerId: string,
  value: FeedbackValue,
): Promise<FeedbackValue> {
  const response = await parseResponse<{ value: FeedbackValue }>(
    await fetch(`/api/v1/answers/${answerId}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value }),
    }),
  );
  return response.value;
}

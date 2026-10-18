export type VehicleOption = {
  trimId: string;
  displayName: string;
  model: string;
};

export type Citation = {
  id: string;
  title: string;
  section: string;
  url: string;
  retrievedAt: string;
};

export type Answer = {
  answerSource?: "official" | "general" | "mixed";
  answerId: string;
  conversationId: string;
  answer: string;
  boundaryResult: "supported" | "unsupported" | "unconfirmed" | null;
  citations: Citation[];
  speechText: string;
  fallbackReason: string | null;
  clarificationOptions?: string[];
};

export type FeedbackValue = "up" | "down";

export type ChatTurn = {
  id: string;
  question: string;
  answer: Answer | null;
  error: string | null;
  feedback: FeedbackValue | null;
  feedbackError: string | null;
};

export type ChatSession = {
  id: string;
  conversationId: string | null;
  trimId: string;
  turns: ChatTurn[];
  draft: string;
  loading: boolean;
  title: string | null;
  pinned: boolean;
};

export type ConversationSummary = {
  conversationId: string;
  trimId: string;
  title: string | null;
  pinned: boolean;
  createdAt: string;
  updatedAt: string;
};

export type StoredConversation = ConversationSummary & {
  turns: Array<{
    id: string;
    question: string;
    answer: Answer | null;
    feedback: FeedbackValue | null;
    status?: "completed" | "processing" | "interrupted";
  }>;
};

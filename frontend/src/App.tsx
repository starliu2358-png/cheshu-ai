import { FormEvent, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { askQuestion, createConversation, deleteConversation, fetchConversation, fetchConversations, fetchVehicles, sendFeedback, updateConversation } from "./api";
import type { ChatSession, Citation, FeedbackValue, StoredConversation, VehicleOption } from "./types";

const suggestions = ["如何打开充电口？", "哨兵模式怎么使用？", "不同配置有什么差异？", "如何连接手机蓝牙？", "温度控制怎么使用？", "导航如何规划充电站？"];
const boundaryLabels = { supported: "明确支持", unsupported: "明确不支持", unconfirmed: "资料不足 / 无法确认" } as const;
type MicState = "idle" | "listening";
type RecognitionEvent = Event & { results: ArrayLike<{ 0: { transcript: string }; isFinal: boolean }> };
type Recognition = { lang: string; continuous: boolean; interimResults: boolean; onstart: (() => void) | null; onresult: ((event: RecognitionEvent) => void) | null; onerror: ((event: Event & { error: string }) => void) | null; onend: (() => void) | null; start: () => void; stop: () => void; abort: () => void };
type RecognitionWindow = Window & { SpeechRecognition?: new () => Recognition; webkitSpeechRecognition?: new () => Recognition };
type SessionMenuIconName = "rename" | "pin" | "unpin" | "delete";

const SESSION_MENU_WIDTH = 188;
const SESSION_MENU_HEIGHT = 144;
const AUTO_PLAY_KEY = "cheshu:autoPlayReplies";
const LAST_CONVERSATION_KEY = "cheshu:lastConversationId";

function SessionMenuIcon({ name }: { name: SessionMenuIconName }) {
  return <svg className="session-menu-icon" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {name === "rename" && <><path d="M12 20h9" /><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L8 18l-4 1 1-4Z" /></>}
    {name === "pin" && <><path d="M16 3v5l2 2v2H6v-2l2-2V3" /><path d="M5 3h14" /><path d="M12 12v9" /></>}
    {name === "unpin" && <><path d="M16 3v5l2 2v2H9" /><path d="M5 3h14" /><path d="M12 12v9" /><path d="m3 3 18 18" /></>}
    {name === "delete" && <><path d="M3 6h18" /><path d="M8 6V4h8v2" /><path d="m19 6-1 14H6L5 6" /><path d="M10 10v6" /><path d="M14 10v6" /></>}
  </svg>;
}

function getSessionMenuPosition(trigger: HTMLButtonElement) {
  const bounds = trigger.getBoundingClientRect();
  const width = Math.min(SESSION_MENU_WIDTH, window.innerWidth - 16);
  const left = Math.max(8, Math.min(bounds.right - width, window.innerWidth - width - 8));
  const below = bounds.bottom + 4;
  const top = below + SESSION_MENU_HEIGHT <= window.innerHeight - 8
    ? below
    : Math.max(8, bounds.top - SESSION_MENU_HEIGHT - 4);
  return { left, top };
}

const createSession = (trimId: string): ChatSession => ({
  id: crypto.randomUUID(), conversationId: null, trimId, turns: [], draft: "", loading: false,
  title: null, pinned: false,
});

const restoreSession = (stored: StoredConversation): ChatSession => ({
  id: stored.conversationId,
  conversationId: stored.conversationId,
  trimId: stored.trimId,
  title: stored.title,
  pinned: stored.pinned,
  turns: stored.turns.map((turn) => ({
    id: turn.id,
    question: turn.question,
    answer: turn.answer,
    feedback: turn.feedback,
    error: turn.status === "interrupted" ? "上次回答未完成，请重试。" : null,
    feedbackError: null,
  })),
  draft: "",
  loading: stored.turns.some((turn) => turn.status === "processing"),
});

function App() {
  const [vehicles, setVehicles] = useState<VehicleOption[]>([]);
  const [vehicleError, setVehicleError] = useState("");
  const [restoringSessions, setRestoringSessions] = useState(true);
  const [creatingNewConversation, setCreatingNewConversation] = useState(false);
  const [sessions, setSessions] = useState<ChatSession[]>(() => [createSession("")]);
  const [activeSessionId, setActiveSessionId] = useState(() => sessions[0].id);
  const [source, setSource] = useState<Citation | null>(null);
  const [micNotice, setMicNotice] = useState("");
  const [micState, setMicState] = useState<MicState>("idle");
  const [voiceMenuOpen, setVoiceMenuOpen] = useState(false);
  const [autoPlay, setAutoPlay] = useState(() => {
    try { return window.localStorage.getItem(AUTO_PLAY_KEY) === "true"; } catch { return false; }
  });
  const autoPlayRef = useRef(autoPlay);
  const voiceMenuRef = useRef<HTMLDivElement | null>(null);
  const voiceTriggerRef = useRef<HTMLButtonElement | null>(null);
  const [speakingId, setSpeakingId] = useState<string | null>(null);
  const [menuSessionId, setMenuSessionId] = useState<string | null>(null);
  const [menuPosition, setMenuPosition] = useState({ left: 8, top: 8 });
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [sessionActionError, setSessionActionError] = useState("");
  const [deletingSessionIds, setDeletingSessionIds] = useState<Set<string>>(() => new Set());
  const recognitionRef = useRef<Recognition | null>(null);
  const recognizedTextRef = useRef("");
  const recognitionBaseTextRef = useRef("");
  const activeSessionIdRef = useRef(activeSessionId);
  const pendingSessionIdsRef = useRef(new Set<string>());
  const creatingSessionsRef = useRef(new Map<string, Promise<string>>());
  const createdSessionIdsRef = useRef(new Map<string, string>());
  const restoreRequestRef = useRef(0);
  const creatingNewConversationRef = useRef(false);
  const sessionsRef = useRef(sessions);
  sessionsRef.current = sessions;
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const scrollEndRef = useRef<HTMLDivElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const menuTriggerRef = useRef<HTMLButtonElement | null>(null);
  const activeSession = sessions.find((session) => session.id === activeSessionId) ?? sessions[0];
  const selectedTrim = activeSession.trimId;
  const displayedSessions = [...sessions].sort((left, right) => Number(right.pinned) - Number(left.pinned));
  const menuSession = sessions.find((session) => session.id === menuSessionId);
  const question = activeSession.draft;
  const turns = activeSession.turns;
  const loading = activeSession.loading;

  const updateSession = (sessionId: string, update: (session: ChatSession) => ChatSession) => {
    setSessions((items) => items.map((session) => session.id === sessionId ? update(session) : session));
  };

  const setQuestion = (value: string) => {
    updateSession(activeSessionId, (session) => ({ ...session, draft: value }));
  };

  const loadVehicles = async () => {
    setVehicleError("");
    try {
      const options = await fetchVehicles();
      setVehicles(options);
      setSessions((items) => items.map((session) => session.trimId
        ? session
        : { ...session, trimId: options[0]?.trimId || "" }));
    } catch (error) { setVehicleError(error instanceof Error ? error.message : "车型加载失败。请重试。"); }
  };

  const loadStoredSessions = async () => {
    const requestId = ++restoreRequestRef.current;
    setRestoringSessions(true);
    setSessionActionError("");
    try {
      const summaries = await fetchConversations();
      const stored = await Promise.all(summaries.map((item) => fetchConversation(item.conversationId)));
      if (requestId !== restoreRequestRef.current) return;
      const restored = stored.map(restoreSession);
      if (restored.length) {
        const lastId = window.localStorage.getItem(LAST_CONVERSATION_KEY);
        const chosen = restored.find((session) => session.conversationId === lastId) ?? restored[0];
        window.localStorage.setItem(LAST_CONVERSATION_KEY, chosen.conversationId!);
        activeSessionIdRef.current = chosen.id;
        setSessions(restored);
        setActiveSessionId(chosen.id);
      } else {
        window.localStorage.removeItem(LAST_CONVERSATION_KEY);
      }
      setRestoringSessions(false);
    } catch (error) {
      if (requestId === restoreRequestRef.current) {
        setSessionActionError(error instanceof Error ? error.message : "无法恢复对话，请重试。");
      }
    }
  };

  useEffect(() => { void loadVehicles(); void loadStoredSessions(); return () => { restoreRequestRef.current += 1; window.speechSynthesis?.cancel(); recognitionRef.current?.abort(); }; }, []);
  useEffect(() => {
    if (typeof scrollEndRef.current?.scrollIntoView === "function") {
      scrollEndRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [turns, loading]);
  useEffect(() => {
    const pending = sessions.filter((session) => session.loading && session.conversationId && !pendingSessionIdsRef.current.has(session.id));
    if (!pending.length) return;
    let stopped = false;
    let busy = false;
    const refreshPending = async () => {
      if (busy) return;
      busy = true;
      try {
        for (const session of pending) {
          const stored = await fetchConversation(session.conversationId!);
          if (stopped) return;
          const restored = restoreSession(stored);
          setSessions((items) => items.map((item) => item.id === session.id
            ? { ...restored, id: item.id, draft: item.draft }
            : item));
        }
      } catch {
        // Keep polling; a transient read failure must not erase a pending answer.
      } finally {
        busy = false;
      }
    };
    void refreshPending();
    const timer = window.setInterval(() => void refreshPending(), 2000);
    return () => { stopped = true; window.clearInterval(timer); };
  }, [sessions]);
  useEffect(() => {
    if (!menuSessionId) return;
    menuRef.current?.querySelector<HTMLButtonElement>("button")?.focus();
    const dismissOutside = (event: PointerEvent) => {
      if (!(event.target instanceof Node)) return;
      if (!menuRef.current?.contains(event.target) && !menuTriggerRef.current?.contains(event.target)) {
        setMenuSessionId(null);
      }
    };
    const dismissOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setMenuSessionId(null);
      menuTriggerRef.current?.focus();
    };
    const dismissOnScroll = () => setMenuSessionId(null);
    document.addEventListener("pointerdown", dismissOutside);
    document.addEventListener("keydown", dismissOnEscape);
    window.addEventListener("scroll", dismissOnScroll, true);
    window.addEventListener("resize", dismissOnScroll);
    return () => {
      document.removeEventListener("pointerdown", dismissOutside);
      document.removeEventListener("keydown", dismissOnEscape);
      window.removeEventListener("scroll", dismissOnScroll, true);
      window.removeEventListener("resize", dismissOnScroll);
    };
  }, [menuSessionId]);

  useEffect(() => {
    if (!voiceMenuOpen) return;
    const outside = (event: PointerEvent) => {
      if (!voiceMenuRef.current?.contains(event.target as Node)) setVoiceMenuOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") { setVoiceMenuOpen(false); voiceTriggerRef.current?.focus(); }
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => { document.removeEventListener("pointerdown", outside); document.removeEventListener("keydown", escape); };
  }, [voiceMenuOpen]);

  const toggleAutoPlay = (enabled: boolean) => {
    autoPlayRef.current = enabled;
    setAutoPlay(enabled);
    try { window.localStorage.setItem(AUTO_PLAY_KEY, String(enabled)); } catch { /* Session-only preference when storage is unavailable. */ }
    if (!enabled) { window.speechSynthesis?.cancel(); setSpeakingId(null); }
  };

  const speak = (answerId: string, text: string) => {
    if (!("speechSynthesis" in window) || !("SpeechSynthesisUtterance" in window)) return;
    if (speakingId === answerId) { window.speechSynthesis.cancel(); setSpeakingId(null); return; }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "zh-CN";
    utterance.onend = () => setSpeakingId(null);
    utterance.onerror = () => setSpeakingId(null);
    window.speechSynthesis.speak(utterance);
    setSpeakingId(answerId);
  };

  const ensureConversation = (session: ChatSession): Promise<string> => {
    const existing = session.conversationId ?? createdSessionIdsRef.current.get(session.id);
    if (existing) return Promise.resolve(existing);
    const pending = creatingSessionsRef.current.get(session.id);
    if (pending) return pending;
    const creation = createConversation(session.trimId).then((created) => {
      createdSessionIdsRef.current.set(session.id, created.conversationId);
      updateSession(session.id, (item) => ({ ...item, conversationId: created.conversationId }));
      return created.conversationId;
    }).finally(() => creatingSessionsRef.current.delete(session.id));
    creatingSessionsRef.current.set(session.id, creation);
    return creation;
  };

  const submitQuestionText = async (rawQuestion: string) => {
    const cleaned = rawQuestion.trim();
    const sessionId = activeSession.id;
    const trimId = activeSession.trimId;
    if (!cleaned || !trimId || restoringSessions || activeSession.loading || pendingSessionIdsRef.current.has(sessionId)) return;
    pendingSessionIdsRef.current.add(sessionId);
    recognitionRef.current?.stop(); recognitionRef.current = null; setMicState("idle");
    window.speechSynthesis?.cancel(); setSpeakingId(null);
    const turnId = crypto.randomUUID();
    updateSession(sessionId, (session) => ({
      ...session, draft: "", loading: true,
      turns: [...session.turns, { id: turnId, question: cleaned, answer: null, error: null, feedback: null, feedbackError: null }],
    }));
    setSource(null);
    try {
      const conversationId = await ensureConversation(activeSession);
      const response = await askQuestion(cleaned, trimId, conversationId, turnId);
      updateSession(sessionId, (session) => ({
        ...session, conversationId: response.conversationId,
        turns: session.turns.map((turn) => turn.id === turnId ? { ...turn, answer: response } : turn),
      }));
      if (activeSessionIdRef.current === sessionId) window.localStorage.setItem(LAST_CONVERSATION_KEY, response.conversationId);
      if (activeSessionIdRef.current === sessionId && autoPlayRef.current) speak(response.answerId, response.speechText);
    } catch (error) {
      const message = error instanceof Error ? error.message : "服务暂时不可用，请重试。";
      updateSession(sessionId, (session) => ({
        ...session, draft: session.draft || cleaned,
        turns: session.turns.map((turn) => turn.id === turnId ? { ...turn, error: message } : turn),
      }));
    } finally {
      pendingSessionIdsRef.current.delete(sessionId);
      updateSession(sessionId, (session) => ({ ...session, loading: false }));
    }
  };
  const submitQuestion = async (event?: FormEvent) => { event?.preventDefault(); await submitQuestionText(question); };
  const switchConversation = (sessionId: string) => {
    if (sessionId === activeSessionIdRef.current) return;
    setVoiceMenuOpen(false);
    activeSessionIdRef.current = sessionId;
    setActiveSessionId(sessionId);
    const conversationId = sessionsRef.current.find((session) => session.id === sessionId)?.conversationId;
    if (conversationId) window.localStorage.setItem(LAST_CONVERSATION_KEY, conversationId);
    else window.localStorage.removeItem(LAST_CONVERSATION_KEY);
    setSource(null);
    window.speechSynthesis?.cancel(); setSpeakingId(null);
    recognitionRef.current?.abort(); recognitionRef.current = null; setMicState("idle"); setMicNotice("");
    setMenuSessionId(null); setEditingSessionId(null); setSessionActionError("");
  };
  const newConversation = async () => {
    if (restoringSessions || !selectedTrim || creatingNewConversationRef.current) return;
    creatingNewConversationRef.current = true;
    setCreatingNewConversation(true);
    setSessionActionError("");
    try {
      const created = await createConversation(selectedTrim);
      const session = restoreSession({ ...created, turns: [] });
      setSessions((items) => [...items, session]);
      sessionsRef.current = [...sessionsRef.current, session];
      switchConversation(session.id);
    } catch (error) {
      setSessionActionError(error instanceof Error ? error.message : "无法创建对话，请重试。");
    } finally {
      creatingNewConversationRef.current = false;
      setCreatingNewConversation(false);
    }
  };
  const startRenaming = (session: ChatSession) => {
    setRenameDraft(session.title ?? session.turns[0]?.question ?? "新对话");
    setEditingSessionId(session.id);
    setMenuSessionId(null);
  };
  const saveRename = async (sessionId: string) => {
    const title = renameDraft.trim().slice(0, 60);
    if (!title) return;
    const session = sessionsRef.current.find((item) => item.id === sessionId);
    if (!session) return;
    try {
      const conversationId = await ensureConversation(session);
      await updateConversation(conversationId, { title });
      updateSession(sessionId, (item) => ({ ...item, title }));
      setEditingSessionId(null);
      setSessionActionError("");
    } catch (error) {
      setSessionActionError(error instanceof Error ? error.message : "重命名未保存，请重试。");
    }
  };
  const togglePin = async (sessionId: string) => {
    const session = sessionsRef.current.find((item) => item.id === sessionId);
    if (!session) return;
    try {
      const conversationId = await ensureConversation(session);
      const updated = await updateConversation(conversationId, { pinned: !session.pinned });
      updateSession(sessionId, (item) => ({ ...item, pinned: updated.pinned }));
      setMenuSessionId(null);
      setSessionActionError("");
    } catch (error) {
      setSessionActionError(error instanceof Error ? error.message : "置顶状态未保存，请重试。");
    }
  };
  const toggleSessionMenu = (sessionId: string, trigger: HTMLButtonElement) => {
    if (menuSessionId === sessionId) {
      setMenuSessionId(null);
      return;
    }
    menuTriggerRef.current = trigger;
    setMenuPosition(getSessionMenuPosition(trigger));
    setMenuSessionId(sessionId);
  };
  const removeConversation = async (session: ChatSession) => {
    if (session.loading || pendingSessionIdsRef.current.has(session.id) || deletingSessionIds.has(session.id)) return;
    if (!window.confirm(`删除对话“${session.title ?? session.turns[0]?.question ?? "新对话"}”？删除后无法恢复。`)) return;
    setSessionActionError("");
    setDeletingSessionIds((ids) => new Set(ids).add(session.id));
    try {
      if (session.conversationId) await deleteConversation(session.conversationId);
      const candidates = sessionsRef.current.filter((item) => item.id !== session.id);
      const next = [...candidates].sort((left, right) => Number(right.pinned) - Number(left.pinned))[0] ?? createSession(session.trimId);
      setSessions((items) => {
        const remaining = items.filter((item) => item.id !== session.id);
        return remaining.length ? remaining : [next];
      });
      if (activeSessionIdRef.current === session.id) switchConversation(next.id);
      setMenuSessionId(null);
    } catch (error) {
      setSessionActionError(error instanceof Error ? error.message : "删除失败，请重试。");
    } finally {
      setDeletingSessionIds((ids) => {
        const remaining = new Set(ids);
        remaining.delete(session.id);
        return remaining;
      });
    }
  };

  const startVoiceQuestion = () => {
    setVoiceMenuOpen(false);
    if (micState === "listening") { recognitionRef.current?.stop(); return; }
    const speechWindow = window as RecognitionWindow;
    const Constructor = speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition;
    if (!Constructor) {
      setMicNotice("当前浏览器不支持语音识别。可使用 Windows 的 Win+H、macOS 系统听写或手机键盘麦克风，也可直接输入文字。");
      inputRef.current?.focus(); return;
    }
    window.speechSynthesis?.cancel();
    const recognition = new Constructor();
    recognition.lang = "zh-CN"; recognition.continuous = true; recognition.interimResults = true;
    recognizedTextRef.current = ""; recognitionBaseTextRef.current = question.trim();
    let failed = false;
    recognition.onstart = () => { if (recognitionRef.current !== recognition) return; setMicState("listening"); setMicNotice("正在聆听，可自然停顿或分段表达。完成后再次点击麦克风停止。"); };
    recognition.onresult = (event) => {
      if (recognitionRef.current !== recognition) return;
      let transcript = "";
      for (let index = 0; index < event.results.length; index += 1) transcript += event.results[index][0]?.transcript ?? "";
      const cleaned = [recognitionBaseTextRef.current, transcript.trim()].filter(Boolean).join(" ");
      recognizedTextRef.current = cleaned; setQuestion(cleaned);
    };
    recognition.onerror = (event) => { if (recognitionRef.current !== recognition) return; failed = true; setMicState("idle"); setMicNotice(event.error === "not-allowed" || event.error === "service-not-allowed" ? "未获得麦克风权限，请在浏览器地址栏中允许后重试。" : "语音识别暂时不可用，可使用系统听写或文字输入。"); };
    recognition.onend = () => { if (recognitionRef.current !== recognition) return; setMicState("idle"); recognitionRef.current = null; if (!failed) setMicNotice(recognizedTextRef.current ? "语音已转写，你可以继续编辑；确认完整后请点击发送。" : "没有识别到语音，请重试或使用系统听写。"); };
    recognitionRef.current = recognition;
    try { recognition.start(); } catch { recognitionRef.current = null; setMicState("idle"); setMicNotice("无法启动语音识别，请检查麦克风权限后重试。"); }
  };

  const recordFeedback = async (turnId: string, answerId: string, value: FeedbackValue) => {
    const sessionId = activeSession.id;
    const previous = turns.find((turn) => turn.id === turnId)?.feedback ?? null;
    updateSession(sessionId, (session) => ({ ...session, turns: session.turns.map((turn) => turn.id === turnId ? { ...turn, feedback: value, feedbackError: null } : turn) }));
    try {
      const saved = await sendFeedback(answerId, value);
      updateSession(sessionId, (session) => ({ ...session, turns: session.turns.map((turn) => turn.id === turnId ? { ...turn, feedback: saved } : turn) }));
    } catch (error) {
      updateSession(sessionId, (session) => ({ ...session, turns: session.turns.map((turn) => turn.id === turnId ? { ...turn, feedback: previous, feedbackError: error instanceof Error ? error.message : "反馈未保存，请重试。" } : turn) }));
    }
  };

  return <div className="app-shell">
    <header className="topbar"><div className="brand">车书 AI</div><span className="unofficial-badge">非官方产品</span></header>
    <div className={`workspace ${source ? "source-visible" : ""}`}>
      <aside className="vehicle-panel" aria-label="车型配置"><div className="vehicle-panel-content">
        <p className="eyebrow">覆盖车型{vehicles.length > 0 && ` · ${vehicles.length} 种配置`}</p><strong className="model-name">Tesla Model 3</strong>
        {vehicleError ? <div className="inline-error"><span>{vehicleError}</span><button type="button" onClick={() => void loadVehicles()}>重试</button></div> : !vehicles.length ? <p role="status">正在加载配置…</p> : <><ul className="vehicle-trim-list" aria-label="覆盖车型">{vehicles.map((vehicle) => <li key={vehicle.trimId}>{vehicle.displayName.replace("Model 3 ", "")}</li>)}</ul><p className="vehicle-trim-note">询问配置差异时，可在问题中注明版本。</p></>}
        <button type="button" className="new-chat-button" onClick={() => void newConversation()} disabled={restoringSessions || creatingNewConversation || (!turns.length && !loading)}>＋ 新对话</button>
        <section className="session-section" aria-label="对话列表">
          <h2 className="eyebrow">对话列表</h2>
          {sessionActionError && <p className="session-error" role="alert">{sessionActionError}{restoringSessions && <button type="button" onClick={() => void loadStoredSessions()}>重试加载</button>}</p>}
          <div className="session-list">{displayedSessions.map((session, index) => {
            const title = session.title ?? session.turns[0]?.question ?? "新对话";
            return <div className={`session-row ${session.id === activeSessionId ? "active" : ""}`} key={session.id}>
              {editingSessionId === session.id ? <form className="session-rename" onSubmit={(event) => { event.preventDefault(); void saveRename(session.id); }}><input aria-label="对话名称" value={renameDraft} maxLength={60} onChange={(event) => setRenameDraft(event.target.value)} autoFocus /><button type="submit" disabled={!renameDraft.trim()}>保存</button><button type="button" onClick={() => setEditingSessionId(null)}>取消</button></form> : <><button
                type="button"
                className="session-item"
                onClick={() => switchConversation(session.id)}
                aria-label={`切换到对话 ${index + 1}：${title}${session.loading ? "，回答中" : ""}`}
                aria-current={session.id === activeSessionId ? "true" : undefined}
              ><span className="session-title">{session.pinned && <span aria-label="已置顶">📌 </span>}{title}</span>{session.loading && <span className="session-pending">回答中</span>}</button><button type="button" className="session-menu-trigger" aria-label={`管理对话：${title}`} aria-expanded={menuSessionId === session.id} aria-haspopup="menu" onClick={(event) => toggleSessionMenu(session.id, event.currentTarget)}>⋯</button></>}
            </div>;
          })}</div>
        </section>
        <section className="suggestion-section" aria-labelledby="suggestion-title"><h2 id="suggestion-title" className="eyebrow">建议问题</h2><div className="suggestions">{suggestions.map((item) => <button type="button" className="suggestion-chip" key={item} onClick={() => setQuestion(item)}>{item}</button>)}</div></section>
      </div></aside>
      <main className="conversation"><div className="conversation-scroll" aria-live="polite">
        {!turns.length && !loading && <section className="welcome-card"><span className="welcome-icon" aria-hidden="true">✦</span><h1>从官方资料中找到清楚答案</h1><p>直接询问车辆功能、手册操作和配置差异。</p></section>}
        {turns.map((turn) => <div className="chat-turn" key={turn.id}>
          <div className="user-row"><div className="user-message">{turn.question}</div></div>
          {turn.error && <div className="assistant-block status-card status-error"><div><strong>暂时无法连接问答服务</strong><p>{turn.error} 你的问题已放回输入框。</p></div><button type="button" className="secondary-button" onClick={() => void submitQuestionText(turn.question)}>重试</button></div>}
          {turn.answer && <article className={`assistant-block answer-card ${turn.answer.fallbackReason ? "fallback" : ""}`}><div className="assistant-label"><span aria-hidden="true">✦</span>车书 AI</div>
            {turn.answer.answerSource !== "general" && (turn.answer.fallbackReason || turn.answer.boundaryResult) && <span className={`result-label boundary-${turn.answer.fallbackReason ? "unconfirmed" : turn.answer.boundaryResult}`}>{turn.answer.fallbackReason ? boundaryLabels.unconfirmed : boundaryLabels[turn.answer.boundaryResult!]}</span>}
            <div className="answer-text">{turn.answer.answer}</div>
            {turn.answer.fallbackReason === "clarification_required" && Boolean(turn.answer.clarificationOptions?.length) && <div className="clarification-options" aria-label="澄清选项">{turn.answer.clarificationOptions!.map((option) => <button type="button" className="suggestion-chip" key={option} onClick={() => { setQuestion(option); inputRef.current?.focus(); }}>{option}</button>)}</div>}
            <footer className="answer-actions"><div className="citation-actions">{turn.answer.citations.map((citation) => <button type="button" className="text-button" key={`${citation.id}:${citation.section}`} onClick={() => setSource(citation)}><span aria-hidden="true">▧</span>{citation.section}（{citation.title}）</button>)}</div><div className="utility-actions"><button type="button" className="icon-button" onClick={() => speak(turn.answer!.answerId, turn.answer!.speechText)}><span aria-hidden="true">{speakingId === turn.answer.answerId ? "Ⅱ" : "▶"}</span><span className="sr-only">{speakingId === turn.answer.answerId ? "暂停播报" : "播放回答"}</span></button><span className="action-divider" /><button type="button" className={`icon-button feedback-up ${turn.feedback === "up" ? "selected" : ""}`} aria-pressed={turn.feedback === "up"} onClick={() => void recordFeedback(turn.id, turn.answer!.answerId, "up")}><span aria-hidden="true">👍</span><span className="sr-only">点赞</span></button><button type="button" className={`icon-button feedback-down ${turn.feedback === "down" ? "selected" : ""}`} aria-pressed={turn.feedback === "down"} onClick={() => void recordFeedback(turn.id, turn.answer!.answerId, "down")}><span aria-hidden="true">👎</span><span className="sr-only">点踩</span></button></div></footer>
            {turn.feedback && !turn.feedbackError && <p className="feedback-note">感谢反馈，已记录本次选择。</p>}{turn.feedbackError && <p className="feedback-note error-text">{turn.feedbackError}</p>}
          </article>}
        </div>)}
        {loading && <div className="assistant-block loading-card" role="status"><span className="spinner" aria-hidden="true" /><div>我来看看这个问题…<p>车书回复可能会出错，请核查重要信息。</p></div></div>}<div ref={scrollEndRef} />
      </div>
      <form className="composer-wrap" onSubmit={(event) => void submitQuestion(event)}>{micNotice && <div className="mic-notice" role="status">{micNotice}<button type="button" onClick={() => setMicNotice("")} aria-label="关闭提示">×</button></div>}<div className="composer"><textarea ref={inputRef} value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="询问 Model 3 的功能或操作…" rows={1} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void submitQuestion(); } }} disabled={restoringSessions} /><div className="voice-control" ref={voiceMenuRef}>
          <button ref={voiceTriggerRef} type="button" className={`composer-button mic-button ${micState === "listening" ? "listening" : ""}`} onClick={() => micState === "listening" ? startVoiceQuestion() : setVoiceMenuOpen((open) => !open)} aria-label={micState === "listening" ? "停止语音输入" : "语音设置"} aria-pressed={micState === "listening"} aria-expanded={voiceMenuOpen} aria-controls="voice-settings" disabled={restoringSessions}>{micState === "listening" ? "■" : "🎙"}</button>
          {voiceMenuOpen && <section id="voice-settings" className="voice-popover" aria-label="语音设置">
            <h2>语音设置</h2>
            <button type="button" className="voice-input-action" onClick={startVoiceQuestion} disabled={loading || restoringSessions}>开始语音输入<span>点击后才使用麦克风</span></button>
            <label className="auto-play-option"><span>自动播放回复<small>默认关闭 · 记住本机偏好</small></span><input type="checkbox" role="switch" aria-label="自动播放回复" checked={autoPlay} onChange={(event) => toggleAutoPlay(event.target.checked)} /><span className="voice-switch" aria-hidden="true" /></label>
            <p>关闭后仍可点击回答下方的 ▶ 手动播放。</p>
          </section>}
        </div><button type="submit" className="composer-button send-button" disabled={!question.trim() || !selectedTrim || loading || restoringSessions} aria-label="发送问题">↑</button></div><p className="safety-copy">请优先在车辆静止且确保安全时操作；驾驶中请勿使用该功能。</p></form>
      </main>
      {source && <aside className="source-panel" aria-label="官方资料来源"><header className="source-header"><strong>官方资料来源</strong><button type="button" className="close-button" onClick={() => setSource(null)} aria-label="关闭来源详情">×</button></header><div className="source-content"><h2>{source.section}</h2><p className="source-meta">Tesla 中国官方资料 · 核验于 {source.retrievedAt}</p><div className="source-excerpt"><p>回答中有资料依据的部分参考了“{source.title}”的“{source.section}”；常识补充不属于该引用。</p><p>打开原文可查看完整资料和适用说明。</p></div></div><a className="source-link" href={source.url} target="_blank" rel="noreferrer">打开完整官方资料 <span aria-hidden="true">↗</span></a></aside>}
    </div>
    {menuSession && createPortal(<div ref={menuRef} className="session-menu" role="menu" aria-label={`对话操作：${menuSession.title ?? menuSession.turns[0]?.question ?? "新对话"}`} style={menuPosition}>
      <button type="button" role="menuitem" aria-label="重命名对话" onClick={() => startRenaming(menuSession)}><SessionMenuIcon name="rename" /><span>重命名</span></button>
      <button type="button" role="menuitem" aria-label={menuSession.pinned ? "取消置顶" : "置顶对话"} onClick={() => void togglePin(menuSession.id)}><SessionMenuIcon name={menuSession.pinned ? "unpin" : "pin"} /><span>{menuSession.pinned ? "取消置顶" : "置顶"}</span></button>
      <div className="session-menu-divider" role="separator" />
      <button type="button" role="menuitem" className="session-delete" aria-label="删除对话" disabled={menuSession.loading || deletingSessionIds.has(menuSession.id)} onClick={() => void removeConversation(menuSession)}><SessionMenuIcon name="delete" /><span>{deletingSessionIds.has(menuSession.id) ? "删除中…" : "删除"}</span></button>
    </div>, document.body)}
  </div>;
}

export default App;

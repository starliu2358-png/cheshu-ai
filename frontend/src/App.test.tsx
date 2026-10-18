import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

const vehicles = [
  {
    trimId: "model3-rwd-cn-current",
    displayName: "Model 3 后轮驱动版",
    model: "Model 3",
  },
  {
    trimId: "model3-lr-rwd-cn-current",
    displayName: "Model 3 长续航后轮驱动版",
    model: "Model 3",
  },
];

const answer = {
  answerId: "answer-1",
  conversationId: "conversation-1",
  answer: "打开充电端口\n\n车辆处于驻车挡时，可以按下充电电缆按钮。",
  boundaryResult: null,
  citations: [
    {
      id: "tesla-model3-charging",
      title: "Tesla 中国 Model 3 车主手册",
      section: "打开充电端口",
      url: "https://www.tesla.cn/ownersmanual/model3/zh_cn/example.html",
      retrievedAt: "2026-10-09",
    },
  ],
  speechText: "车辆处于驻车挡时，可以按下充电电缆按钮。",
  fallbackReason: null,
};

function response(payload: unknown, ok = true): Response {
  return {
    ok,
    json: async () => payload,
  } as Response;
}

function installFetchMock(mock: (url: string, options?: RequestInit) => Promise<Response>) {
  let nextId = 0;
  const stored = new Map<string, { conversationId: string; trimId: string; title: string | null; pinned: boolean; createdAt: string; updatedAt: string }>();
  vi.stubGlobal("fetch", vi.fn((url: string, options?: RequestInit) => {
    if (url === "/api/v1/conversations" && !options?.method) return Promise.resolve(response([]));
    if (url === "/api/v1/conversations" && options?.method === "POST") {
      const conversationId = `conversation-${++nextId}`;
      const item = { conversationId, trimId: JSON.parse(String(options?.body)).trimId, title: null, pinned: false, createdAt: "2026-10-16", updatedAt: "2026-10-16" };
      stored.set(conversationId, item);
      return Promise.resolve(response(item));
    }
    if (url.startsWith("/api/v1/conversations/") && options?.method === "PATCH") {
      const id = url.split("/").at(-1)!;
      const item = { ...stored.get(id)!, ...JSON.parse(String(options?.body)) };
      stored.set(id, item);
      return Promise.resolve(response(item));
    }
    return mock(url, options);
  }));
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((complete) => { resolve = complete; });
  return { promise, resolve };
}

function mockSpeech() {
  const speak = vi.fn();
  const cancel = vi.fn();
  const pause = vi.fn();
  const resume = vi.fn();
  class MockUtterance {
    lang = "";
    rate = 1;
    onend: (() => void) | null = null;
    onerror: (() => void) | null = null;

    constructor(public text: string) {}
  }
  vi.stubGlobal("SpeechSynthesisUtterance", MockUtterance);
  Object.defineProperty(window, "speechSynthesis", {
    configurable: true,
    value: { speak, cancel, pause, resume },
  });
  return { speak, cancel, pause, resume };
}

describe("车书 AI", () => {
  beforeEach(() => {
    window.localStorage.clear();
    mockSpeech();
    vi.stubGlobal("SpeechRecognition", undefined);
    vi.stubGlobal("webkitSpeechRecognition", undefined);
  });

  it("shows trims as read-only and opens voice settings without microphone access", async () => {
    installFetchMock(vi.fn().mockResolvedValue(response(vehicles)));
    const Recognition = vi.fn();
    vi.stubGlobal("SpeechRecognition", Recognition);
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText("后轮驱动版");
    expect(screen.getByText("长续航后轮驱动版")).toBeVisible();
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "语音设置" }));
    expect(screen.getByRole("switch", { name: "自动播放回复" })).not.toBeChecked();
    expect(Recognition).not.toHaveBeenCalled();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("switch")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "语音设置" })).toHaveFocus();
  });

  it("uses the latest playback setting while an answer is pending and persists it", async () => {
    const pending = deferred<Response>();
    installFetchMock(vi.fn().mockResolvedValueOnce(response(vehicles)).mockImplementationOnce(() => pending.promise));
    const user = userEvent.setup();
    const view = render(<App />);
    await screen.findByText("后轮驱动版");
    await user.click(screen.getByRole("button", { name: "如何打开充电口？" }));
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    await user.click(screen.getByRole("button", { name: "语音设置" }));
    await user.click(screen.getByRole("switch", { name: "自动播放回复" }));
    expect(localStorage.getItem("cheshu:autoPlayReplies")).toBe("true");
    await act(async () => pending.resolve(response(answer)));
    await waitFor(() => expect(window.speechSynthesis.speak).toHaveBeenCalledTimes(1));
    view.unmount();
    installFetchMock(vi.fn().mockResolvedValue(response(vehicles)));
    render(<App />);
    await screen.findByText("后轮驱动版");
    await user.click(screen.getByRole("button", { name: "语音设置" }));
    expect(screen.getByRole("switch")).toBeChecked();
    const cancellations = vi.mocked(window.speechSynthesis.cancel).mock.calls.length;
    await user.click(screen.getByRole("switch"));
    expect(window.speechSynthesis.cancel).toHaveBeenCalledTimes(cancellations + 1);
    expect(localStorage.getItem("cheshu:autoPlayReplies")).toBe("false");
  });

  it("does not autoplay a pending answer after the switch is turned off", async () => {
    localStorage.setItem("cheshu:autoPlayReplies", "true");
    const pending = deferred<Response>();
    installFetchMock(vi.fn().mockResolvedValueOnce(response(vehicles)).mockImplementationOnce(() => pending.promise));
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText("后轮驱动版");
    await user.click(screen.getByRole("button", { name: "如何打开充电口？" }));
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    await user.click(screen.getByRole("button", { name: "语音设置" }));
    await user.click(screen.getByRole("switch"));
    await act(async () => pending.resolve(response(answer)));
    expect(window.speechSynthesis.speak).not.toHaveBeenCalled();
  });

  it("shows general answers without source badges or official citations", async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(response(vehicles)).mockResolvedValueOnce(response({
      ...answer, answer: "低温通常会降低充电速度。", answerSource: "general",
      boundaryResult: "unconfirmed", citations: [],
    }));
    installFetchMock(fetchMock);
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText("后轮驱动版");
    await user.click(screen.getByRole("button", { name: "如何打开充电口？" }));
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    expect(await screen.findByText("低温通常会降低充电速度。")).toBeVisible();
    expect(screen.queryByText("ai 生成，仅供参考")).not.toBeInTheDocument();
    expect(screen.queryByText("模型常识补充 · 非官方结论")).not.toBeInTheDocument();
    expect(screen.queryByText("资料不足 / 无法确认")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Tesla 中国 Model 3 车主手册/ })).not.toBeInTheDocument();
  });

  it("keeps mixed-answer citations scoped without restoring badges", async () => {
    installFetchMock(vi.fn().mockResolvedValueOnce(response(vehicles)).mockResolvedValueOnce(response({
      ...answer, answerSource: "mixed", answer: "手册支持的内容。\n\n一般来说，可先确认需求。",
    })));
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText("后轮驱动版");
    await user.click(screen.getByRole("button", { name: "如何打开充电口？" }));
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    await user.click(await screen.findByRole("button", { name: /打开充电端口.*Tesla/ }));
    expect(screen.getByText(/常识补充不属于该引用/)).toBeVisible();
    expect(screen.queryByText("ai 生成，仅供参考")).not.toBeInTheDocument();
    expect(screen.queryByText("模型常识补充 · 非官方结论")).not.toBeInTheDocument();
  });

  it("loads trims and completes the charging-port answer flow", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(response(vehicles))
      .mockResolvedValueOnce(response(answer));
    installFetchMock(fetchMock);
    const user = userEvent.setup();

    render(<App />);

    expect(await screen.findByText("后轮驱动版")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "如何打开充电口？" }));
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    expect(await screen.findByText(/车辆处于驻车挡时，可以按下充电电缆按钮/)).toBeVisible();
    expect(fetchMock.mock.lastCall?.[0]).toBe("/api/v1/answers");
    expect(JSON.parse(String(fetchMock.mock.lastCall?.[1]?.body))).toEqual(expect.objectContaining({
      question: "如何打开充电口？",
      trimId: "model3-rwd-cn-current",
      conversationId: "conversation-1",
      turnId: expect.any(String),
    }));
    expect(window.speechSynthesis.speak).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "播放回答" }));
    expect(window.speechSynthesis.speak).toHaveBeenCalledTimes(1);
  });

  it("offers clarification choices and sends the selected detail in the same conversation", async () => {
    const clarification = {
      ...answer,
      answer: "你想了解哪项车辆功能？",
      speechText: "你想了解哪项车辆功能？",
      citations: [],
      fallbackReason: "clarification_required",
      clarificationOptions: ["哨兵模式录像如何保存？", "USB 存储设备有什么要求？"],
    };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response(vehicles))
      .mockResolvedValueOnce(response(clarification))
      .mockResolvedValueOnce(response({ ...answer, conversationId: "conversation-1" }));
    installFetchMock(fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("后轮驱动版");
    await user.type(screen.getByPlaceholderText("询问 Model 3 的功能或操作…"), "沙发JFK数据飞轮书法家");
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    await user.click(await screen.findByRole("button", { name: "USB 存储设备有什么要求？" }));

    expect(screen.getByPlaceholderText("询问 Model 3 的功能或操作…")).toHaveValue("USB 存储设备有什么要求？");
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    await waitFor(() => expect(JSON.parse(String(fetchMock.mock.lastCall?.[1]?.body))).toEqual(expect.objectContaining({
      question: "USB 存储设备有什么要求？",
      trimId: "model3-rwd-cn-current",
      conversationId: "conversation-1",
      turnId: expect.any(String),
    })));
  });

  it("opens the official citation in a separate responsive panel", async () => {
    installFetchMock(
      vi.fn().mockResolvedValueOnce(response(vehicles)).mockResolvedValueOnce(response(answer)),
    );
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("后轮驱动版");
    await user.click(screen.getByRole("button", { name: "如何打开充电口？" }));
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    await user.click(await screen.findByRole("button", { name: /打开充电端口/ }));

    expect(screen.getByRole("complementary", { name: "官方资料来源" })).toBeVisible();
    expect(screen.getByRole("link", { name: /打开完整官方资料/ })).toHaveAttribute(
      "href",
      answer.citations[0].url,
    );
    await user.click(screen.getByRole("button", { name: "关闭来源详情" }));
    expect(screen.queryByRole("complementary", { name: "官方资料来源" })).not.toBeInTheDocument();
  });

  it("records and switches feedback", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(response(vehicles))
      .mockResolvedValueOnce(response(answer))
      .mockResolvedValueOnce(response({ value: "up" }))
      .mockResolvedValueOnce(response({ value: "down" }));
    installFetchMock(fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("后轮驱动版");
    await user.click(screen.getByRole("button", { name: "如何打开充电口？" }));
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    const up = await screen.findByRole("button", { name: "点赞" });
    const down = screen.getByRole("button", { name: "点踩" });

    await user.click(up);
    await waitFor(() => expect(up).toHaveAttribute("aria-pressed", "true"));
    await user.click(down);
    await waitFor(() => expect(down).toHaveAttribute("aria-pressed", "true"));
    expect(up).toHaveAttribute("aria-pressed", "false");
  });

  it("keeps text input usable when microphone recognition is unavailable", async () => {
    installFetchMock(vi.fn().mockResolvedValue(response(vehicles)));
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("后轮驱动版");
    await user.click(screen.getByRole("button", { name: "语音设置" }));
    await user.click(screen.getByRole("button", { name: /开始语音输入/ }));

    expect(screen.getByText(/当前浏览器不支持语音识别/)).toBeVisible();
    expect(screen.getByPlaceholderText(/询问 Model 3/)).toBeEnabled();
  });

  it("keeps transcribed speech editable until the user explicitly sends it", async () => {
    let activeRecognition: MockRecognition | undefined;

    class MockRecognition {
      lang = "";
      continuous = false;
      interimResults = false;
      onstart: (() => void) | null = null;
      onresult: ((event: any) => void) | null = null;
      onerror: ((event: any) => void) | null = null;
      onend: (() => void) | null = null;

      constructor() {
        activeRecognition = this;
      }

      start() {
        this.onstart?.();
      }

      stop() {
        this.onend?.();
      }

      abort() {}

      emit(...segments: string[]) {
        this.onresult?.({
          resultIndex: 0,
          results: segments.map((transcript) => ({
            0: { transcript },
            isFinal: true,
          })),
        });
      }
    }

    vi.stubGlobal("SpeechRecognition", MockRecognition);
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(response(vehicles))
      .mockResolvedValueOnce(response(answer));
    installFetchMock(fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("后轮驱动版");
    await user.click(screen.getByRole("button", { name: "语音设置" }));
    await user.click(screen.getByRole("button", { name: /开始语音输入/ }));
    expect(screen.getByRole("button", { name: "停止语音输入" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(activeRecognition?.continuous).toBe(true);

    act(() => activeRecognition?.emit("哨兵模式", "如何使用"));

    const input = screen.getByPlaceholderText(/询问 Model 3/);
    expect(input).toHaveValue("哨兵模式如何使用");
    expect(fetchMock).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "停止语音输入" }));
    expect(screen.getByText(/语音已转写，你可以继续编辑/)).toBeVisible();
    await user.type(input, "？");
    expect(fetchMock).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "发送问题" }));
    expect(await screen.findByText(/车辆处于驻车挡时，可以按下充电电缆按钮/)).toBeVisible();
    expect(JSON.parse(String(fetchMock.mock.lastCall?.[1]?.body))).toEqual(expect.objectContaining({
      question: "哨兵模式如何使用？",
      trimId: "model3-rwd-cn-current",
      conversationId: "conversation-1",
    }));
  });

  it("does not show the removed vehicle-control scope card", async () => {
    installFetchMock(vi.fn().mockResolvedValue(response(vehicles)));
    render(<App />);

    await screen.findByText("后轮驱动版");
    expect(
      screen.queryByText("本产品不控制车辆，也不读取电量、位置或车速等实时状态。"),
    ).not.toBeInTheDocument();
  });

  it("shows a recoverable API error", async () => {
    installFetchMock(
      vi
        .fn()
        .mockResolvedValueOnce(response(vehicles))
        .mockResolvedValueOnce(response({ detail: "后端暂时不可用" }, false)),
    );
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("后轮驱动版");
    await user.click(screen.getByRole("button", { name: "如何打开充电口？" }));
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    expect(await screen.findByText("暂时无法连接问答服务")).toBeVisible();
    expect(screen.getByText(/后端暂时不可用/)).toBeVisible();
    expect(screen.getByRole("button", { name: "重试" })).toBeEnabled();
  });

  it("shows a three-state label only for an explicit availability result", async () => {
    const lccAnswer = {
      ...answer,
      answerId: "answer-lcc",
      answer: "Tesla 官方称为辅助转向，仅凭所选配置无法确认是否具备。",
      boundaryResult: "unconfirmed" as const,
    };
    installFetchMock(
      vi.fn().mockResolvedValueOnce(response(vehicles)).mockResolvedValueOnce(response(lccAnswer)),
    );
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("后轮驱动版");
    const input = screen.getByPlaceholderText(/询问 Model 3/);
    await user.type(input, "是否具备LCC功能？");
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    expect(await screen.findByText("资料不足 / 无法确认")).toBeVisible();
  });

  it("keeps earlier turns visible and sends conversation context for a follow-up", async () => {
    const followUpAnswer = {
      ...answer,
      answerId: "answer-2",
      answer: "不插入符合要求的 USB 存储设备时，哨兵模式无法保存录像。",
    };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response(vehicles))
      .mockResolvedValueOnce(response(answer))
      .mockResolvedValueOnce(response(followUpAnswer));
    installFetchMock(fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("后轮驱动版");
    const input = screen.getByPlaceholderText(/询问 Model 3/);
    await user.type(input, "哨兵模式怎么使用？");
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    await screen.findByText(/车辆处于驻车挡时/);
    await user.type(input, "不插U盘呢？");
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    expect(await screen.findByText(/无法保存录像/)).toBeVisible();
    expect(screen.getAllByText("哨兵模式怎么使用？").some((item) => item.classList.contains("user-message"))).toBe(true);
    expect(screen.getByText("不插U盘呢？")).toBeVisible();
    expect(JSON.parse(String(fetchMock.mock.lastCall?.[1]?.body))).toEqual(expect.objectContaining({
      question: "不插U盘呢？",
      trimId: "model3-rwd-cn-current",
      conversationId: "conversation-1",
    }));
  });

  it("keeps an unfinished first answer in its original chat while a new chat runs", async () => {
    localStorage.setItem("cheshu:autoPlayReplies", "true");
    const first = deferred<Response>();
    const second = deferred<Response>();
    const fetchMock = vi.fn((url: string, options?: RequestInit) => {
      if (url === "/api/v1/vehicles") return Promise.resolve(response(vehicles));
      if (url !== "/api/v1/answers") throw new Error(`Unexpected request: ${url}`);
      const question = JSON.parse(String(options?.body)).question;
      if (question === "第一问") return first.promise;
      if (question === "第二问") return second.promise;
      return Promise.resolve(response({ ...answer, conversationId: "conversation-first", answer: "旧对话的追问回答" }));
    });
    installFetchMock(fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("后轮驱动版");
    const input = screen.getByPlaceholderText(/询问 Model 3/);
    await user.type(input, "第一问");
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    expect(screen.getByText("我来看看这个问题…")).toBeVisible();
    expect(screen.getByText("车书回复可能会出错，请核查重要信息。")).toBeVisible();

    await user.click(screen.getByRole("button", { name: "＋ 新对话" }));
    expect(screen.getByRole("button", { name: /切换到对话 1：第一问，回答中/ })).toBeVisible();
    expect(screen.getByRole("button", { name: "发送问题" })).toBeDisabled();
    await user.type(input, "第二问");
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    await act(async () => first.resolve(response({ ...answer, conversationId: "conversation-first", answer: "第一条回答", speechText: "第一条回答" })));
    expect(screen.queryByText("第一条回答")).not.toBeInTheDocument();
    expect(window.speechSynthesis.speak).not.toHaveBeenCalled();
    await act(async () => second.resolve(response({ ...answer, answerId: "answer-2", conversationId: "conversation-second", answer: "第二条回答", speechText: "第二条回答" })));
    expect(screen.getByText("第二条回答")).toBeVisible();
    expect(screen.queryByText("第一条回答")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /切换到对话 1：第一问/ }));
    expect(screen.getByText("第一条回答")).toBeVisible();
    expect(screen.queryByText("第二条回答")).not.toBeInTheDocument();
    await user.type(input, "旧追问");
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    expect(await screen.findByText("旧对话的追问回答")).toBeVisible();
    expect(JSON.parse(String(fetchMock.mock.lastCall?.[1]?.body))).toEqual(expect.objectContaining({
      question: "旧追问", trimId: "model3-rwd-cn-current", conversationId: "conversation-first",
    }));
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes("/conversations/"))).toBe(false);
  });

  it("does not delete an established chat when its follow-up finishes after switching", async () => {
    const pending = deferred<Response>();
    const fetchMock = vi.fn((url: string, options?: RequestInit) => {
      if (url === "/api/v1/vehicles") return Promise.resolve(response(vehicles));
      if (url !== "/api/v1/answers") throw new Error(`Unexpected request: ${url}`);
      const question = JSON.parse(String(options?.body)).question;
      return question === "追问" ? pending.promise : Promise.resolve(response(answer));
    });
    installFetchMock(fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("后轮驱动版");
    const input = screen.getByPlaceholderText(/询问 Model 3/);
    await user.type(input, "原问题");
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    await screen.findByText(/车辆处于驻车挡时/);
    await user.type(input, "追问");
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    expect(JSON.parse(String(fetchMock.mock.lastCall?.[1]?.body))).toEqual(expect.objectContaining({
      question: "追问", trimId: "model3-rwd-cn-current", conversationId: "conversation-1",
    }));

    await user.click(screen.getByRole("button", { name: "＋ 新对话" }));
    await act(async () => pending.resolve(response({ ...answer, answerId: "answer-follow-up", answer: "追问的回答" })));
    expect(screen.queryByText("追问的回答")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /切换到对话 1：原问题/ }));
    expect(screen.getByText("追问的回答")).toBeVisible();
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes("/conversations/"))).toBe(false);
  });

  it("renames, pins, and explicitly deletes an established conversation", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response(vehicles))
      .mockResolvedValueOnce(response(answer))
      .mockResolvedValueOnce({ ok: true, status: 204 } as Response);
    installFetchMock(fetchMock);
    vi.spyOn(window, "confirm").mockReturnValue(true);
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("后轮驱动版");
    await user.click(screen.getByRole("button", { name: "如何打开充电口？" }));
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    await screen.findByText(/车辆处于驻车挡时/);
    await user.click(screen.getByRole("button", { name: "管理对话：如何打开充电口？" }));
    await user.click(screen.getByRole("menuitem", { name: "重命名对话" }));
    const title = screen.getByRole("textbox", { name: "对话名称" });
    await user.clear(title);
    await user.type(title, "充电问题");
    await user.click(screen.getByRole("button", { name: "保存" }));
    expect(screen.getByRole("button", { name: /切换到对话 1：充电问题/ })).toBeVisible();

    await user.click(screen.getByRole("button", { name: "管理对话：充电问题" }));
    await user.click(screen.getByRole("menuitem", { name: "置顶对话" }));
    expect(screen.getByLabelText("已置顶")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "管理对话：充电问题" }));
    await user.click(screen.getByRole("menuitem", { name: "删除对话" }));

    await waitFor(() => expect(fetchMock).toHaveBeenLastCalledWith(
      "/api/v1/conversations/conversation-1", { method: "DELETE" },
    ));
    expect(screen.queryByRole("button", { name: /切换到对话 1：充电问题/ })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /切换到对话 1：新对话/ })).toBeVisible();
  });

  it("does not offer deletion while a conversation is answering", async () => {
    const pending = deferred<Response>();
    const fetchMock = vi.fn((url: string) => url === "/api/v1/vehicles"
      ? Promise.resolve(response(vehicles)) : pending.promise);
    installFetchMock(fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("后轮驱动版");
    await user.type(screen.getByPlaceholderText(/询问 Model 3/), "哨兵模式怎么用？");
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    await user.click(screen.getByRole("button", { name: "管理对话：哨兵模式怎么用？" }));
    expect(screen.getByRole("menuitem", { name: "删除对话" })).toBeDisabled();
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes("/conversations/"))).toBe(false);
    await act(async () => pending.resolve(response(answer)));
  });

  it("moves a pinned chat ahead of other chats and can unpin it", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response(vehicles))
      .mockResolvedValueOnce(response(answer))
      .mockResolvedValueOnce(response({ ...answer, answerId: "answer-2", conversationId: "conversation-2" }));
    installFetchMock(fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("后轮驱动版");
    await user.type(screen.getByPlaceholderText(/询问 Model 3/), "第一问");
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    await screen.findByText(/车辆处于驻车挡时/);
    await user.click(screen.getByRole("button", { name: "＋ 新对话" }));
    await user.type(screen.getByPlaceholderText(/询问 Model 3/), "第二问");
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    await screen.findAllByText(/车辆处于驻车挡时/);
    await user.click(screen.getByRole("button", { name: "管理对话：第二问" }));
    await user.click(screen.getByRole("menuitem", { name: "置顶对话" }));

    expect(screen.getByRole("button", { name: /切换到对话 1：第二问/ })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "管理对话：第二问" }));
    await user.click(screen.getByRole("menuitem", { name: "取消置顶" }));
    expect(screen.getByRole("button", { name: /切换到对话 1：第一问/ })).toBeVisible();
  });

  it("keeps a chat when deletion is cancelled or the API rejects it", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response(vehicles))
      .mockResolvedValueOnce(response(answer))
      .mockResolvedValueOnce(response({ detail: "无法删除" }, false));
    installFetchMock(fetchMock);
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true);
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("后轮驱动版");
    await user.type(screen.getByPlaceholderText(/询问 Model 3/), "充电问题");
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    await screen.findByText(/车辆处于驻车挡时/);
    await user.click(screen.getByRole("button", { name: "管理对话：充电问题" }));
    await user.click(screen.getByRole("menuitem", { name: "删除对话" }));
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(screen.getByRole("button", { name: /切换到对话 1：充电问题/ })).toBeVisible();

    await user.click(screen.getByRole("menuitem", { name: "删除对话" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("无法清理旧对话");
    expect(screen.getByRole("button", { name: /切换到对话 1：充电问题/ })).toBeVisible();
    expect(confirm).toHaveBeenCalledTimes(2);
  });

  it("renders the three-action menu above the scrolling list and dismisses it", async () => {
    installFetchMock(vi.fn().mockResolvedValue(response(vehicles)));
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("后轮驱动版");
    const trigger = screen.getByRole("button", { name: "管理对话：新对话" });
    await user.click(trigger);
    const menu = screen.getByRole("menu", { name: "对话操作：新对话" });
    expect(menu.parentElement).toBe(document.body);
    expect(screen.getAllByRole("menuitem")).toHaveLength(3);
    expect(trigger).toHaveAttribute("aria-expanded", "true");

    await user.click(screen.getByRole("heading", { name: "建议问题" }));
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
    await user.click(trigger);
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });

  it("restores renamed, pinned conversation and its answer after remount", async () => {
    const saved = new Map<string, {
      conversationId: string; trimId: string; title: string | null; pinned: boolean;
      createdAt: string; updatedAt: string;
      turns: Array<{ id: string; question: string; answer: typeof answer; feedback: null }>;
    }>();
    const fetchMock = vi.fn((url: string, options?: RequestInit) => {
      if (url === "/api/v1/vehicles") return Promise.resolve(response(vehicles));
      if (url === "/api/v1/conversations" && !options?.method) {
        return Promise.resolve(response([...saved.values()].map(({ turns: _turns, ...summary }) => summary)));
      }
      if (url === "/api/v1/conversations" && options?.method === "POST") {
        const item = { conversationId: "conversation-1", trimId: JSON.parse(String(options.body)).trimId, title: null, pinned: false, createdAt: "2026-10-16", updatedAt: "2026-10-16", turns: [] };
        saved.set(item.conversationId, item);
        return Promise.resolve(response(item));
      }
      if (url === "/api/v1/conversations/conversation-1" && !options?.method) {
        return Promise.resolve(response(saved.get("conversation-1")));
      }
      if (url === "/api/v1/conversations/conversation-1" && options?.method === "PATCH") {
        const item = Object.assign(saved.get("conversation-1")!, JSON.parse(String(options.body)));
        return Promise.resolve(response(item));
      }
      if (url === "/api/v1/answers") {
        const body = JSON.parse(String(options?.body));
        const result = { ...answer, conversationId: body.conversationId };
        saved.get(body.conversationId)!.turns.push({ id: body.turnId, question: body.question, answer: result, feedback: null });
        return Promise.resolve(response(result));
      }
      throw new Error(`Unexpected request: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    const first = render(<App />);
    await screen.findByText("后轮驱动版");
    await user.type(screen.getByPlaceholderText(/询问 Model 3/), "如何打开充电口？");
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    await screen.findByText(/车辆处于驻车挡时/);
    await user.click(screen.getByRole("button", { name: "管理对话：如何打开充电口？" }));
    await user.click(screen.getByRole("menuitem", { name: "重命名对话" }));
    await user.clear(screen.getByRole("textbox", { name: "对话名称" }));
    await user.type(screen.getByRole("textbox", { name: "对话名称" }), "我的充电笔记");
    await user.click(screen.getByRole("button", { name: "保存" }));
    await screen.findByRole("button", { name: /切换到对话 1：我的充电笔记/ });
    await user.click(screen.getByRole("button", { name: "管理对话：我的充电笔记" }));
    await user.click(screen.getByRole("menuitem", { name: "置顶对话" }));
    await waitFor(() => expect(saved.get("conversation-1")?.pinned).toBe(true));

    first.unmount();
    render(<App />);
    expect(await screen.findByRole("button", { name: /切换到对话 1：我的充电笔记/ })).toBeVisible();
    expect(screen.getByLabelText("已置顶")).toBeVisible();
    expect(await screen.findByText(/车辆处于驻车挡时/)).toBeVisible();
    expect(window.localStorage.getItem("cheshu:lastConversationId")).toBe("conversation-1");
  });

  it("recovers an answer that was still processing during page reload", async () => {
    const summary = { conversationId: "conversation-1", trimId: "model3-rwd-cn-current", title: "充电追问", pinned: false, createdAt: "2026-10-16", updatedAt: "2026-10-16" };
    let reads = 0;
    vi.stubGlobal("fetch", vi.fn((url: string) => {
      if (url === "/api/v1/vehicles") return Promise.resolve(response(vehicles));
      if (url === "/api/v1/conversations") return Promise.resolve(response([summary]));
      if (url === "/api/v1/conversations/conversation-1") {
        reads += 1;
        return Promise.resolve(response({ ...summary, turns: [{
          id: "pending-1", question: "如何打开充电口？",
          answer: reads === 1 ? null : answer,
          feedback: null, status: reads === 1 ? "processing" : "completed",
        }] }));
      }
      throw new Error(`Unexpected request: ${url}`);
    }));
    render(<App />);

    expect(await screen.findByText(/车辆处于驻车挡时/)).toBeVisible();
    expect(screen.getByRole("button", { name: /切换到对话 1：充电追问/ })).toBeVisible();
    expect(reads).toBeGreaterThanOrEqual(2);
  });
});

# History Truncation TODO & Improvements

## Current Status

As of the current version, the history truncation and summarization system in ChainLite (`history_truncator_config`) exclusively processes **Tool Output** (`ToolReturnPart`).

When a tool returns a result that exceeds the `truncation_threshold`, the system handles it according to the configured mode (`simple`, `auto`, or `custom`). However, regular **User Messages** and **Assistant Responses** are not currently subject to this truncation or summarization logic.

## Proposed Improvements

### 1. General Message Truncation

Extend the truncation logic to support all message types.

- **Goal**: Prevent very long user prompts or AI responses from exhausting the context window.
- **Strategy**: Similar to tool outputs, use a threshold to trigger summarization or character-based truncation for long text in any part of a `ModelRequest` or `ModelResponse`.

### 2. Token-based Truncation

Shift from character-based thresholds to token-based counts.

- **Goal**: Align better with how LLM providers bill and restrict context (tokens vs. characters).
- **Strategy**: Integrate a tokenizer (like `tiktoken`) to accurately measure the cost of history and trigger truncation based on token budget.

### 3. Sliding Window Summarization

Instead of summarizing individual parts, summarize the entire "older" portion of the history.

- **Goal**: Maintain a coherent "memory" of the conversation while keeping the active context small.
- **Strategy**: When history reaches a certain size, take the first N messages and compress them into a single summary message, keeping the most recent M messages in their raw form.

### 4. Configurable Retention Policies

Allow users to define which parts of the history are "sacred" and should never be truncated.

- **Goal**: Preserve critical instructions or user identity throughout a long session.
- **Strategy**: Add tags or config options to protect specific messages (e.g., the first user request or specific system additions) from the truncator.

### 5. Multi-stage Summarization

Use hierarchical summarization for extremely long sessions.

- **Goal**: Support sessions with hundreds of messages.
- **Strategy**: Summarize summersaries recursively or use a vector database to retrieve relevant historical summaries as needed.

目前的 chainlite 系統中，關於 Truncator (摘要/截斷) 的執行時機與邏輯，針對你提到的「在一個 loop 中 call 了多個 agent/tool」的情況，具體運作方式如下：

1. 摘要發生的時機：執行完成後 (Post-Run)
   chainlite 是在
   run()
   或
   arun()
   整個執行流程結束後，才統一處理這一次執行產生的所有新訊息。

單次 agent.run() 內部呼叫多個 Tool： 如果在一次 agent.run() 中，LLM 決定連續呼叫 5 個 Tool，這 5 個 Tool 的執行結果會先以完整內容存在於 pydantic-ai 的當前對話上下文中。這意味著：在這一次執行過程中，後續的 Tool 呼叫或最終回答，都能看到前面 Tool 的「完整原始內容」，不會在執行中途被摘要。
摘要觸發點： 當
run()
結束並返回結果時，chainlite 會呼叫 result.new_messages() 取得這次運作產生的所有新訊息（包含所有的 ToolReturnPart），然後將這些訊息一次性丟給 HistoryManager.add_messages()。這時 Truncator 才會介入。2. 摘要的對象：訊息逐一檢查
當訊息進到
HistoryManager
時，Truncator（如
AutoSummarizer
）會遍歷這次進來的所有新訊息：

它會針對每一個 ToolReturnPart（工具回傳的部分）單獨檢查。
只有超過 truncation_threshold（預設 5000 字元）的 Tool 輸出才會被執行摘要。
如果這次 loop 產生了 3 個很大的 Tool 輸出，Truncator 確實會對這 3 個輸出各做一次摘要。3. 不同的 Loop 情境分析
情境 A：單次 agent.run() 內的多工具循環
python

# 一次執行，內含多次 tool call

result = agent.run("請幫我搜尋 A, B, C 的資料")
運作： 執行中不摘要。執行結束後，若 A, B, C 的結果太長，會一次性摘要完畢並存入 History。
優點： LLM 在當次思考中能看到最精確的資料。
缺點： 如果 A 的結果超級大（例如 10 萬字），當次執行後續的 B 呼叫可能會因為 token 超限而失敗。
情境 B：外部 Loop 多次呼叫 chainlite.run()
python

# 外部手寫 loop

for task in ["A", "B", "C"]:
res = agent.run(task) # 每次結束都會更新 history 並摘要
運作： 這就是「每 call 一次就摘要一次」。執行完 A 後立刻摘要，執行 B 時看到的是 A 的摘要。
總結
不是「即時」摘要： 在同一次
run
指令中，不會每 call 一次 tool 就立刻摘要；摘要發生在整趟
run
結束存入歷史紀錄的時候。
單獨檢查： 只要有設定 truncator，存入歷史時，每一條超過門檻的 Tool 訊息都會被單獨摘要。
下一輪生效： 摘要過後的內容，只有在下一個 agent.run() 被呼叫時，LLM 才會讀取到那個精簡過的版本。
如果你希望在一次長對話的 Tool Loop 中，讓後面的 Tool 呼叫能省 token，目前的設計是建議將任務拆解，或是增加 truncation_threshold 的敏感度。

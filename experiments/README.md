# Token Usage Ablation 實驗說明

本資料夾的實驗用來回答一個核心問題：

> 在多輪、工具驅動（tool-calling）的 Agent 場景中，不同歷史壓縮策略會如何影響整體 token 成本與可擴展性？

```
pixi run python experiments/ablation_token_usage.py

pixi run python experiments/ablation_token_usage.py \
  --num-runs 10 \
  --tool-calls-per-run 3 \
  --tool-output-size 5000

```

---

## 實驗目的

這個 ablation（消融）實驗的主要目的，不是追求回答品質本身，而是**量化上下文管理策略對 token 消耗的影響**。具體來說，目標有四個：

1. 建立 token 成本基線（Baseline）
   - 以 `no_truncation`（不做歷史壓縮）作為對照組，觀察 token 隨輪次累積的自然增長曲線。

2. 評估壓縮策略是否有效降本
   - 比較 `post_run_compaction_auto` 與 `post_and_in_run_compaction_auto`（同時啟用 post-run 與 in-run 壓縮），檢查它們是否能降低主代理（primary agent）的 input token 成長速度。

3. 量化壓縮本身的代價
   - 壓縮策略通常依賴 summarizer，summarizer 也會消耗 token。本實驗會把這部分獨立統計，避免只看主流程而低估真實成本。

4. 找到「總成本」最優策略
   - 最終比較的是 `Primary + Summarizer` 的 grand total，而不是單看某一側。這能幫助決策：在長會話下，哪種策略整體最划算。

---

## 為什麼要做這個實驗

在有工具呼叫的多輪任務中，歷史會快速膨脹，尤其當：

- 每輪工具輸出很長（例如檢索結果、JSON、混合格式文字）
- 每輪需要多次工具呼叫
- 任務輪次增加（例如長鏈式推理、分步分析）

若沒有上下文管理，模型容易遇到兩種問題：

- 成本快速上升（input token 隨歷史成長而遞增）
- 接近或超過 context window（觸發截斷或失敗）

因此本實驗把焦點放在「成本與上下文安全」兩件事，而非模型能力本身。

---

## 實驗想回答的關鍵問題

1. 不同壓縮策略下，`每輪` 與 `累積` input token 的曲線差異是什麼？
2. 壓縮策略省下的主流程 token，是否會被 summarizer 開銷抵消？
3. 在工具輸出變長、工具呼叫次數增加時，哪種策略更穩定？
4. 當接近 context 上限時，保守安全機制（context guard）是否有效降低爆窗風險？

---

## 受控變因與比較設計

實驗固定同一組問題模板與同一模型，僅改變歷史壓縮機制：

- `no_truncation`
- `post_run_compaction_auto`
- `post_and_in_run_compaction_auto`

同時透過可控參數調整壓力場景：

- `num_runs`
- `tool_calls_per_run`
- `tool_output_size`
- 各種 truncation threshold

工具輸出使用「可重現但具變化」的假資料（主題、長度、格式、噪聲比例會變動），用來模擬實際工具回傳在結構與大小上的波動。

---

## 成功判準與解讀原則

本實驗的主要觀察指標：

- 主代理 token（Primary Agent）
- 壓縮器 token（Summarizer Overhead）
- 總 token（Grand Total）
- 每輪與累積趨勢圖

解讀時建議遵循：

1. 先看 `Grand Total`，再看主流程細節
2. 比較曲線斜率，不只看單點
3. 在高壓參數（高輪次/高工具輸出）下驗證策略穩定性

---

## 實驗邊界與限制

- 這是「token 成本」實驗，不直接評估答案品質
- 使用假工具資料，結論偏向上下文與成本行為，不代表特定真實資料源的品質表現
- 結果會受模型、prompt、閾值設定影響，需在目標工作負載下重跑驗證

---

## 一句話總結

這個實驗的本質，是在長上下文 Agent 系統裡，找出「不爆窗、又省 token」的歷史管理策略，並用可量化數據比較各策略的真實總成本。

---
topic: "Bounded web search integration for a local multi-provider Auto-Research platform"
created: "2026-03-23"
session_id: "v317-benchmark-qwen3.5-35b-a3b-1774252779"
provider: "lmstudio"
model: "qwen/qwen3.5-35b-a3b"
sources_count: 1
---

## 摘要
本設計文件描述了一個本地多提供者 Auto-Research 平台，透過整合受限的網頁搜尋功能來增強研究品質。系統採用 DuckDuckGo HTML 搜尋（無需 API 金鑰），並結合 SourceFetcher 進行內容抓取與清洗，最終將解析後的資料寫入會話目錄。整個流程強調本地執行，避免依賴雲端 LLM API，並透過嚴格的資源限制與品質閘道確保穩定性與可追溯性。

## 關鍵發現
*   **搜尋與抓取機制**：使用 DuckDuckGo HTML 搜尋，SourceFetcher 負責移除腳本、樣式及導航元素。每頁內容限制為最多 5000 字，單頁 fetch 超時設為 15 秒。
*   **資源約束**：每次運行最多抓取 3 個網頁來源，所有執行均在本地完成。
*   **資料合約**：解析後的來源遵循穩定格式，包含 title、source_url、fetched_at、source_type 及 word_count。
*   **提供者架構**：支援 Ollama、LM Studio 與 vLLM，具備推理感知自動選擇、快取機制（316x 加速）及電路斷開器。
*   **控制平面**：透過 Telegram 提供對話介面，採用三層混合意圖解析（關鍵字 → LLM → 澄清），並執行政策閘道動作（SAFE/CONFIRM/DISABLED）。
*   **追蹤與品質**：結果記錄於 results.tsv（19 欄位），品質閘道衡量覆蓋率、結構完整性及來源證明。

## 後續建議
*   **維持資源限制**：嚴格遵守每次運行最多 3 頁及每頁 5000 字的約束，以確保系統穩定性與可預測性。
*   **強化本地執行**：持續保持所有執行在本地進行，避免引入雲端 LLM API 依賴。
*   **優化快取效能**：利用現有的準備狀態快取機制，發揮其 316x 的加速效果以提升回應速度。
*   **監控品質閘道**：定期檢視覆蓋率、結構與來源多樣性等指標，確保研究成果符合既定標準。

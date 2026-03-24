---
topic: "Bounded web search integration for a local multi-provider Auto-Research platform"
created: "2026-03-23"
session_id: "v317-benchmark-nemotron-3-nano-1774252973"
provider: "lmstudio"
model: "nvidia/nemotron-3-nano"
sources_count: 1
---

## 摘要
本文件概述局部網路搜尋整合於本地多供應商 Auto‑Research 平台的設計與流程，說明如何透過 DuckDuckGo HTML 搜尋、SourceFetcher 解析與寫入解析稿，以受限資源與時間條件提升研究品質。

## 關鍵發現
- **搜尋與抓取**：使用 DuckDuckGo HTML 搜尋（不需 API 金鑰），最多抓取 **3 個頁面**，每頁字數上限 **5000 個字**，每頁抓取超時 **15 秒**。  
- **解析與標準化**：SourceFetcher 移除 `<script>`、`<style>`、`<nav>`、`<footer>` 等元素，並依照固定前置資料（title、source_url、fetched_at、source_type、word_count）撰寫解析稿。  
- **流水線**：`WebSearchAdapter → List[SearchResult] → SourceFetcher → FetchResult → write_to_session() → parsed/web-{slug}.md → SynthesizerService.synthesize() → quality gate → results.tsv`。  
- **供應商支援**：平台同時支援 Ollama、LM Studio、vLLM，具備自動選擇、就緒快取（316× 加速）與斷路器機制。  
- **控制平面**：Telegram 介面結合 3 層混合意圖解析（關鍵字 → LLM → 說明），並通過政策門檻（SAFE/CONFIRM/DISABLED）執行動作。  
- **結果追蹤**：所有執行結果寫入 `results.tsv`，共 19 個欄位包括 `run_kind`、`search_result_count`、`fetched_source_count`、`program_hash` 等，並透過覆蓋率、結構、證據多樣性等品質門檻驗證。

## 後續建議
- **擴充搜尋上限**：評估在資源允許的範圍內適度增加抓取頁面數，以提升證據多樣性。  
- **字數彈性**：可探索動態字數上限或分段壓縮策略，以減少資訊遺失。  
- **品質門檻優化**：針對覆蓋率與結構指標調整門檻值，以提高合成結果的可靠性。  
- **多模態整合**：考慮將圖像或表格解析納入解析流程，以擴充證據類型。  
- **監控與日誌**：強化對每一步執行時間與資源使用的監控，協助偵測瓶頸與異常行為。

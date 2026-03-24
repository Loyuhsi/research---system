---
topic: "Bounded web search integration for a local multi-provider Auto-Research platform"
created: "2026-03-23"
session_id: "v317-benchmark-nemotron-3-nano-1774252507"
provider: "lmstudio"
model: "nvidia/nemotron-3-nano"
sources_count: 1
---

## 摘要
本設計說明局部網路搜尋整合於本地多供應商 Auto‑Research 平台的實作方式，強調資源受限、全程本地執行與標準化資料流。

## 關鍵發現
- 使用 DuckDuckGo HTML 搜尋（純標準庫，無 API 金鑰）取得搜尋結果。  
- `SourceFetcher` 會剔除 `<script>`、`<style>`、`<nav>`、`<footer>` 等元素，只保留正文。  
- 每次執行最多抓取 **3 個頁面**，每頁字數上限 **5000 個字**，逾時 **15 秒**／頁。  
- 解析後的來源遵循統一前置資料（frontmatter）合約：`title、source_url、fetched_at、source_type、word_count`。  
- 整合流程為：`WebSearchAdapter → List[SearchResult] → SourceFetcher → FetchResult → write_to_session() → parsed/web-{slug}.md → SynthesizerService.synthesize() → quality gate → results.tsv`。  
- 供應商層支援 Ollama、LM Studio、vLLM，具備推理感知的自動選擇、 readiness 緩存（提升 316 倍）與斷路器機制。  
- Telegram 控制平面提供 3 層混合意圖解析（關鍵字 → LLM → 需要確認）與政策管控行動（SAFE/CONFIRM/DISABLED）。  
- 結果記錄於 `results.tsv`，包含 19 個欄位如 `run_kind、search_result_count、fetched_source_count、program_hash`。  
- 品質門檻檢查包括 **覆蓋度（n‑gram 重疊）**、**結構（frontmatter+標題）**、**來源多樣性（evidence+diversity）**。

## 後續建議
- 在抓取前先設定更嚴格的關鍵字過濾，降低無關頁面的比例。  
- 可考慮擴充前置資料欄位，加入 `source_language` 或 `confidence_score` 以提升合成品質。  
- 針對 Telegram 介面加入更多篩選規則，減少政策違規的自動執行風險。  
- 定期檢查 `results.tsv` 的欄位完整性，確保追蹤與可重現性需求持續滿足。

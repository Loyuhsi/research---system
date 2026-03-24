---
topic: "Bounded web search integration for a local multi-provider Auto-Research platform"
created: "2026-03-23"
session_id: "v317-benchmark-nemotron-3-nano-1774260954"
provider: "lmstudio"
model: "nvidia/nemotron-3-nano"
sources_count: 1
---

## 摘要
本文件描述了本地多供應商 Auto‑Research 平台的有限網路搜索整合設計，強調在不使用雲端 LLM API 的情況下，透過 DuckDuckGo HTML 搜尋、SourceFetcher 解析與前置作業，將搜索結果寫入會話解析目錄，並交給合成流程產生最終報告。

## 關鍵發現
- **有限頁面數量**：每次執行最多抓取 3 個頁面，以限制資源消耗。  
  > 「Maximum 3 fetched pages per run (bounded resource usage)」
- **字數上限**：每頁最多 5000 個單字，並以決定性的詞界切割進行截斷。  
  > 「Maximum 5000 words per page (deterministic word-boundary truncation)」
- **解析流程**：SourceFetcher 會剔除腳本、樣式、導航與頁腳元素，只保留正文，並依照固定前置資料格式（title、source_url、fetched_at、source_type、word_count）寫入 `parsed/web-{slug}.md`。  
  > 「All execution local — no cloud LLM APIs」  
  > 「Parsed sources follow a stable contract: title, source_url, fetched_at, source_type, word_count」
- **合成流程**：解析結果會送入 `SynthesizerService.synthesize()`，經過品質門檻檢查後，結果匯出至 `results.tsv`，其中包含 19 個欄位如 `run_kind`、`search_result_count`、`fetched_source_count` 等。  
  > 「Results are tracked in results.tsv with 19 columns including run_kind, search_result_count, fetched_source_count, and program_hash for traceability」
- **品質門檻**：包括覆蓋度（n‑gram 重疊）、結構（前置資料+標題）以及來源多樣性（evidence+diversity）等指標。  
  > 「Quality gates measure coverage (n-gram overlap), structure (frontmatter+headings), and provenance (evidence+diversity)」

## 後續建議
- 在實作時確保每次搜索的頁面數不超過 3，並嚴格控制單頁字數不超過 5000，以維持資源效率。  
- 針對解析後的前置資料格式進行單元測試，確保 `title、source_url、fetched_at、source_type、word_count` 欄位正確填入。  
- 在合成前加入額外的前置資料完整性檢查，以提升 `coverage` 與 `structure` 的品質評分。  
- 若需擴充搜索範圍，可考慮在不超過 3 頁限制內使用更精細的關鍵字或查詢語句，以提升證據多樣性。

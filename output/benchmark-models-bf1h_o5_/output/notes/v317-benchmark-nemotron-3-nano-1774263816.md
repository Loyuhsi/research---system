---
topic: "Bounded web search integration for a local multi-provider Auto-Research platform"
created: "2026-03-23"
session_id: "v317-benchmark-nemotron-3-nano-1774263816"
provider: "lmstudio"
model: "nvidia/nemotron-3-nano"
sources_count: 1
---

## 摘要
本設計說明局部網路搜尋整合於本地多供應商 Auto‑Research 平台的實作方式，強調資源受限、解析穩定與流程串接。

## 關鍵發現
- 搜尋受限：最多抓取 3 個頁面，每頁不超 5000 個字，逾時 15 秒。  
- 解析流程：使用 SourceFetcher 移除腳本、樣式、導航等元素，僅保留文字內容。  
- 稳定合約：解析結果以 title、source_url、fetched_at、source_type、word_count 等欄位寫入 `parsed/web-{slug}.md`。  
- 流程串接：WebSearchAdapter → List[SearchResult] → SourceFetcher → write_to_session() → SynthesizerService.synthesize() → quality gate → results.tsv。  
- 供應商支援：Ollama、LM Studio、vLLM 具備自動選擇、即時緩存（316× 加速）與斷路器機制。  
- 監控與追蹤：結果寫入 `results.tsv`（19 欄），包括 run_kind、search_result_count、fetched_source_count、program_hash 等欄位，並透過 coverage、structure、provenance 等品質門檻評估。

## 後續建議
- 可考慮擴充字數上限或調整抓取次數上限，以提升資訊完整度。  
- 針對品質門檻加入更細粒度的多樣性度量，以提升證據多元性。  
- 監測資源使用情況，動態調整 timeout 與字數限制，以平衡效能與品質。

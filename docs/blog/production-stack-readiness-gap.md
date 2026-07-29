---
description: "A measured vLLM Production Stack failure where 177/177 health probes passed but only 3/12 LLM requests completed, separating event-loop health from service readiness."
date: "2026-07-29"
language:
  - en
  - zh-Hant
image: "https://wayne.is-a.dev/assets/blog/production-stack-readiness.svg"
tags:
  - Serving
  - Kubernetes
  - vLLM
  - LMCache
  - Reliability
---

# 177/177 Healthy, 3/12 Complete: The LLM Readiness Gap｜177 次健康檢查全綠，只有 3/12 請求完成

*2026-07-29 · LLM serving / Kubernetes reliability*

<figure class="pb-article-hero">
  <img src="/assets/blog/production-stack-readiness.svg" alt="Kubernetes probes and LLM requests sharing a KV-aware router while tokenizer and backend work run on separate paths" loading="eager" decoding="async">
  <figcaption>Probe health and request completion travel through different dependencies.／健康檢查與請求完成會經過不同的相依元件。</figcaption>
</figure>

The router passed **177 out of 177 health checks**. Its health latency stayed below 15 ms, and Kubernetes recorded no probe failure or restart. Under the same test, only **3 out of 12 LLM requests completed** within the 14-second client timeout.

Router 通過了 **177／177 次健康檢查**，健康檢查延遲都低於 15 ms，Kubernetes 也沒有記錄任何 probe failure 或 restart。但在同一輪測試裡，12 個 LLM request 只有 **3 個在 14 秒的 client timeout 內完成**。
{ .pb-translation lang=zh-Hant }

Both results were correct. The event loop was responsive, while the request path was still waiting on tokenizer initialization and backend warm-up. A green `/health` endpoint proved one layer had recovered; it did not prove the service could complete traffic.

兩個結果都沒錯。Event loop 已經能正常回應，但 request path 仍卡在 tokenizer 初始化與 backend warm-up。`/health` 全綠只能證明其中一層恢復，不能證明服務已經能完成請求。
{ .pb-translation lang=zh-Hant }

## 1. The failure under test｜這次測的是什麼故障

The experiment reproduced [vLLM Production Stack issue #1016](https://github.com/vllm-project/production-stack/issues/1016), where the KV-aware router could block its uvicorn event loop during tokenization. When external access to the Hugging Face Hub was unavailable, `AutoTokenizer.from_pretrained` could take long enough to starve `/health`. Kubernetes then treated the router as unhealthy and restarted it even though the vLLM backend remained Ready.[^repro]

這次實驗重現的是 [vLLM Production Stack issue #1016](https://github.com/vllm-project/production-stack/issues/1016)。KV-aware router 進行 tokenization 時，可能阻塞 uvicorn event loop。當 router 無法連到 Hugging Face Hub，`AutoTokenizer.from_pretrained` 可能拖到 `/health` 無法回應。此時 vLLM backend 仍是 Ready，Kubernetes 卻會把 router 判定為 unhealthy 並重新啟動。[^repro]
{ .pb-translation lang=zh-Hant }

The bounded lab used:

測試環境如下：
{ .pb-translation lang=zh-Hant }

- **Hardware:** NVIDIA RTX PRO 6000 Blackwell, SM120, 96 GB.<br><span class="pb-inline-translation" lang="zh-Hant">**硬體：** NVIDIA RTX PRO 6000 Blackwell，SM120，96 GB。</span>
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`, pinned revision `7ae55760`, served as alias `default`.<br><span class="pb-inline-translation" lang="zh-Hant">**模型：** `Qwen/Qwen2.5-0.5B-Instruct`，固定 revision `7ae55760`，對外 alias 為 `default`。</span>
- **Serving path:** KV-aware routing, vLLM backend, LMCache controller and storage.<br><span class="pb-inline-translation" lang="zh-Hant">**Serving path：** KV-aware routing、vLLM backend、LMCache controller 與 storage。</span>
- **Traffic:** an exactly 30,000-token prompt every 15 seconds, with eight output tokens.<br><span class="pb-inline-translation" lang="zh-Hant">**流量：** 每 15 秒送出一個正好 30,000 tokens 的 prompt，並生成 8 個 tokens。</span>
- **Probe budget:** two seconds for startup, readiness, and liveness probes.<br><span class="pb-inline-translation" lang="zh-Hant">**Probe 預算：** startup、readiness 與 liveness probes 都是 2 秒。</span>

Kubernetes uses readiness probes to decide whether a Pod should receive traffic and liveness probes to decide whether it should be restarted.[^probes] The two-second budget made event-loop starvation visible as an operational failure rather than a slow log line.

Kubernetes 透過 readiness probe 決定 Pod 是否應繼續接收流量，透過 liveness probe 決定是否重新啟動 Pod。[^probes] 兩秒的 probe budget 讓 event-loop starvation 直接表現在維運結果上，而不只是一行「比較慢」的 log。
{ .pb-translation lang=zh-Hant }

## 2. Main failed at the health plane｜Main 在 health plane 失敗

With normal router egress, the control run completed all 12 requests and all 177 one-second health samples returned HTTP 200. Health latency peaked at 0.221 seconds, and the router had no restart.

Router 能正常連外時，control run 的 12 個 requests 全部完成，177 次每秒健康檢查也全部回傳 HTTP 200。Health latency 最高為 0.221 秒，router 沒有 restart。
{ .pb-translation lang=zh-Hant }

I then blocked external router egress while keeping DNS, backend `/tokenize`, the LMCache controller, and worker reachability available. On Production Stack main at `a0f4980a`, the first long request stalled before the backend fallback:

接著只封鎖 router 對外連線，DNS、backend `/tokenize`、LMCache controller 與 worker 仍保持可達。Production Stack main `a0f4980a` 上的第一個 long request，在進入 backend fallback 前就卡住：
{ .pb-translation lang=zh-Hant }

| Result／結果 | Normal egress／正常連外 | Restricted egress on main／Main 封鎖連外 |
| --- | ---: | ---: |
| Completed requests／完成請求 | 12/12 | failed during trial／測試期間失敗 |
| Health samples over 2 s／超過 2 秒的健康檢查 | 0 | 81 |
| Readiness failures | 0 | 18 |
| Liveness failures | 0 | 6 |
| Router restarts | 0 | 2 |
| Backend restarts | 0 | 0 |

The backend never became unhealthy. The router restarted because tokenization work occupied the same event loop that had to answer `/health`.[^repro]

Backend 從未變成 unhealthy。Router 會 restart，是因為 tokenization 工作占用了原本也要回應 `/health` 的 event loop。[^repro]
{ .pb-translation lang=zh-Hant }

## 3. What PR #1025 changed｜PR #1025 改了什麼

[Production Stack PR #1025](https://github.com/vllm-project/production-stack/pull/1025) moved the two blocking operations off the event loop with `asyncio.to_thread`:

[Production Stack PR #1025](https://github.com/vllm-project/production-stack/pull/1025) 使用 `asyncio.to_thread`，把兩個 blocking operations 移出 event loop：
{ .pb-translation lang=zh-Hant }

1. `AutoTokenizer.from_pretrained` for local tokenizer loading.
2. The synchronous `requests.post` call to the backend `/tokenize` fallback.

1. 用來載入 local tokenizer 的 `AutoTokenizer.from_pretrained`。
2. 呼叫 backend `/tokenize` fallback 的 synchronous `requests.post`。
{ .pb-translation lang=zh-Hant }

It also cached a failed tokenizer load so later requests could go directly to the backend fallback instead of repeating the same doomed attempt. The PR deliberately left the LMCache controller lookup unchanged because that path is an in-memory registry read in the tested version.[^pr]

PR 也會記住 tokenizer load 已經失敗，讓後續 requests 直接走 backend fallback，不必重複同一個注定失敗的嘗試。測試版本中的 LMCache controller lookup 是 in-memory registry read，因此這次沒有更動那條路徑。[^pr]
{ .pb-translation lang=zh-Hant }

At exact head `e104ed573`, the original health failure disappeared:

使用確切的 PR head `e104ed573` 後，原本的 health failure 消失了：
{ .pb-translation lang=zh-Hant }

| Metric／指標 | PR #1025 head `e104ed573` |
| --- | ---: |
| Health HTTP 200 | 177/177 |
| Health p95 | 0.007987 s |
| Health maximum | 0.014888 s |
| Probe failures | 0 |
| Router restarts | 0 |

Backend `/tokenize`, controller lookup, KV-aware routing, generation, and LMCache hits all executed during the trial. This confirmed the patch fixed the event-loop health failure without bypassing the serving path.[^validation]

測試期間，backend `/tokenize`、controller lookup、KV-aware routing、generation 與 LMCache hits 都有實際執行。這證明 patch 修復 event-loop health failure 時，沒有直接繞過 serving path。[^validation]
{ .pb-translation lang=zh-Hant }

## 4. Why only 3/12 requests completed｜為什麼只有 3／12 requests 完成

Moving work to threads protected the event loop, but it did not serialize tokenizer initialization. Six requests arrived before the first roughly 80-second `from_pretrained` failure completed. Each request started another tokenizer load. The router stayed healthy while those client deadlines expired.

把工作移到 threads 之後，event loop 不再被拖住，但 tokenizer initialization 仍沒有被序列化。第一個約 80 秒的 `from_pretrained` failure 結束前，又有 6 個 requests 抵達；每個 request 都另外啟動一次 tokenizer load。Router 一直保持 healthy，client deadline 卻已經到期。
{ .pb-translation lang=zh-Hant }

The next three requests reached `/tokenize` fallback quickly because the failed-load flag was now set. They still timed out during the backend's first-request LMCache post-initialization. Three later clients completed after the backend became warm.

接下來 3 個 requests 因為 failed-load flag 已經設定，很快就進入 `/tokenize` fallback，但又遇上 backend 第一次使用 LMCache 的 post-initialization，最後仍然 timeout。Backend warm 起來後，最後 3 個 clients 才順利完成。
{ .pb-translation lang=zh-Hant }

This exposed two separate readiness gaps:

這次測試拆出了兩個不同的 readiness gap：
{ .pb-translation lang=zh-Hant }

- **Router initialization:** concurrent requests could launch duplicate tokenizer loads before the first failure was cached.<br><span class="pb-inline-translation" lang="zh-Hant">**Router initialization：** 第一個失敗被記錄前，多個 concurrent requests 可能重複啟動 tokenizer load。</span>
- **Backend initialization:** the backend was Kubernetes-Ready before its first LMCache request path had finished warming.<br><span class="pb-inline-translation" lang="zh-Hant">**Backend initialization：** 第一次 LMCache request path 完成 warm-up 前，backend 就已經被 Kubernetes 判定為 Ready。</span>

The first gap belongs to single-flight initialization or another form of load deduplication. The second belongs to deployment readiness: pre-warm the path, extend the initial request budget, or expose a readiness condition that includes the required initialization.

第一個 gap 需要 single-flight initialization 或其他 load deduplication。第二個則是 deployment readiness 問題：可以預先 warm-up、放寬首次 request budget，或讓 readiness condition 納入必要的初始化狀態。
{ .pb-translation lang=zh-Hant }

## 5. A stronger acceptance gate｜更完整的驗收條件

For this serving path, `/health == 200` is necessary but incomplete. I would keep four independent gates:

對這條 serving path 而言，`/health == 200` 是必要條件，但還不完整。驗收時應分開保留四道 gates：
{ .pb-translation lang=zh-Hant }

| Gate／驗收層 | What it proves／能證明什麼 | Example criterion／範例條件 |
| --- | --- | --- |
| Event-loop health | Router can answer probes under load／Router 在負載下仍能回 probe | 177/177 HTTP 200, no sample over 2 s |
| Request completion | Clients receive usable responses／Client 能拿到可用 response | 12/12 complete within the declared timeout |
| Dependency path | Fallback and KV components actually run／Fallback 與 KV 元件真的有執行 | `/tokenize`, controller lookup, generation, LMCache activity observed |
| Day-2 recovery | Cold Pod or dependency failure recovers safely／Cold Pod 或相依元件故障後能安全恢復 | no restart loop, bounded warm-up, repeatable second request |

These gates stop a partial repair from being reported as full service recovery. They also show who owns the next change: router concurrency, backend initialization, Kubernetes readiness, or the client timeout contract.

這四道 gates 可以避免把局部修復誤報成完整服務恢復，也能看出下一步該由誰處理：router concurrency、backend initialization、Kubernetes readiness，還是 client timeout contract。
{ .pb-translation lang=zh-Hant }

## 6. Boundary of the result｜結果適用範圍

This was a single-node, single-GPU, restricted-egress experiment with one small model and a deliberately long prompt. It does not establish production throughput, an SLA, or behavior on a multi-node deployment. PR #1025 remains open and review-gated as of 2026-07-29.

這是 single-node、single-GPU、restricted-egress 的實驗，只使用一個小模型與刻意拉長的 prompt。結果不能直接代表 production throughput、SLA 或 multi-node deployment；截至 2026-07-29，PR #1025 仍是 open 且等待 review。
{ .pb-translation lang=zh-Hant }

The next meaningful rerun requires a new PR head that prevents duplicate concurrent tokenizer loads. The completion gate is explicit: 12/12 clients, 177/177 healthy samples, no sample over two seconds, no probe event, and no router restart. Until the code changes, repeating the same trial would only reproduce the same boundary.

下一次有意義的重跑，必須等新 PR head 解決 duplicate concurrent tokenizer loads。Completion gate 已經寫清楚：12／12 clients 完成、177／177 health samples 成功、沒有任何 sample 超過 2 秒、沒有 probe event，也沒有 router restart。程式碼沒變之前，重跑只會再次得到相同邊界。
{ .pb-translation lang=zh-Hant }

[^repro]: [Bounded reproduction on Production Stack main](https://github.com/vllm-project/production-stack/issues/1016#issuecomment-5101605222), including the exact workload, control arm, restricted-egress failure, probe events, and restart counts.／Production Stack main 上的 bounded reproduction，包含 workload、control arm、restricted-egress failure、probe events 與 restart counts。
[^pr]: [Production Stack PR #1025](https://github.com/vllm-project/production-stack/pull/1025), including the event-loop fix, failed-load cache, focused tests, and stated non-goals.／包含 event-loop fix、failed-load cache、focused tests 與明確 non-goals。
[^validation]: [Validation of PR #1025 head `e104ed573`](https://github.com/vllm-project/production-stack/pull/1025#issuecomment-5113671870), with health, request-completion, and dependency-path results.／PR #1025 head `e104ed573` 的 health、request completion 與 dependency path 結果。
[^probes]: [Kubernetes documentation: Configure Liveness, Readiness and Startup Probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/).／Kubernetes 官方 probe 行為說明。

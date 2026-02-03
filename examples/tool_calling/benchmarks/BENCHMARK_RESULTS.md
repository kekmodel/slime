# Tool Calling Benchmark Results

> LLM reasoning/thinking 토큰 효율성 및 tool calling 능력 종합 벤치마크

**측정일**: 2026-02
**테스트 환경**: OpenRouter API
**Sampling**: `temperature=1.0, top_p=0.95`
**측정 단위**: 토큰 (각 모델의 HuggingFace tokenizer)

---

## TL;DR

### 🏆 최종 추천

| 용도 | 모델 | Latency | 특징 |
|------|------|---------|------|
| **최속 + 병렬** | glm4.7-flash:think-off | ~0.4s | R=0, MTP, 4턴, Acc 100% 🏆 |
| **토큰 최소** | gpt-oss-20b:low | ~0.4s | 93토큰, 5턴, Acc 100% |
| **Reasoning + 안정** | glm4.7-flash:think-all | ~2.4s | CV 25%, MTP, 병렬 |
| **Reasoning + 최고안정** | gpt-oss-120b:high | ~4.6s | CV 19%, Acc 100% |

### ❌ 비추천

| 모델 | 문제점 |
|------|--------|
| nemotron3-nano | CV 72~249%, Acc 40~90% |
| qwen3-30b | Acc 60% (tool 순서 오류) |
| qwen3-next-80b | Acc 80%, Latency 8.4s |
| gpt-oss-20b:high | CV 140% (reasoning 폭발) |

---

## 목차

1. [TL;DR](#tldr)
2. [모델 아키텍처](#모델-아키텍처)
3. [벤치마크 결과](#벤치마크-결과)
4. [컬럼 설명](#컬럼-설명)
5. [Latency 분석](#latency-분석)
6. [API 설정 가이드](#모델별-api-설정)

---

## 모델 아키텍처

| 모델 | 아키텍처 | 총 파라미터 | **Active** | Experts | Top-K | MTP/Eagle |
|------|----------|------------|------------|---------|-------|-----------|
| glm4.7-flash | MoE | 30B | **3B** | 64 | 4 | ✅ |
| gpt-oss-20b | MoE | 21B | **3.6B** | 32 | 4 | ❌ |
| nemotron3-nano | **Hybrid Mamba-MoE** | 30B | **3.5B** | 128+1 | 5 | ❌ |
| qwen3-30b | MoE | 30.5B | **3.3B** | 128 | 8 | ❌ |
| qwen3-next-80b | MoE | 80B | **3.9B** | 512 | 10 | ✅ |
| gpt-oss-120b | MoE | 117B | **5.1B** | 128 | 4 | ❌ |

> **nemotron3-nano**: Mamba-2 (23 layers) + GQA Attention (6 layers) + MoE 하이브리드

**핵심**:
- 총 파라미터가 아닌 **Active 파라미터 (3~5B)** 가 Latency 결정
- **MTP/Eagle 지원**: glm4.7-flash, qwen3-next-80b → Speculative Decoding으로 추가 속도 향상

---

## 벤치마크 결과

### 싱글턴 Tool Calling Benchmark (n=20)

> 첫 응답만 측정. 프롬프트: "How much would it cost to buy an apple and a banana?"

| Model | R.Avg | R.Range | R.CV | C.Avg |
|-------|-------|---------|------|-------|
| glm4.7-flash:think-off | **0** | 0~0 | 0% ✅ | 22 |
| glm4.7-flash:think-turn | 80 | 47~137 | 33% ✅ | 22 |
| glm4.7-flash:think-all | 65 | 31~107 | 30% ✅ | 20 |
| gpt-oss-120b:low | 12 | 6~14 | 18% ✅ | 0 |
| gpt-oss-120b:medium | 59 | 21~97 | 33% ✅ | 0 |
| gpt-oss-120b:high | 243 | 119~569 | 49% ⚠️ | 0 |
| gpt-oss-20b:low | 13 | 7~22 | 27% ✅ | 0 |
| gpt-oss-20b:medium | 60 | 11~372 | **125%** ❌ | 0 |
| gpt-oss-20b:high | 1610 | 183~10438 | **140%** ❌ | 0 |
| nemotron3-nano:think-off | **0** | 0~0 | 0% ✅ | 47 |
| nemotron3-nano | **4920** | 180~**54992** | **249%** ❌ | 9 |
| qwen3-30b | 551 | 103~2285 | **89%** ❌ | 0 |
| qwen3-next-80b | 1059 | 584~1804 | 33% ✅ | 2 |

### 멀티턴 Tool Calling Benchmark (n=10)

> 인사 → 가격 조회 도구(2개) → 계산 도구 → 최종 응답 시나리오 (병렬 tool calling 지원시 4턴, 미지원시 5턴)

| Model | R.Avg | R.Range | R.CV | C.Avg | Turns | Acc | Parallel |
|-------|-------|---------|------|-------|-------|-----|----------|
| glm4.7-flash:think-off | **0** | 0~0 | 0% ✅ | 78 | 4.0 | 100% ✅ | ✅ |
| glm4.7-flash:think-turn | 489 | 281~579 | 20% ✅ | 88 | 4.0 | 100% ✅ | ✅ |
| glm4.7-flash:think-all | 463 | 308~615 | 21% ✅ | 80 | 4.0 | 100% ✅ | ✅ |
| gpt-oss-120b:low | 47 | 34~60 | 21% ✅ | 30 | 5.0 | 100% ✅ | ❌ |
| gpt-oss-120b:medium | 187 | 137~229 | 13% ✅ | 44 | 5.0 | 100% ✅ | ❌ |
| gpt-oss-120b:high | 647 | 471~869 | 22% ✅ | 56 | 5.0 | 100% ✅ | ❌ |
| gpt-oss-20b:low | 41 | 34~52 | 14% ✅ | 25 | 5.0 | 100% ✅ | ❌ |
| gpt-oss-20b:medium | 164 | 77~280 | 37% ✅ | 31 | 4.7 | 90% ⚠️ | ❌ |
| gpt-oss-20b:high | 1398 | 379~2414 | 45% ⚠️ | 43 | 5.0 | 100% ✅ | ❌ |
| nemotron3-nano:think-off | **0** | 0~0 | 0% ✅ | 61 | 4.1 | 80% ⚠️ | ✅ |
| nemotron3-nano | 1396 | 399~3202 | 59% ⚠️ | 61 | 4.6 | 90% ⚠️ | ✅ |
| qwen3-30b | 628 | 522~899 | 18% ✅ | 51 | 3.9 | 80% ⚠️ | ✅ |
| qwen3-next-80b | 2545 | 1498~3559 | 23% ✅ | 63 | 3.9 | 80% ⚠️ | ✅ |

### 멀티턴 Tool Calling Benchmark - 한국어 (n=10)

> 한국어 프롬프트로 동일 시나리오 테스트. "안녕하세요!" → "apple과 banana의 가격을 조회한 후, 계산기 도구를 사용해서 평균 가격을 계산해주세요."

| Model | R.Avg | R.Range | R.CV | C.Avg | Acc | Parallel |
|-------|-------|---------|------|-------|-----|----------|
| glm4.7-flash:think-off | **0** | 0~0 | 0% ✅ | 172 | 100% ✅ | ✅ |
| glm4.7-flash:think-turn | 792 | 410~1564 | 51% ⚠️ | 181 | 100% ✅ | ✅ |
| glm4.7-flash:think-all | 943 | 526~1270 | 25% ✅ | 192 | 100% ✅ | ✅ |
| gpt-oss-120b:low | 52 | 43~71 | 16% ✅ | 37 | 100% ✅ | ❌ |
| gpt-oss-120b:medium | 154 | 87~189 | 23% ✅ | 56 | 100% ✅ | ❌ |
| gpt-oss-120b:high | 737 | 548~1000 | 19% ✅ | 97 | 100% ✅ | ❌ |
| gpt-oss-20b:low | 46 | 38~66 | 18% ✅ | 47 | 100% ✅ | ❌ |
| gpt-oss-20b:medium | 183 | 129~368 | 37% ✅ | 62 | 100% ✅ | ❌ |
| gpt-oss-20b:high | 1881 | 774~3126 | 38% ✅ | 56 | 100% ✅ | ❌ |
| qwen3-30b | 692 | 280~840 | 26% ✅ | 110 | 60% ❌ | ✅ |
| qwen3-next-80b | 2928 | 1387~3702 | 27% ✅ | 66 | 80% ⚠️ | ✅ |
| nemotron3-nano:think-off | **0** | 0~0 | 0% ✅ | 91 | 40% ❌ | ✅ |
| nemotron3-nano | 2391 | 774~6344 | 72% ❌ | 83 | 90% ⚠️ | ✅ |

### 영어 vs 한국어 비교

| Model | EN R.Avg | KO R.Avg | EN Acc | KO Acc | 비고 |
|-------|----------|----------|--------|--------|------|
| glm4.7-flash:think-all | 463 | 943 | 100% | 100% | KO에서 reasoning 2배 증가 |
| glm4.7-flash:think-turn | 489 | 792 | 100% | 100% | KO에서 CV 20%→51% 불안정 |
| gpt-oss-120b:high | 647 | 737 | 100% | 100% | 비슷 |
| gpt-oss-20b:high | 1398 | 1881 | 100% | 100% | KO에서 35% 증가 |
| qwen3-30b | 628 | 692 | 80% | 60% | **KO에서 Acc 하락** |
| nemotron3-nano:think-off | 0 | 0 | 80% | 40% | **KO에서 Acc 절반** |

**Key Findings**:
- GLM/gpt-oss는 한국어에서도 Acc 100% 유지
- Qwen/Nemotron은 한국어 프롬프트에서 Acc 하락 (tool 순서 오류 증가)
- 대부분 모델에서 한국어 reasoning 토큰이 영어보다 증가

---

## 컬럼 설명

| 컬럼 | 설명 |
|------|------|
| **R.Avg** | Reasoning 평균 토큰 수 |
| **R.Range** | Reasoning min~max 토큰 |
| **R.CV** | Reasoning 변동계수 (낮을수록 안정적) |
| **C.Avg** | Content 평균 토큰 수 |
| **Turns** | 평균 대화 턴 수 |
| **Acc** | Tool calling 순서 정확도 (price_req → calc_req) |
| **Parallel** | 병렬 tool call 지원 여부 |

### CV (변동계수) 판정 기준

| CV | 판정 | 의미 |
|----|------|------|
| < 40% | ✅ | 안정적 |
| 40~60% | ⚠️ | 주의 필요 |
| > 60% | ❌ | 불안정 (프로덕션 부적합) |

### Parallel Tool Calling 판정 기준

> 정확한 샘플(Acc 100%) 중 **4턴인 샘플이 하나라도 있으면** 병렬 지원

| Turn 수 | 의미 |
|---------|------|
| **4턴** | greeting → price_req(2개 병렬) → calc_req → final |
| **5턴** | greeting → price_req → price_req → calc_req → final (순차) |
| **< 4턴** | 일부 단계 생략 (비정상) |

---

## Latency 분석

> RTT 무시, 순수 Decode 시간만 계산
> MTP Accept Rate ~55% 가정 (÷1.55 속도 향상)

### TPS 가정

| 모델 | Active | TPS | 비고 |
|------|--------|-----|------|
| glm4.7-flash | 3B | 300 | MoE + MTP |
| qwen3-30b | 3.3B | 280 | MoE |
| nemotron3-nano | 3.5B | 260 | Hybrid Mamba-MoE |
| gpt-oss-20b | 3.6B | 250 | MoE |
| qwen3-next-80b | 3.9B | 230 | MoE + MTP |
| gpt-oss-120b | 5.1B | 180 | MoE |

### 전체 모델 Latency 순위 (한국어)

| 순위 | 모델 | Decode 토큰 | MTP | 예상 Latency | Acc | 추천 |
|------|------|-------------|-----|--------------|-----|------|
| 1 | nemotron3-nano:think-off | 91 | ❌ | ~0.35s | **40%** ❌ | ❌ |
| 2 | gpt-oss-20b:low | 93 | ❌ | **~0.37s** | 100% ✅ | ✅ |
| 2 | **glm4.7-flash:think-off** | 172 | ✅ | **~0.37s** | 100% ✅ | 🏆 |
| 4 | gpt-oss-120b:low | 89 | ❌ | ~0.49s | 100% ✅ | ✅ |
| 5 | gpt-oss-20b:medium | 245 | ❌ | ~0.98s | 100% ✅ | ✅ |
| 6 | gpt-oss-120b:medium | 210 | ❌ | ~1.17s | 100% ✅ | ✅ |
| 7 | glm4.7-flash:think-turn | 973 | ✅ | ~2.09s | 100% ✅ | ⚠️ |
| 8 | **glm4.7-flash:think-all** | 1135 | ✅ | **~2.44s** | 100% ✅ | ✅ |
| 9 | qwen3-30b | 802 | ❌ | ~2.86s | **60%** ❌ | ❌ |
| 10 | gpt-oss-120b:high | 834 | ❌ | ~4.63s | 100% ✅ | ✅ |
| 11 | gpt-oss-20b:high | 1937 | ❌ | ~7.75s | 100% ✅ | ✅ |
| 12 | qwen3-next-80b | 2994 | ✅ | ~8.40s | **80%** ⚠️ | ❌ |
| 13 | nemotron3-nano | 2474 | ❌ | ~9.52s | 90% ⚠️ | ❌ |

### think-turn 비추천 이유

| 항목 | think-turn | think-all |
|------|------------|-----------|
| CV (안정성) | **51% ⚠️** | 25% ✅ |
| KV 캐시 | ❌ 삭제됨 | ✅ 유지 |
| Latency | ~2.1s | ~2.4s |

- think-turn: 매 턴 reasoning 생성 후 삭제 → **KV 캐시 미스**
- Latency 이점 적고 CV 불안정 → **사용 이유 없음**

### Key Insights

#### 1. 멀티턴이 싱글턴보다 안정적

| Model | Single R.CV | Multi R.CV |
|-------|-------------|------------|
| nemotron3-nano | 249% ❌ | 59% ⚠️ |
| gpt-oss-20b:high | 140% ❌ | 45% ⚠️ |
| qwen3-30b | 89% ❌ | 18% ✅ |

**원인**: 대화 컨텍스트가 모델의 reasoning을 "가이드"
- 싱글턴: 매번 처음부터 상황 파악 → 불확실성 높음 → 폭발 가능
- 멀티턴: 이전 턴들이 방향 설정 → reasoning이 bounded됨

#### 2. Reasoning 폭발 현상

nemotron3-nano 싱글턴에서 **54,992 토큰** 폭발 사례 발생
- 모델이 "길을 잃고" 끝없이 reasoning
- 비용 예측 불가 → 프로덕션 위험

#### 3. 모델 크기와 안정성

| Model | Size | R.CV (Single) | R.CV (Multi) |
|-------|------|---------------|--------------|
| gpt-oss-120b:high | 120B | 49% ⚠️ | 22% ✅ |
| gpt-oss-20b:high | 20B | 140% ❌ | 45% ⚠️ |

더 큰 모델이 더 안정적인 reasoning 출력

### Parallel Tool Calling 지원

| model_id | Parallel | Turn 분포 | 비고 |
|----------|----------|-----------|------|
| **glm4.7-flash** | ✅ Yes | 4턴: 30회 (100%) | 완벽한 병렬 |
| **gpt-oss-120b** | ❌ No | 5턴: 30회 (100%) | 완벽한 순차 |
| **gpt-oss-20b** | ❌ No | 5턴: 29회, 2턴: 1회 | 순차 |
| **nemotron3-nano** | ✅ Yes | 4턴 있음 | 병렬 but Acc 80~90%, CV 높음 |
| **qwen3-30b** | ✅ Yes | 4턴: 7회, 3턴: 2회, 5턴: 1회 | 병렬 but Acc 80% |
| **qwen3-next-80b** | ✅ Yes | 4턴: 5회, 5턴: 3회, 2턴: 2회 | 병렬 but Acc 80% |

> 동일 model_id는 동일 특성 (예: gpt-oss-120b:low/medium/high 모두 순차)

---

## 모델별 API 설정

| Model | API Parameter | 설명 |
|-------|---------------|------|
| gpt-oss:* | `reasoning: {effort: "low\|medium\|high"}` | reasoning 깊이 조절 |
| glm4.7-flash:think-all | `thinking: {type: "enabled", clear_thinking: false}` | 전체 thinking 유지 |
| glm4.7-flash:think-turn | `thinking: {type: "enabled", clear_thinking: true}` | 턴별 thinking (후 제거) |
| glm4.7-flash:think-off | `reasoning: {effort: "none"}` | thinking 끔 |
| qwen3-* | - | 기본 설정만 지원 |

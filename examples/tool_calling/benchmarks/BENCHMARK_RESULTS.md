# Reasoning Length Benchmark Results

> 모델별 reasoning/thinking 토큰 효율성 및 tool calling 능력 측정 결과

**측정일**: 2026-02
**테스트 환경**: OpenRouter API
**Sampling Parameters**: `temperature=1.0, top_p=0.95`
**측정 단위**: 토큰 (각 모델의 HuggingFace tokenizer 사용)

---

## 싱글턴 Tool Calling Benchmark (n=20)

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

---

## 멀티턴 Tool Calling Benchmark (n=10)

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

---

## 멀티턴 Tool Calling Benchmark - 한국어 (n=10)

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

## 모델별 Parallel Tool Calling 지원

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

## Key Insights

### 1. 멀티턴이 싱글턴보다 안정적

| Model | Single R.CV | Multi R.CV |
|-------|-------------|------------|
| nemotron3-nano | 249% ❌ | 59% ⚠️ |
| gpt-oss-20b:high | 140% ❌ | 45% ⚠️ |
| qwen3-30b | 89% ❌ | 18% ✅ |

**원인**: 대화 컨텍스트가 모델의 reasoning을 "가이드"
- 싱글턴: 매번 처음부터 상황 파악 → 불확실성 높음 → 폭발 가능
- 멀티턴: 이전 턴들이 방향 설정 → reasoning이 bounded됨

### 2. Reasoning 폭발 현상

nemotron3-nano 싱글턴에서 **54,992 토큰** 폭발 사례 발생
- 모델이 "길을 잃고" 끝없이 reasoning
- 비용 예측 불가 → 프로덕션 위험

### 3. 모델 크기와 안정성

| Model | Size | R.CV (Single) | R.CV (Multi) |
|-------|------|---------------|--------------|
| gpt-oss-120b:high | 120B | 49% ⚠️ | 22% ✅ |
| gpt-oss-20b:high | 20B | 140% ❌ | 45% ⚠️ |

더 큰 모델이 더 안정적인 reasoning 출력

### 4. Parallel Tool Calling의 실질적 의미

병렬 지원 모델이라도 Accuracy가 낮으면 의미 없음:
- glm4.7-flash: Parallel ✅ + Acc 100% → **실제 사용 가능**
- qwen3-30b: Parallel ✅ + Acc 80% → **20% 확률로 순서 틀림**

---

## 비용 효율 분석 (한국어 기준)

> Acc 100% 모델만 비교. 비용 = 토큰 비용 + API 호출 비용

### 총 토큰 (R.Avg + C.Avg) 순위

| 순위 | 모델 | R+C 토큰 | 턴 | 병렬 | 평가 |
|------|------|----------|-----|------|------|
| 1 | gpt-oss-120b:low | **89** | 5 | ❌ | 토큰 최소 |
| 2 | gpt-oss-20b:low | **93** | 5 | ❌ | 저렴 + 토큰 효율 |
| 3 | glm4.7-flash:think-off | 172 | **4** | ✅ | 병렬로 턴 절약 |
| 4 | gpt-oss-120b:medium | 210 | 5 | ❌ | |
| 5 | gpt-oss-20b:medium | 245 | 5 | ❌ | |
| 6 | gpt-oss-120b:high | 834 | 5 | ❌ | |
| 7 | glm4.7-flash:think-turn | 973 | **4** | ✅ | CV 불안정 |
| 8 | glm4.7-flash:think-all | 1135 | **4** | ✅ | reasoning 필요시 |
| 9 | gpt-oss-20b:high | 1937 | 5 | ❌ | |

### 시나리오별 최적 선택

| 시나리오 | 최적 모델 | 이유 |
|----------|-----------|------|
| **토큰 비용 최소** | gpt-oss-120b:low | 89 토큰 (최소) |
| **Latency 최소** | glm4.7-flash:think-off | 병렬 4턴 + R=0 |
| **API 호출 최소** | glm4.7-flash:think-off | 4턴 (병렬) |
| **저렴한 모델 + 안정** | gpt-oss-20b:low | 20B + 93토큰 |
| **Reasoning 필요** | gpt-oss-120b:high | 737 R토큰, CV 19% |

### 비용 시뮬레이션 (1000회 호출)

> 가정: 입력 100토큰, 출력 단가 $0.01/1K tokens, API 호출당 $0.001

| 모델 | 출력 토큰 | 턴 | 토큰 비용 | API 비용 | **총 비용** |
|------|----------|-----|----------|----------|-------------|
| glm4.7-flash:think-off | 172K | 4K | $1.72 | $4.00 | **$5.72** 🏆 |
| gpt-oss-120b:low | 89K | 5K | $0.89 | $5.00 | **$5.89** |
| gpt-oss-20b:low | 93K | 5K | $0.93 | $5.00 | **$5.93** |
| gpt-oss-120b:medium | 210K | 5K | $2.10 | $5.00 | **$7.10** |
| glm4.7-flash:think-all | 1135K | 4K | $11.35 | $4.00 | **$15.35** |

**핵심**: GLM think-off는 토큰은 더 많지만 **턴 1회 절약**으로 총 비용 최저

---

## 추천 설정

### Best Choices

| 용도 | 모델 | 특징 |
|------|------|------|
| **제로 reasoning + 병렬** | glm4.7-flash:think-off | 0토큰, 병렬, Acc 100% 🏆 |
| **최소 reasoning + 안정** | gpt-oss-120b:low | 47토큰, CV 21%, Acc 100% |
| **reasoning + 병렬** | glm4.7-flash:think-all | 463토큰, 병렬, Acc 100% |

### 비추천 모델

| 모델 | 문제점 |
|------|--------|
| **gpt-oss-20b:high** | R.Range 183~10438 (CV 140%) - 폭발 위험 |
| **nemotron3-nano** | 싱글턴 CV 249%, 멀티턴 Acc 80~90% |
| **qwen3-30b/next-80b** | Acc 80% - tool 순서 틀림 |

---

## nemotron3-nano 심층 분석

### 문제점

1. **think-off 모드**: tool 사용 안 함 (~40% 확률로 직접 텍스트 응답)
2. **default 모드**: reasoning 폭발 (싱글턴 CV 249%, 최대 54,992 토큰)

### 싱글 vs 멀티 비교

| Mode | R.Avg | R.Range | R.CV |
|------|-------|---------|------|
| Single-turn (EN) | 4,920 | 180~54,992 | 249% ❌ |
| Multi-turn (EN) | 1,396 | 399~3,202 | 59% ⚠️ |
| Multi-turn (KO) | 2,391 | 774~6,344 | 72% ❌ |

멀티턴에서 range가 1/17로 줄어듦 (54,992 → 3,202)

### 결론

**Tool calling에 부적합**. 어떤 모드든 프로덕션 사용 권장하지 않음.

---

## 모델 아키텍처 (MoE 스펙)

> 모든 테스트 모델은 MoE (Mixture of Experts) 아키텍처. Latency는 Active 파라미터에 비례.

| 모델 | 총 파라미터 | **Active** | Experts | Top-K | MTP/Eagle |
|------|------------|------------|---------|-------|-----------|
| glm4.7-flash | 30B | **3B** | 64 | 4 | ✅ |
| gpt-oss-20b | 21B | **3.6B** | 32 | 4 | ❌ |
| gpt-oss-120b | 117B | **5.1B** | 128 | 4 | ❌ |
| qwen3-30b | 30.5B | **3.3B** | 128 | 8 | ❌ |
| qwen3-next-80b | 80B | **3.9B** | 512 | 10 | ✅ |

**핵심**:
- 총 파라미터가 아닌 **Active 파라미터 (3~5B)** 가 Latency 결정
- **MTP/Eagle 지원**: glm4.7-flash, qwen3-next-80b → Speculative Decoding으로 추가 속도 향상 가능

---

## Latency 예상 순위 (한국어, Acc 100%)

> Active 파라미터가 비슷하므로 **Decode 토큰 수**가 Latency 결정
> MTP/Eagle 지원 모델은 Speculative Decoding으로 **2~3배 추가 속도 향상** 가능

| 순위 | 모델 | Active | 턴 | Decode 토큰 | MTP | 예상 Latency |
|------|------|--------|-----|-------------|-----|--------------|
| 1 | **glm4.7-flash:think-off** | 3B | 4 | 172 | ✅ | **~0.5s** 🏆 |
| 2 | gpt-oss-120b:low | 5.1B | 5 | 89 | ❌ | ~0.7s |
| 3 | gpt-oss-20b:low | 3.6B | 5 | 93 | ❌ | ~0.7s |
| 4 | gpt-oss-20b:medium | 3.6B | 5 | 245 | ❌ | ~1.2s |
| 5 | gpt-oss-120b:medium | 5.1B | 5 | 210 | ❌ | ~1.3s |
| 6 | **glm4.7-flash:think-turn** | 3B | 4 | 973 | ✅ | ~2.3s |
| 7 | **glm4.7-flash:think-all** | 3B | 4 | 1135 | ✅ | ~2.7s |
| 8 | gpt-oss-120b:high | 5.1B | 5 | 834 | ❌ | ~4.7s |
| 9 | gpt-oss-20b:high | 3.6B | 5 | 1937 | ❌ | ~7.4s |

> MTP 지원 모델: 예상 Latency = 기본 Latency ÷ 1.55 (Speculative Decoding, Accept Rate ~55%)

### Latency 기준 추천

| 용도 | 모델 | Latency | 특징 |
|------|------|---------|------|
| **최속 (R=0)** | glm4.7-flash:think-off | **~0.5s** | MTP + 병렬 + R=0 🏆 |
| **최속 (짧은 R)** | gpt-oss-120b:low | ~0.7s | 토큰 최소 (89), MTP 없음 |
| **R 필요 + 최속** | glm4.7-flash:think-all | **~2.7s** | MTP + 병렬 + CV 25% |
| **R 필요 + 안정** | gpt-oss-120b:high | ~4.7s | CV 19% 최고 안정, MTP 없음 |

### think-turn 비추천 이유

| 항목 | think-turn | think-all |
|------|------------|-----------|
| CV (안정성) | **51% ⚠️** | 25% ✅ |
| KV 캐시 | ❌ 삭제됨 | ✅ 유지 |
| Latency | ~3.6s | ~4.2s |

- think-turn: 매 턴 reasoning 생성 후 삭제 → **KV 캐시 미스**
- Latency 이점 적고 CV 불안정 → **사용 이유 없음**

---

## 모델별 API 설정

| Model | API Parameter | 설명 |
|-------|---------------|------|
| gpt-oss:* | `reasoning: {effort: "low\|medium\|high"}` | reasoning 깊이 조절 |
| glm4.7-flash:think-all | `thinking: {type: "enabled", clear_thinking: false}` | 전체 thinking 유지 |
| glm4.7-flash:think-turn | `thinking: {type: "enabled", clear_thinking: true}` | 턴별 thinking (후 제거) |
| glm4.7-flash:think-off | `reasoning: {effort: "none"}` | thinking 끔 |
| qwen3-* | - | 기본 설정만 지원 |

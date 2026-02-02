# Reasoning Length Benchmarks

모델별 reasoning/thinking **토큰** 효율성 측정 벤치마크.

**테스트 환경**: OpenRouter API
**Sampling Parameters**: `temperature=1.0, top_p=0.95`
**측정 단위**: 토큰 (각 모델의 HuggingFace tokenizer 사용)

---

## 측정 기준

### 토큰 측정
- **이전**: `len(text)` - 글자 수 (부정확)
- **현재**: `tokenizer.encode(text)` - 실제 토큰 수 (비용과 직결)

각 모델의 HuggingFace tokenizer를 사용하여 정확한 토큰 수 측정:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
token_count = len(tokenizer.encode(text, add_special_tokens=False))
```

### CV (변동계수) 판정
| CV | 판정 | 의미 |
|----|------|------|
| < 40% | ✅ | 안정적 |
| 40~60% | ⚠️ | 주의 필요 |
| > 60% | ❌ | 불안정 (프로덕션 부적합) |

### Accuracy (정확도) 판정
- **100%**: 모든 샘플이 `price_req → calc_req` 순서 준수
- **< 100%**: 일부 샘플이 순서 틀림 (가격 조회 전 계산 시도 등)

### Parallel Tool Call 판정
> **기준**: 정확한 샘플(Acc 100%) 중 **4턴인 샘플이 하나라도 있으면** 병렬 지원

| Turn 수 | 의미 |
|---------|------|
| **4턴** | greeting → price_req(2개 병렬) → calc_req → final |
| **5턴** | greeting → price_req → price_req → calc_req → final (순차) |
| **< 4턴** | 일부 단계 생략 (비정상) |

**동일 model_id는 동일 특성** (예: gpt-oss-120b:low/medium/high 모두 순차)

---

## 사용법

```bash
# 기본 실행 (모든 기본 모델, 싱글턴 20회 + 멀티턴 10회)
python examples/tool_calling/benchmarks/reasoning_length.py

# 모델 목록 보기
python examples/tool_calling/benchmarks/reasoning_length.py --list

# 특정 모델만 테스트 (부분 매칭)
python examples/tool_calling/benchmarks/reasoning_length.py gpt-oss-120b:low
python examples/tool_calling/benchmarks/reasoning_length.py gpt-oss glm4.7-flash  # 여러 모델

# 테스트 모드 선택
python examples/tool_calling/benchmarks/reasoning_length.py --single   # 싱글턴만 (빠름)
python examples/tool_calling/benchmarks/reasoning_length.py --multi    # 멀티턴만

# 반복 횟수 조정
python examples/tool_calling/benchmarks/reasoning_length.py -n 30      # 30회 반복
```

---

## 벤치마크 결과 (2026-02, 토큰 기준)

### 싱글턴 Tool Calling Benchmark (temperature=1.0, top_p=0.95, n=20)

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

### 멀티턴 Tool Calling Benchmark (temperature=1.0, top_p=0.95, n=10)

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

### 모델별 Parallel 지원 (Turn 분포 기준)

| model_id | Parallel | Turn 분포 | 비고 |
|----------|----------|-----------|------|
| **glm4.7-flash** | ✅ Yes | 4턴: 30회 (100%) | 완벽한 병렬 |
| **gpt-oss-120b** | ❌ No | 5턴: 30회 (100%) | 완벽한 순차 |
| **gpt-oss-20b** | ❌ No | 5턴: 29회, 2턴: 1회 | 순차 |
| **nemotron3-nano** | ✅ Yes | 4턴 있음 | 병렬 but Acc 80~90%, CV 높음 |
| **qwen3-30b** | ✅ Yes | 4턴: 7회, 3턴: 2회, 5턴: 1회 | 병렬 but Acc 80% |
| **qwen3-next-80b** | ✅ Yes | 4턴: 5회, 5턴: 3회, 2턴: 2회 | 병렬 but Acc 80% |

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

### 표 정렬 규칙
- **모델명**: 알파벳순 (glm → gpt-oss → nemotron → qwen)
- **태그 순서**:
  - thinking: `think-off` → `think-turn` → `think-all`
  - reasoning effort: `low` → `medium` → `high`

---

## 모델별 API 설정

| Model | API Parameter | 설명 |
|-------|---------------|------|
| gpt-oss:* | `reasoning: {effort: "low\|medium\|high"}` | reasoning 깊이 조절 |
| glm4.7-flash:think-all | `thinking: {type: "enabled", clear_thinking: false}` | 전체 thinking 유지 |
| glm4.7-flash:think-turn | `thinking: {type: "enabled", clear_thinking: true}` | 턴별 thinking (후 제거) |
| glm4.7-flash:think-off | `reasoning: {effort: "none"}` | thinking 끔 |
| qwen3-* | - | 기본 설정만 지원 |

---

## 추천 설정

| 용도 | 모델 | 특징 |
|------|------|------|
| **제로 reasoning + 병렬** | glm4.7-flash:think-off | 0토큰, 병렬, Acc 100% 🏆 |
| **최소 reasoning + 안정** | gpt-oss-120b:low | 47토큰, CV 21%, Acc 100% |
| **reasoning + 병렬** | glm4.7-flash:think-all | 463토큰, 병렬, Acc 100% |

### 비추천 모델

| 모델 | 문제점 |
|------|--------|
| **gpt-oss-20b:high** | R.Range 183~10438 (CV 140%) - 폭발 위험 |
| **qwen3-30b** | Acc 80% - tool 순서 틀림 |
| **nemotron3-nano** | think-off: tool 미사용 40%, default: CV 100%+ |

---

## 참고: nemotron3-nano 분석

> Optional 모델로, 기본 실행에서 제외됨. 테스트하려면 이름 지정 필요: `reasoning_length.py nemotron3-nano`

### 문제점
1. **think-off 모드**: tool 사용 안 함 (~40% 확률로 직접 텍스트 응답)
2. **default 모드**: reasoning 폭발 (CV 94~134%)

### 결론
**Tool calling에 부적합**. 어떤 모드든 프로덕션 사용 권장하지 않음.

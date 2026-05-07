# UWB + BIOPAC 데이터 수집 프로토콜

이 문서는 팀원 3명이 실제 UWB raw와 BIOPAC respiration 데이터를 수집할 때 따라갈 실험 절차입니다.

## 목표

```text
UWB raw + BIOPAC raw
-> SyncData-like 중간 산출물
-> CNN/SNN 비교
```

1차 목표는 UWB로 BIOPAC respiration waveform을 재구성하는 것입니다. 심박은 1차 목표에서 제외하고, 호흡 신호부터 안정적으로 맞춥니다.

## 전체 구성

| 항목 | 권장 설정 |
|---|---|
| 피험자 | 팀원 3명 |
| 측정 방식 | 한 명씩 순차 측정 |
| 자세 | 앉은 자세 |
| 기본 거리 | 1.0 m |
| 추가 거리 | 0.5 m, 1.5 m |
| 세션 길이 | 약 140초 |
| Target | BIOPAC respiration |
| 최종 산출물 | `UWB_Biopac_SyncData.mat` |

## 세션 목록

각 피험자당 아래 5개 세션을 수집합니다.

| 세션 | 거리 | 호흡 조건 | 자세 | 시간 |
|---|---:|---|---|---:|
| S01 | 1.0 m | normal breathing | sitting | 140초 |
| S02 | 0.5 m | normal breathing | sitting | 140초 |
| S03 | 1.5 m | normal breathing | sitting | 140초 |
| S04 | 1.0 m | slow breathing | sitting | 140초 |
| S05 | 1.0 m | fast breathing | sitting | 140초 |

여유가 있으면 아래 세션을 추가합니다.

| 세션 | 거리 | 호흡 조건 | 자세 | 목적 |
|---|---:|---|---|---|
| S06 | 1.0 m | normal + small movement | sitting | 움직임/artifact 강건성 비교 |

총 기본 데이터량:

```text
3명 x 5세션 x 140초 = 약 35분
```

## 장비 배치

```text
UWB radar
  - 피험자 정면
  - 가슴 방향
  - 높이: 가슴 중앙
  - 거리: 세션 조건에 맞춤

BIOPAC respiration belt
  - 모든 피험자 같은 위치에 착용
  - 가슴 또는 복부 중 하나로 고정
  - 벨트가 느슨하지 않게 확인
```

실험 중에는 주변 사람이 움직이지 않도록 하고, 피험자는 말하지 않습니다.

## 세션 내부 구조

각 세션은 아래 순서로 진행합니다.

```text
0~5초      가만히 대기
5~15초     시작 sync event: 큰 숨 3번
15~125초   본 측정
125~135초  종료 sync event: 큰 숨 3번
135~140초  가만히 대기 후 종료
```

sync event는 UWB와 BIOPAC 양쪽에서 시간축을 맞추기 위한 기준점입니다.

## 세션 절차 체크리스트

각 세션마다 아래 순서를 따릅니다.

```text
[ ] metadata.json 작성
[ ] BIOPAC respiration belt 착용 확인
[ ] UWB 거리/높이/각도 확인
[ ] UWB recording 준비
[ ] BIOPAC recording 준비
[ ] 두 장비 recording 시작
[ ] 5초 대기
[ ] 시작 sync event 수행
[ ] 본 측정 수행
[ ] 종료 sync event 수행
[ ] 두 장비 recording 종료
[ ] 파일 저장
[ ] notes.txt 작성
[ ] quick quality check 수행
```

## 폴더 구조

raw 데이터:

```text
data_raw/
  subject_001/
    session_001_1m_normal/
      metadata.json
      uwb_raw.dat
      biopac_raw.mat
      notes.txt
    session_002_05m_normal/
      ...
  subject_002/
  subject_003/
```

처리 후 데이터:

```text
data_processed/
  subject_001/
    session_001_1m_normal/
      UWB_Biopac_SyncData.mat
      sync_summary.json
      quality_preview.png
  subject_002/
  subject_003/
```

## metadata 예시

```json
{
  "subject_id": "subject_001",
  "session_id": "session_001_1m_normal",
  "distance_m": 1.0,
  "posture": "sitting",
  "breathing_condition": "normal",
  "duration_sec": 140,
  "sync_event_start": "3 deep breaths after 5 sec",
  "sync_event_end": "3 deep breaths before stop",
  "uwb_position": "front_chest",
  "uwb_height": "chest_level",
  "biopac_sensor": "respiration_belt",
  "biopac_position": "chest_or_abdomen",
  "uwb_sampling_rate_hz": 17,
  "biopac_sampling_rate_hz": 250,
  "notes": ""
}
```

## notes.txt에 적을 내용

아래 상황이 있으면 반드시 기록합니다.

```text
기침
웃음
말함
센서 만짐
자세 변화
벨트 헐거움
UWB 앞에 사람 지나감
장비 recording 지연
sync event 실수
```

## 수집 직후 품질 확인

세션이 끝날 때마다 바로 확인합니다.

| 확인 항목 | 기준 |
|---|---|
| UWB 파일 존재 | `uwb_raw.*` 저장됨 |
| BIOPAC 파일 존재 | `biopac_raw.*` 저장됨 |
| 길이 확인 | 약 140초 |
| BIOPAC 파형 | 호흡 주기가 눈으로 보임 |
| UWB heatmap | 피험자 range 위치가 보임 |
| 시작 sync event | 양쪽 신호에 큰 변화가 보임 |
| 종료 sync event | 양쪽 신호에 큰 변화가 보임 |

하나라도 실패하면 해당 세션은 바로 다시 측정합니다.

## 처리 및 비교 흐름

```text
UWB raw + BIOPAC raw
-> 대학원생 코드 스타일 동기화/전처리
-> UWB_Biopac_SyncData.mat
-> CNN continuous baseline
-> SNN rate / delta / level-crossing
-> SNN adaptive delta-rate hybrid
-> SNN hybrid + Spiking TCN
```

평가 지표:

```text
RMSE
MAE
Correlation
input spikes/sec
hidden spike rate
noise/artifact 성능 저하율
```

## Pilot 권장

본수집 전에 아래 조건으로 pilot을 먼저 진행합니다.

| 항목 | 설정 |
|---|---|
| 피험자 | 1명 |
| 거리 | 1.0 m |
| 호흡 | normal breathing |
| 시간 | 140초 |
| 목적 | 저장, 동기화, SyncData 변환 가능 여부 확인 |

pilot에서 아래가 모두 성공해야 본수집으로 넘어갑니다.

```text
UWB raw 저장 성공
BIOPAC raw 저장 성공
sync event 확인
SyncData-like 변환 성공
preview plot 정상
```

## 최종 권장 순서

```text
1. Pilot 1세션 수집
2. quick quality check
3. SyncData-like 변환
4. CNN/SNN smoke test
5. 팀원 3명 본수집
6. 전체 CNN/SNN 비교
7. noise/motion 구간 성능 방어율 분석
```

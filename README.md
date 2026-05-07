# VitalSync-kw

UWB 레이더 데이터와 BIOPAC 호흡 신호를 활용한 생체신호 분석 프로젝트입니다.

## 현재 실험 정리

- 전체 비교 결과와 추천 방법론: [docs/experiment_summary_and_recommendation.md](docs/experiment_summary_and_recommendation.md)
- UWB + BIOPAC 데이터 수집 프로토콜: [docs/data_collection_protocol.md](docs/data_collection_protocol.md)
- MobiVital SNN 방향성 실험: [docs/uwb_snn_direction_check.md](docs/uwb_snn_direction_check.md)
- AAL SyncData-like CNN/SNN 비교: [docs/aal_syncdata_cnn_snn_comparison.md](docs/aal_syncdata_cnn_snn_comparison.md)

## 개발 환경 설정

팀원 간 환경 차이를 줄이기 위해 Python 3.11.x 사용을 권장합니다.

### 1. 저장소 받기

```powershell
git clone https://github.com/ldw0228/VitalSync-kw.git
cd VitalSync-kw
git checkout develop
```

이미 저장소를 받은 상태라면 최신 내용을 가져옵니다.

```powershell
git pull origin develop
```

### 2. Python 버전 확인

```powershell
py -3.11 --version
```

`Python 3.11.x`가 출력되면 준비된 상태입니다.

### 3. 가상환경 생성

```powershell
py -3.11 -m venv .venv
```

### 4. 가상환경 활성화

Windows PowerShell 기준:

```powershell
.\.venv\Scripts\activate
```

활성화되면 터미널 앞에 `(.venv)`가 표시됩니다.

### 5. 패키지 설치

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. 설치 확인

```powershell
python -c "import numpy, pandas, scipy, matplotlib, snntorch; print('환경 설정 완료')"
```

## 작업 시 주의사항

- `.venv` 폴더는 개인 로컬 환경이므로 GitHub에 올리지 않습니다.
- 대용량 데이터 파일(`.mat` 등)은 저장소에 직접 올리지 않고 별도 공유 저장소를 사용합니다.
- 새로운 패키지가 필요하면 `requirements.txt`에 추가하고 팀원에게 공유합니다.

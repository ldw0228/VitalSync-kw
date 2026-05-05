# MobiVital SNN Starter

이 폴더는 MobiVital README의 컬럼 정의만 참고해서, 우리가 직접 PyTorch용 loader와 spike encoding 파이프라인을 만드는 실험용 코드입니다.

현재 목적은 최종 모델 완성이 아니라, 졸업작품에서 어떤 방식으로 UWB 생체신호를 SNN에 넣을지 방향을 잡는 것입니다.

## 데이터 가정

MobiVital CSV는 한 행이 하나의 timestamp입니다.

- 샘플링 주파수: 50 Hz
- 1-6열: IMU
- 13-132열: UWB I 데이터, 120개 range bin
- 133-252열: UWB Q 데이터, 120개 range bin
- 253열: respiration waveform label
- 254열: pulse waveform label

위 컬럼 번호는 데이터셋 README 기준의 1-based 번호입니다.

## 현재 파이프라인

```text
sample.csv
-> UWB I/Q 분리
-> magnitude 또는 phase 특징 생성
-> range bin 선택
-> sliding window 생성
-> delta spike encoding
-> PyTorch Dataset 출력
```

## 빠른 실행

```powershell
python quick_inspect.py --csv C:\Users\hai\Desktop\uwb_sample\sample.csv
```

Python 환경이 없다면 먼저 Python을 설치한 뒤 아래 명령을 실행합니다.

```powershell
pip install -r requirements.txt
```

Windows에서는 PyTorch CPU wheel index를 사용하는 것이 안전합니다.

```powershell
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

## 왜 이 방향인가

선배 연구실 MATLAB 코드는 UWB/BIOPAC 전처리 baseline으로 활용할 수 있습니다. 하지만 그 코드는 SNN을 위한 spike representation을 만드는 코드는 아닙니다.

졸업작품에서 우리가 가져갈 수 있는 기여는 아래처럼 잡는 것이 좋습니다.

```text
기존/전처리된 radar signal
vs
delta 또는 delta-rate spike representation + SNN
```

즉, 핵심은 "SNN 모델을 그냥 붙인다"가 아니라 "UWB 생체신호를 SNN에 맞는 spike-friendly 데이터 표현으로 바꾸고, 그 효과를 CNN baseline과 비교한다"입니다.

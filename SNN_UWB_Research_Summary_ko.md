# SNN + UWB 연구 요약

최종 업데이트: 2026-05-12

## 1. Executive Summary

이 문서는 Spiking Neural Network(SNN)를 Ultra-Wideband(UWB) 데이터 처리에 어떻게 활용할 수 있는지 조사한 리서치 요약이다. 범위는 연구 조사와 방법론 정리에 한정하며, 구현은 포함하지 않는다.

핵심 결론:

- SNN + UWB 직접 연구는 존재하지만 아직 많지 않다. 가장 관련성이 높은 직접 사례는 UWB 채널 추정, IR-UWB 제스처 인식, UWB 레이더 기반 의료 분류다.
- 가장 강한 인접 근거는 FMCW/mmWave/automotive radar 및 RF 신호 처리 분야에서 나온다. 이 분야에서는 SNN이 제스처 인식, 객체 탐지, 스펙트럼 감지, 레이더 방사체 인식, 저전력 신호 처리 파이프라인에 사용되고 있다.
- UWB CIR/range-profile 데이터에 대해 가장 현실적인 첫 접근은 raw waveform을 바로 spike화하는 것이 아니다. 먼저 CIR 또는 range bin을 정규화한 뒤 rate, latency, threshold/event encoding을 적용하는 feature-first encoding 접근이 더 실용적이다.
- 저전력 또는 실시간 sensing에서는 sparse spike가 연산 활동량을 줄일 수 있기 때문에 SNN이 매력적이다. 다만 초기 실험에서 정확도를 확보하려면 pure STDP보다 surrogate-gradient SNN 또는 hybrid CNN-SNN이 더 현실적이다.
- 추천 첫 연구 경로는 CIR/range-profile sequence를 입력으로 하는 LOS/NLOS classification 또는 gesture/activity classification이며, 작은 LIF 기반 convolutional SNN 또는 recurrent SNN으로 시작하는 것이 좋다.

아래에서는 UWB 직접 근거와 인접 근거를 분리한다. UWB에 대한 응용 가능성 추론은 그렇게 표시했다.

## 2. SNN Background and Methodologies

### 2.1 Neuron Models

| 모델 | 핵심 아이디어 | 장점 | 한계 | UWB 관련성 |
|---|---|---|---|---|
| IF | 입력을 threshold까지 적분한 뒤 spike 발생 | 단순하고 효율적 | leakage dynamics가 없음 | encoding된 CIR/range bin의 baseline |
| LIF | membrane leak가 있는 integrate-and-fire | 일반적이고 안정적이며 hardware-friendly | time constant 튜닝 필요 | UWB sequence의 강력한 첫 선택지 |
| ALIF | adaptive threshold 또는 adaptive current | history와 temporal adaptation 포착 | 파라미터 증가 | multipath와 temporal drift에 유용 |
| Izhikevich | 생물학적으로 더 풍부한 spiking dynamics | 다양한 firing pattern | LIF보다 무거움 | 첫 UWB 실험에는 보통 불필요 |
| SRM | kernel을 사용하는 Spike Response Model | 유연한 event response modeling | 설계 선택지가 많음 | UWB event를 temporal impulse로 다룰 때 유용 |

실용적 권장: LIF 또는 ALIF부터 시작한다. UWB 데이터는 time-of-flight, multipath, frame sequence에서 이미 시간 구조를 갖고 있으므로, 생물학적으로 복잡한 모델보다 time constant를 잘 제어할 수 있는 단순 neuron이 디버깅하기 쉽다.

### 2.2 Network Families

| 계열 | 설명 | UWB에서 사용할 상황 |
|---|---|---|
| Feedforward SNN | dense 또는 MLP 형태의 spiking layer | 작은 feature vector, channel-state classification |
| Convolutional SNN | bin 또는 spectrogram 위의 spatial/temporal filter | CIR, range profile, time-frequency map |
| Recurrent SNN | recurrent state를 통한 temporal memory | tracking, activity recognition, sequential localization |
| Hybrid ANN-SNN | ANN front-end 또는 feature extractor와 spiking layer 결합 | 정확도 우선 실험, noisy UWB data |
| Liquid State Machine (LSM) | 고정 recurrent reservoir와 학습 가능한 readout | label이 제한된 channel estimation 또는 time-series classification |
| ANN-to-SNN conversion | ANN을 학습한 뒤 activation을 spike로 변환 | ANN baseline이 이미 잘 동작하고 latency가 허용될 때 |

### 2.3 Learning Methods

| 방법 | 요약 | 장점 | 단점 | UWB 적합성 |
|---|---|---|---|---|
| Surrogate gradient learning | non-differentiable spike function을 smooth approximation으로 우회해 backpropagation 수행 | supervised 정확도 측면에서 가장 실용적 | time-unrolled training 필요 | LOS/NLOS, gesture, activity의 강한 기본 선택지 |
| STDP | local spike-timing-dependent plasticity | 생물학적으로 그럴듯하고 unsupervised | 목표 metric 최적화가 어려움 | feature discovery 또는 neuromorphic demo에 유용 |
| ANN-to-SNN conversion | 학습된 ANN activation을 firing rate로 변환 | 성숙한 ANN training 활용 | 많은 time step이 필요할 수 있음 | power/latency가 부차적이면 좋음 |
| Reservoir/LSM training | 고정 spiking reservoir와 trainable readout | 효율적이고 temporal dynamics 활용 | reservoir 설계가 중요 | UWB channel estimation에 유망 |
| Evolutionary topology search | SNN topology/weight를 진화적으로 탐색 | compact model을 찾을 수 있음 | 비용이 크고 재현이 어려움 | 초기 UWB medical radar 논문에서 사용 |

## 3. UWB Data Representations

| 표현 | 설명 | 장점 | 위험/한계 | SNN 변환 아이디어 |
|---|---|---|---|---|
| Raw waveform | time-domain 수신 UWB pulse sample | 정보량 최대 | high rate, noisy, difficult | peak에 threshold-crossing 또는 latency coding |
| CIR / Channel Impulse Response | delay에 따른 channel tap | multipath를 직접 포착 | synchronization과 calibration에 민감 | tap별 또는 선택 path별 rate/latency encoding |
| Range profile | range bin별 energy | compact하고 해석 가능 | phase/fine waveform detail 손실 | bin energy rate coding 또는 변화량 event coding |
| Time-of-flight features | first path, peak path, delay spread | 작은 feature set | feature engineering이 cue를 숨길 수 있음 | LIF/MLP-SNN에 직접 current injection |
| Spectrogram / time-frequency map | 시간에 따른 frequency content | CNN-like model과 잘 맞음 | 짧은 pulse에는 과할 수 있음 | 2D map 위 convolutional SNN |
| Radar-like frame sequence | Range-Doppler, range-angle 또는 frame stack | 인접 radar SNN 문헌과 잘 맞음 | UWB가 항상 Doppler/angle을 갖는 것은 아님 | frame sequence 위 ConvSNN 또는 recurrent SNN |
| Event/spike representation | amplitude 변화 또는 threshold crossing에서 event 생성 | sparse하고 neuromorphic-friendly | encoding 설계가 결과에 큰 영향 | delta modulation, threshold crossing, TTFS |

실용적인 첫 선택지는 CIR 또는 range profile이다. 이 표현들은 UWB 특유의 time-of-flight/multipath 구조를 보존하면서 full raw waveform을 직접 모델링해야 하는 부담을 피한다.

## 4. Spike Encoding Methods for UWB-like Signals

| Encoding | 동작 방식 | 적합한 UWB 입력 | 장점 | 주의점 |
|---|---|---|---|---|
| Rate coding | 값이 클수록 window 동안 더 많은 spike 생성 | CIR amplitude, range energy, spectrogram magnitude | 단순하고 robust하며 널리 지원됨 | spike 수와 latency 증가 |
| Temporal coding | spike timing pattern에 정보 저장 | CIR peak, waveform peak | timing 구조 보존 | synchronization에 민감 |
| Latency coding | 값이 강할수록 더 빨리 spike 발생 | range bin, CIR tap | 효율적이고 해석 가능 | normalization 필요, 약한 feature가 사라질 수 있음 |
| Time-to-first-spike | 첫 spike timing으로 의사결정 | first path / strongest path feature | 낮은 latency | 뒤쪽 multipath 정보를 버릴 수 있음 |
| Population coding | 하나의 변수를 여러 neuron으로 encoding | ToF, range, angle, channel statistics | smooth representation | neuron 수 증가 |
| Threshold/event encoding | signal이 threshold를 넘거나 충분히 변할 때 spike 발생 | raw waveform, frame 간 CIR 변화 | sparse하고 event-like | threshold tuning이 중요 |
| Frequency/rate bin encoding | numeric feature를 firing frequency로 매핑 | channel-estimation feature vector | UWB channel-estimation 작업에서 사용 | precise delay ordering 손실 가능 |

UWB waveform/CIR/range profile에 대한 권장:

- Raw waveform: threshold/event encoding 또는 temporal coding.
- CIR: path delay에는 latency coding, tap strength에는 rate coding, frame-to-frame 변화에는 event coding.
- Range profile: range bin 위 rate 또는 latency coding.
- Time-series UWB sensing: encoding된 frame sequence 위 recurrent SNN 또는 ConvSNN.

## 5. Paper Survey

### 5.1 Direct UWB + SNN Papers

#### Paper D1 - Exploring the Potential of Spiking Neural Networks in UWB Channel Estimation

- 저자: Youdong Zhang, Xu He, Xiaolin Meng
- 연도: 2025
- venue/source: arXiv
- 원문: https://arxiv.org/abs/2512.23975
- PDF: https://arxiv.org/pdf/2512.23975
- 분류: Direct UWB
- 연구 문제: SNN을 이용한 UWB channel estimation.
- 데이터/신호: eWINE/UWB 관련 channel model에서 나온 UWB channel data. RF/channel feature와 channel impulse response 정보를 사용한다.
- 입력 표현: 한 실험에서는 10차원 RF feature vector, 다른 실험에서는 50 또는 120차원 CIR vector를 사용.
- Spike encoding: channel feature를 spike train으로 변환하기 위해 rate/frequency-style encoding 사용.
- SNN 구조: 400 또는 500개의 LIF neuron을 가진 Liquid State Machine과 spiking self-organizing map component.
- 학습 방법: channel estimation task를 위한 STDP-like unsupervised spiking learning 및 readout/evaluation.
- 평가/결과: 설명된 설정에서 약 80 percent 수준의 estimation performance 보고.
- 한계: 최근 exploratory 성격의 논문으로 보이며, 성숙한 non-spiking UWB estimator와 폭넓은 benchmark는 아직 부족하다.
- UWB 의의: UWB channel feature와 CIR vector를 SNN 입력으로 다루는 데 가장 직접적으로 관련된 자료.

#### Paper D2 - Hand Gesture Recognition Using IR-UWB Radar with Spiking Neural Networks

- 저자: Shule Wang, Yulong Yan, Haoming Chu, Guangxi Hu, Zhi Zhang, Zhuo Zou, Lirong Zheng
- 연도: 2022
- venue/source: IEEE International Conference on Artificial Intelligence Circuits and Systems (AICAS)
- DOI: 10.1109/AICAS54282.2022.9870013
- 원문/DOI page: https://ieeexplore.ieee.org/document/9870013
- proceedings index: https://researchr.org/publication/aicas-2022
- 분류: Direct UWB
- 연구 문제: impulse-radio UWB radar와 SNN을 이용한 hand gesture recognition.
- 데이터/신호: IR-UWB radar gesture signal.
- 입력 표현: 공개 metadata는 IR-UWB gesture recognition을 설명하지만, 정확한 tensor/feature 형태는 full paper 확인이 필요하다.
- Spike encoding: 공개 metadata만으로는 완전히 확인되지 않음.
- SNN 구조: 공개 metadata만으로는 완전히 확인되지 않음.
- 학습 방법: 공개 metadata만으로는 완전히 확인되지 않음.
- 평가/결과: 공개 metadata는 비교 deep-learning baseline보다 높은 gesture-recognition accuracy를 보였다고 설명하지만, 정확한 metric은 IEEE full text 확인이 필요하다.
- 한계: 자세한 방법과 수치 결과 확인에는 IEEE 접근이 필요할 수 있음.
- UWB 의의: UWB sensing과 직접적으로 연결된다. IR-UWB radar와 SNN을 구체적인 classification task로 연결하므로 full-text 우선순위가 높다.

#### Paper D3 - Spiking Neural Networks for Breast Cancer Classification in a Dielectrically Heterogeneous Breast

- 저자: Martin O'Halloran, Brendan McGinley, Elfed Lewis, Martin Glavin, Edward Jones
- 연도: 2011
- venue/source: Progress In Electromagnetics Research
- 원문/PDF: https://www.jpier.org/issues/volume.html?paper=11071904
- 분류: Direct UWB radar medical sensing
- 연구 문제: UWB radar-derived target signature를 이용한 breast cancer classification.
- 데이터/신호: dielectrically heterogeneous breast model에서 나온 UWB radar target signature.
- 입력 표현: radar target-signature feature.
- Spike encoding: 논문은 SNN classifier를 사용하지만, 재현 전 정확한 encoding detail은 PDF에서 확인해야 한다.
- SNN 구조: fixed-topology SNN classifier.
- 학습 방법: supervised SNN classification approach.
- 평가/결과: UWB radar medical signature에 대한 SNN-based classification의 가능성을 보임.
- 한계: medical simulation task이며 오래된 SNN 방법론이다. indoor UWB sensing에 바로 전이되지는 않는다.
- UWB 의의: UWB radar signature를 SNN classifier에 매핑할 수 있다는 초기 근거.

#### Paper D4 - Evolving Spiking Neural Networks for Breast Cancer Classification in a Dielectrically Heterogeneous Breast

- 저자: Brendan McGinley, Martin O'Halloran, Elfed Lewis, Martin Glavin, Edward Jones
- 연도: 2011
- venue/source: Progress In Electromagnetics Research Letters
- 원문/PDF: https://www.jpier.org/issues/volume.html?paper=11100304
- 분류: Direct UWB radar medical sensing
- 연구 문제: evolved SNN topology를 이용해 UWB radar breast-cancer classification 개선.
- 데이터/신호: heterogeneous breast simulation에서 나온 UWB radar target signature.
- 입력 표현: target-signature feature.
- Spike encoding: feature-to-spike 기반 SNN classification으로 보이며, 정확한 encoding은 full text 확인 필요.
- SNN 구조: evolved-topology SNN.
- 학습 방법: evolutionary topology/parameter search.
- 평가/결과: fixed-topology SNN의 compact alternative를 제시.
- 한계: 오래된 medical radar setting이며, 현대 dataset 대비 small/simulation-based일 가능성이 높다.
- UWB 의의: hand-designed LIF network가 불안정할 경우 topology search를 고려할 수 있음을 보여줌.

#### Paper D5 - An Embedded Hardware Spiking Neural Network Targeted for Unsupervised UWB Radar-Based Bladder Volume Monitoring

- 저자: Irene Krewer, Damien Coyle, Barry McGinley, Martin O'Halloran, Martin Glavin, Edward Jones
- 연도: 2013
- venue/source: IEEE BioCAS poster/proceedings record
- 원문/record: https://pure.ulster.ac.uk/en/publications/an-embedded-hardware-spiking-neural-network-targeted-for-unsupervi
- 분류: Direct UWB radar medical sensing
- 연구 문제: UWB radar-based bladder volume monitoring을 위한 embedded unsupervised SNN.
- 데이터/신호: bladder volume과 관련된 UWB radar measurement.
- 입력 표현: radar-derived feature.
- Spike encoding/SNN/learning: unsupervised embedded SNN이며, 자세한 기술 내용은 source paper/poster가 필요하다.
- 평가/결과: source는 방법과 target application을 식별하지만, 공개 metadata는 제한적이다.
- 한계: full artifact 없이는 재현성을 평가하기 어렵다.
- UWB 의의: low-power embedded UWB-SNN application의 선례로 유용하다.

### 5.2 Adjacent Radar, RF, and Wireless SNN Papers

#### Paper A1 - Radar-Based Hand Gesture Recognition Using Spiking Neural Networks

- 저자: S. H. Tsang et al.
- 연도: 2021
- venue/source: IEEE Sensors Journal
- 원문/DOI page: https://ieeexplore.ieee.org/document/9420974
- 관련 오픈소스 코드: https://github.com/SoftwareImpacts/SIMPAC-2021-111
- 분류: Adjacent radar sensing
- 연구 문제: SNN을 이용한 radar hand-gesture classification.
- 데이터/신호: FMCW radar gesture data. 일반적으로 time/range/range-Doppler-like tensor로 표현된다.
- 입력 표현: preprocessed radar frame.
- Spike encoding: convolutional SNN pipeline에 구현되어 있으며, 정확한 encoding은 paper 확인 필요.
- SNN 구조: Convolutional SNN.
- 학습 방법: supervised learning.
- 평가/결과: public code와 함께 SNN 기반 radar gesture recognition의 실제 가능성을 보여준다.
- 한계: FMCW radar는 UWB가 아니므로 representation transfer를 조정해야 한다.
- UWB 의의: UWB gesture/activity classification architecture를 위한 가장 좋은 open implementation reference.

#### Paper A2 - Improving the Accuracy of Spiking Neural Networks for Radar Gesture Recognition through Preprocessing

- 저자: B. Safa et al.
- 연도: 2021/2022
- venue/source: IEEE 관련 publication. ConvSNN repository에서 code/paper가 자주 참조됨.
- 원문/source hub: https://github.com/SoftwareImpacts/SIMPAC-2021-111
- 분류: Adjacent radar sensing
- 연구 문제: signal preprocessing으로 radar SNN gesture recognition 개선.
- 데이터/신호: radar gesture data.
- 입력 표현: preprocessed radar feature.
- Spike encoding: ConvSNN pipeline.
- SNN 구조: Convolutional SNN.
- 학습 방법: supervised SNN training.
- 평가/결과: preprocessing이 model choice만큼 중요할 수 있음을 보여줌.
- 한계: formal citation 전에 정확한 publication record와 metric 확인 필요.
- UWB 의의: UWB preprocessing과 normalization이 SNN 성능을 좌우할 수 있다는 중요한 경고.

#### Paper A3 - A 2-uJ, 12-Class, 91% Accuracy Spiking Neural Network Approach for Radar Gesture Recognition

- 저자: Ali Safa, Andre Bourdoux, Ilja Ocket, Francky Catthoor, Georges G. E. Gielen
- 연도: 2021
- venue/source: arXiv / Electronics-related radar SNN work
- 원문: https://arxiv.org/abs/2108.02669
- 분류: Adjacent low-power radar sensing
- 연구 문제: SNN을 이용한 low-energy radar gesture recognition.
- 데이터/신호: radar gesture data.
- 입력 표현: radar feature frame.
- Spike encoding: SNN spike-based inference.
- SNN 구조: compact SNN.
- 학습 방법: supervised training.
- 평가/결과: 두 radar gesture dataset에서 91 percent 이상의 accuracy와 classification당 약 2 microjoule energy를 보고.
- 한계: energy number를 deployment claim으로 사용하기 전 hardware assumption 확인 필요.
- UWB 의의: SNN + UWB embedded sensing의 low-power motivation을 뒷받침한다.

#### Paper A4 - Resource-Efficient Gesture Sensing Based on FMCW Radar Using Spiking Neural Networks

- 저자: M. Arsalan et al.
- 연도: 2021
- venue/source: IEEE MTT-S International Microwave Symposium (IMS)
- 원문: https://ieeexplore.ieee.org/document/9574994
- 분류: Adjacent radar sensing
- 연구 문제: SNN을 이용한 resource-efficient radar gesture recognition.
- 데이터/신호: FMCW radar intermediate-frequency 또는 radar-derived signal.
- 입력 표현: SNN processing에 적합한 radar signal representation.
- Spike encoding: spike-based temporal processing.
- SNN 구조: SNN gesture classifier.
- 학습 방법: supervised SNN approach.
- 평가/결과: FMCW radar data에서 energy/resource-oriented radar gesture sensing을 보여줌.
- 한계: FMCW assumption은 impulse UWB와 다르다.
- UWB 의의: compute와 power가 중요한 UWB edge-sensing experiment 설계에 유용하다.

#### Paper A5 - Automotive Radar Processing With Spiking Neural Networks: Concepts and Challenges

- 저자: Bernhard Vogginger, Felix Kreutz, Javier Lopez-Randulfe, Chen Liu, Robin Dietrich, Hector A. Gonzalez, Daniel Scholz, Nico Reeb, Daniel Auge, Julian Hille, Muhammad Arsalan, Florian Mirus, Cyprian Grassmann, Alois Knoll, Christian Mayr, et al.
- 연도: 2022
- venue/source: Frontiers in Neuroscience
- 원문: https://www.frontiersin.org/articles/10.3389/fnins.2022.851774
- 분류: Adjacent radar methodology
- 연구 문제: radar processing에서 SNN 사용을 survey/discuss.
- 데이터/신호: signal processing과 perception task를 포함한 radar pipeline 전반.
- 입력 표현: 여러 radar representation 논의.
- Spike encoding: radar signal을 spike로 mapping하는 어려움을 논의.
- SNN 구조: general radar SNN concept.
- 학습 방법: general.
- 평가/결과: 단일 benchmark보다는 conceptual/methodological 논문.
- 한계: UWB-specific은 아니다.
- UWB 의의: radar-to-spike conversion이 어려운 이유를 이해하는 데 좋은 background.

#### Paper A6 - SpikingRTNH: Spiking Residual Transformer with Neural Heterogeneity for 4D Radar Object Detection

- 저자: Dong-Hee Paek, Seung-Hyun Kong
- 연도: 2025
- venue/source: arXiv / IEEE Intelligent Vehicles Symposium (IV) project listing
- 원문: https://arxiv.org/abs/2502.00074
- project/code: https://github.com/kaist-avelab/K-Radar
- 분류: Adjacent 4D radar object detection
- 연구 문제: spiking transformer-style architecture를 이용한 4D radar object detection.
- 데이터/신호: 4D radar tensor data.
- 입력 표현: radar cube/tensor representation.
- Spike encoding: SNN-compatible radar tensor processing.
- SNN 구조: RTNH-style 4D radar detector의 LIF 기반 spiking replacement와 biological top-down inference(BTI).
- 학습 방법: supervised deep SNN training.
- 평가/결과: ANN counterpart와 비슷한 3D/BEV detection performance를 유지하면서 78 percent energy reduction 보고.
- 한계: 4D radar detection은 UWB CIR classification과 거리가 있으며, 첫 UWB experiment로는 model이 너무 무겁다.
- UWB 의의: high-dimensional radar tensor 위 spiking architecture에 대한 advanced reference.

#### Paper A7 - Spiking Neural Network for Fourier Transform and Object Detection for Automotive Radar

- 저자: S. Lopez-Randulfe et al.
- 연도: 2021/2022
- venue/source: Frontiers in Neurorobotics / neuromorphic radar literature
- 원문: https://www.frontiersin.org/articles/10.3389/fnbot.2021.688344/full
- 분류: Adjacent radar signal processing
- 연구 문제: radar processing과 object detection에서 Fourier-transform-like 처리를 SNN으로 수행.
- 데이터/신호: automotive radar signal.
- 입력 표현: detection으로 이어지는 radar signal pipeline.
- Spike encoding: SNN-based signal processing component.
- SNN 구조: transform과 detection을 위한 SNN module.
- 학습 방법: task-specific spiking processing.
- 평가/결과: SNN이 final classifier뿐 아니라 radar processing chain의 더 앞단에서도 사용될 수 있음을 보여줌.
- 한계: automotive radar pipeline은 UWB time-of-flight sensing과 다르다.
- UWB 의의: 미래 연구로 spike-domain preprocessing을 통한 CIR/range extraction 가능성을 제안한다.

#### Paper A8 - NeuroRadar: A Neuromorphic Radar Sensor for Low-Power IoT Systems

- 저자: Kai Zheng, Kun Qian, Timothy Woodford, Xinyu Zhang
- 연도: 2023; CACM research-highlight version은 2025년에 게재
- venue/source: ACM SenSys / Communications of the ACM highlight
- 원문/DOI page: https://dl.acm.org/doi/10.1145/3625687.3625795
- Open PDF: https://xyzhang.ucsd.edu/papers/Kai.Zheng_SenSys23_NeuroRadar.pdf
- 분류: Adjacent neuromorphic radar hardware
- 연구 문제: low-power IoT를 위한 neuromorphic radar sensing.
- 데이터/신호: radar sensor data.
- 입력 표현: event/neuromorphic radar representation.
- Spike encoding: sensor-level neuromorphic/event processing.
- SNN 구조: neuromorphic processing pipeline.
- 학습 방법: model training만이 아니라 system-level design.
- 평가/결과: 기존 radar pipeline보다 훨씬 낮은 power로 gesture recognition과 localization case study를 보고.
- 한계: hardware/system assumption이 commodity UWB에 그대로 전이되지 않을 수 있음.
- UWB 의의: event-driven UWB sensor라는 장기 vision에 유용하다.

#### Paper A9 - Spiking Neural Networks for Radar Emitter Recognition

- 저자: Y. Luo et al.
- 연도: 2024
- venue/source: MDPI Remote Sensing
- 원문: https://www.mdpi.com/2072-4292/16/14/2680
- 분류: Adjacent RF/radar classification
- 연구 문제: SNN을 이용한 radar emitter classification.
- 데이터/신호: radar emitter signal feature.
- 입력 표현: classification을 위한 signal feature 또는 time-series representation.
- Spike encoding: radar emitter data의 SNN-compatible encoding.
- SNN 구조: SNN classifier.
- 학습 방법: supervised learning.
- 평가/결과: non-image radar/RF signal에 대한 SNN classification 가능성을 보여줌.
- 한계: emitter recognition은 UWB localization/sensing과 다르다.
- UWB 의의: UWB task가 geometric localization보다 signal/channel state classification에 가까울 경우 유용하다.

#### Paper A10 - RF Fingerprinting Identification Based on Spiking Neural Network for LEO-MIMO Systems

- 저자: Q. Jiang, J. Sha
- 연도: 2023
- venue/source: IEEE Wireless Communications Letters
- DOI/record: https://doi.org/10.1109/LWC.2022.3223939
- 분류: Adjacent RF/wireless signal processing
- 연구 문제: LEO-MIMO communication system을 위한 energy-efficient RF fingerprint identification.
- 데이터/신호: OFDM/RF fingerprinting feature.
- 입력 표현: channel-independent RF fingerprint feature와 augmentation.
- Spike encoding: SNN-compatible RF feature encoding.
- SNN 구조: RF fingerprint identification용 SNN classifier.
- 학습 방법: supervised learning.
- 평가/결과: 25 dB SNR에서 최대 95.26 percent identification accuracy와 FPGA에서 comparable model 대비 63.3 percent power reduction 보고.
- 한계: wireless device identification은 UWB ranging 또는 radar sensing이 아니다.
- UWB 의의: SNN을 이용한 low-power RF feature classification의 강한 인접 근거.

### 5.3 General SNN Methodology Papers to Read

#### Paper M1 - Surrogate Gradient Learning in Spiking Neural Networks

- 저자: E. O. Neftci, H. Mostafa, F. Zenke
- 연도: 2019
- 원문: https://arxiv.org/abs/1901.09948
- 역할: modern supervised SNN을 위한 핵심 training-method reference.
- UWB 의의: CIR/range-profile classification에서 surrogate gradient 사용을 정당화하는 데 활용.

#### Paper M2 - Deep Learning in Spiking Neural Networks

- 저자: A. Tavanaei, M. Ghodrati, S. R. Kheradpisheh, T. Masquelier, A. Maida
- 연도: 2019
- 원문: https://www.sciencedirect.com/science/article/pii/S0893608018303332
- 역할: deep SNN architecture와 learning에 대한 broad overview.
- UWB 의의: conversion, STDP, surrogate learning 중 선택할 때 좋은 background.

#### Paper M3 - A Review of Encoding Techniques for Spiking Neural Networks

- 저자: D. Auge, J. Hille, E. Mueller, A. Knoll
- 연도: 2021
- 원문: https://www.sciencedirect.com/science/article/pii/S0925231221009722
- 역할: encoding-method reference.
- UWB 의의: rate, latency, temporal, population coding 선택을 정당화하는 데 사용.

#### Paper M4 - A Review of Learning in Biologically Plausible Spiking Neural Networks

- 저자: A. Taherkhani et al.
- 연도: 2020
- 원문: https://www.sciencedirect.com/science/article/pii/S0893608020303573
- 역할: STDP와 biologically plausible learning review.
- UWB 의의: 연구 방향이 unsupervised 또는 neuromorphic learning을 강조할 때 유용.

## 6. Open-source and Project Survey

| 프로젝트 | 링크 | 관련 논문/문서 | 프레임워크 | 신호/데이터 종류 | SNN/encoding 지원 | 유지보수/유용성 | UWB 의의 |
|---|---|---|---|---|---|---|---|
| ConvSNN radar gesture code | https://github.com/SoftwareImpacts/SIMPAC-2021-111 | Tsang/Safa et al.의 radar gesture SNN 논문 | Python/PyTorch 계열 연구 코드 | radar gesture data | radar gesture용 convolutional SNN | task-adjacent이고 공개 코드라 가치 있음 | UWB gesture/activity recognition의 가장 구체적인 시작 reference |
| K-Radar / SpikingRTNH | https://github.com/kaist-avelab/K-Radar | K-Radar 및 SpikingRTNH 논문 | PyTorch ecosystem | 4D radar tensor | spiking residual transformer-style radar detection | 큰 active research repo | high-dimensional radar SNN의 advanced reference. MVP에는 너무 무거움 |
| snnTorch | https://github.com/jeshraghian/snntorch | Docs: https://snntorch.readthedocs.io/ | PyTorch | general SNN | LIF, surrogate gradient, tutorial | 강력한 실용 library | 첫 UWB classification experiment에 추천 |
| Norse | https://github.com/norse/norse | Docs: https://norse.github.io/norse/ | PyTorch | general SNN | LIF/LSNN module, surrogate training | 성숙한 research library | UWB sequence용 recurrent/temporal SNN에 좋음 |
| SpikingJelly | https://github.com/fangwei123456/spikingjelly | Docs: https://spikingjelly.readthedocs.io/ | PyTorch | general SNN | ANN-to-SNN, surrogate training, encoder | 기능 폭이 넓음 | direct SNN training과 ANN-to-SNN conversion 비교에 유용 |
| Lava | https://github.com/lava-nc/lava | Docs: https://lava-nc.org/ | Intel neuromorphic software stack | neuromorphic algorithm 및 deployment | process-based SNN modeling, Loihi-oriented workflow | neuromorphic deployment에 강함 | model concept이 안정된 뒤 hardware mapping이 중요할 때 유용 |
| BindsNET | https://github.com/BindsNET/bindsnet | repo 내 paper/docs | PyTorch | general SNN, STDP experiment | STDP 및 biologically inspired model | 오래됐지만 STDP prototype에 유용 | unsupervised UWB feature learning 탐색에 좋음 |
| Rockpool | https://github.com/synsense/rockpool | Docs: https://rockpool.ai/ | Python/JAX/Torch ecosystem | neuromorphic SNN 및 audio/event data | spiking layer와 deployment-oriented tooling | edge/neuromorphic workflow에 유용 | 효율적 deployment idea를 위한 later-stage tool |
| Tonic | https://github.com/neuromorphs/tonic | Docs: https://tonic.readthedocs.io/ | Python | event-based dataset | dataset transform, event handling | 유용한 support library | UWB를 event stream으로 변환할 때 도움 |
| Brian2 | https://github.com/brian-team/brian2 | Docs: https://brian2.readthedocs.io/ | Python simulation | biophysical/neural simulation | flexible neuron modeling | scientific simulation에는 뛰어나지만 deep learning용은 덜 적합 | custom UWB spike encoding 또는 neuron dynamics 테스트에 유용 |

오픈소스 주의점:

- 대부분의 SNN library는 general-purpose이며, UWB-specific loader를 포함한 경우는 거의 없다.
- radar SNN repository는 직접 code reuse보다 architecture와 preprocessing pattern 참고용으로 더 유용하다.
- 이 프로젝트에서는 snnTorch 또는 Norse가 재현 가능한 실험으로 가장 빨리 가는 길일 가능성이 높다. Lava/Rockpool은 neuromorphic hardware deployment가 목표에 포함될 때 더 관련성이 커진다.

## 7. Task-by-task Applicability

| UWB task | SNN 적합성 | 추천 입력 | 추천 encoding | 추천 모델 | 메모 |
|---|---|---|---|---|---|
| Ranging | Medium | CIR peak, first path, ToF feature | Latency 또는 TTFS | small feedforward/recurrent LIF | 정확도는 timing calibration에 크게 의존 |
| Indoor localization | Medium-high | CIR/range profile sequence | Rate + latency hybrid | ConvSNN 또는 recurrent SNN | label과 anchor가 있으면 좋음 |
| LOS/NLOS classification | High | CIR, delay spread, first-path/peak feature | Rate 또는 threshold/event | small ConvSNN 또는 MLP-SNN | 가장 좋은 첫 classification task |
| Gesture recognition | High | IR-UWB frame/range sequence | Rate, latency, event | ConvSNN/recurrent SNN | direct IR-UWB SNN precedent 존재 |
| Human activity recognition | High | range-time map 또는 CIR sequence | frame 위 rate/event | ConvSNN + recurrent SNN | adjacent radar literature가 잘 전이됨 |
| Object/radar sensing | Medium | radar-like frame/range-angle data | Rate/event | ConvSNN 또는 spiking transformer | 더 풍부한 UWB sensing setup 필요 |
| Channel state classification | High | CIR vector 또는 RF feature | Frequency/rate coding | LSM 또는 MLP-SNN | direct UWB channel-estimation evidence 존재 |

## 8. Recommended Research Directions

### Direction 1 - CIR 기반 LOS/NLOS Classification with LIF ConvSNN

- UWB 데이터 형태: CIR vector 또는 range profile.
- Encoding: amplitude에는 rate coding, first-path timing에는 optional latency coding.
- SNN 구조: LIF neuron을 가진 shallow convolutional SNN.
- 핵심 참고: Paper D1, Paper A1, Paper M1, Paper M3.
- 장점: label이 명확하고 feature가 해석 가능하며 model size가 관리 가능.
- 예상 난점: dataset quality와 synchronization.
- 첫 실험: 동일한 CIR split에서 ANN baseline, rate-coded LIF MLP, ConvSNN 비교.

### Direction 2 - Radar-SNN Transfer를 이용한 IR-UWB Gesture Recognition

- UWB 데이터 형태: IR-UWB radar에서 나온 range-time map 또는 frame sequence.
- Encoding: frame 변화에 대한 rate 또는 threshold/event encoding.
- SNN 구조: ConvSNN 또는 ConvSNN + recurrent LIF layer.
- 핵심 참고: Paper D2, Paper A1, Paper A2, Paper A4.
- 장점: direct UWB gesture precedent와 public adjacent radar code 존재.
- 예상 난점: FMCW example의 preprocessing을 IR-UWB data에 맞추는 것.
- 첫 실험: UWB range-time frame으로 ConvSNN gesture pipeline 재현.

### Direction 3 - Liquid State Machine 기반 UWB Channel Estimation

- UWB 데이터 형태: RF feature vector 또는 CIR vector.
- Encoding: frequency/rate encoding.
- SNN 구조: LIF reservoir와 trainable readout을 가진 LSM.
- 핵심 참고: Paper D1, Paper M4, BindsNET/Norse.
- 장점: direct UWB channel-estimation paper 존재.
- 예상 난점: classical estimator 및 deep ANN baseline과의 benchmarking.
- 첫 실험: fixed CIR vector를 사용해 readout을 학습하고 MLP/regression baseline과 비교.

### Direction 4 - Event-based UWB Motion or Activity Detection

- UWB 데이터 형태: frame-to-frame CIR 또는 range-profile 변화.
- Encoding: temporal difference에 대한 threshold/event encoding.
- SNN 구조: recurrent LIF/ALIF 또는 small ConvSNN.
- 핵심 참고: Paper A5, Paper A8, Paper M3.
- 장점: 자연스럽게 sparse하고 low-power이며 activity sensing에 잘 맞음.
- 예상 난점: event threshold와 noise suppression.
- 첫 실험: delta-CIR spike에서 movement/no-movement 또는 activity class 탐지.

### Direction 5 - Accuracy-first UWB Sensing을 위한 Hybrid ANN-SNN

- UWB 데이터 형태: CIR/range profile/spectrogram.
- Encoding: ANN feature extractor 뒤에 SNN classifier를 붙이거나 ANN-to-SNN conversion.
- SNN 구조: hybrid CNN-SNN 또는 converted ANN.
- 핵심 참고: Paper M2, SpikingJelly, snnTorch.
- 장점: 초기 accuracy가 더 좋고 debugging이 쉬움.
- 예상 난점: biologically/event-driven purity는 낮아질 수 있고 energy advantage가 줄 수 있음.
- 첫 실험: UWB representation 위 CNN을 학습한 뒤 convert하거나 classifier head를 SNN으로 교체.

## 9. Reading Priority List

1. Zhang, He, Meng - Exploring the Potential of Spiking Neural Networks in UWB Channel Estimation. Direct UWB + SNN이며 channel/CIR processing에 가장 가까움.
2. Hand Gesture Recognition Using IR-UWB Radar with Spiking Neural Networks. Direct IR-UWB sensing task. 가능하면 IEEE full text 확보.
3. Tsang/Safa radar gesture SNN 논문들과 ConvSNN code. radar에서 UWB sensing으로 넘어가는 가장 실용적인 implementation bridge.
4. Auge et al. - Review of Encoding Techniques for SNNs. spike encoding choice를 정당화하는 데 필요.
5. Neftci, Mostafa, Zenke - Surrogate Gradient Learning in SNNs. modern supervised SNN training에 필요.
6. O'Halloran/McGinley UWB medical radar SNN 논문들. historical direct UWB precedent로 유용.
7. NeuroRadar와 SNNs for Radar concepts/challenges. low-power/event-driven motivation에 유용.
8. SpikingRTNH/K-Radar. advanced radar tensor architecture idea를 위해 나중에 읽기.

## 10. References

### Direct UWB References

- Youdong Zhang, Xu He, Xiaolin Meng, "Exploring the Potential of Spiking Neural Networks in UWB Channel Estimation", arXiv, 2025. https://arxiv.org/abs/2512.23975
- Shule Wang, Yulong Yan, Haoming Chu, Guangxi Hu, Zhi Zhang, Zhuo Zou, Lirong Zheng, "Hand Gesture Recognition Using IR-UWB Radar with Spiking Neural Networks", IEEE AICAS 2022. https://ieeexplore.ieee.org/document/9870013
- Martin O'Halloran, Brendan McGinley, Elfed Lewis, Martin Glavin, Edward Jones, "Spiking Neural Networks for Breast Cancer Classification in a Dielectrically Heterogeneous Breast", Progress In Electromagnetics Research, 2011. https://www.jpier.org/issues/volume.html?paper=11071904
- Brendan McGinley, Martin O'Halloran, Elfed Lewis, Martin Glavin, Edward Jones, "Evolving Spiking Neural Networks for Breast Cancer Classification in a Dielectrically Heterogeneous Breast", Progress In Electromagnetics Research Letters, 2011. https://www.jpier.org/issues/volume.html?paper=11100304
- Irene Krewer et al., "An Embedded Hardware Spiking Neural Network Targeted for Unsupervised UWB Radar-Based Bladder Volume Monitoring", 2013. https://pure.ulster.ac.uk/en/publications/an-embedded-hardware-spiking-neural-network-targeted-for-unsupervi

### Adjacent Radar/RF References

- S. H. Tsang et al., "Radar-Based Hand Gesture Recognition Using Spiking Neural Networks", IEEE Sensors Journal, 2021. https://ieeexplore.ieee.org/document/9420974
- ConvSNN radar gesture repository. https://github.com/SoftwareImpacts/SIMPAC-2021-111
- Ali Safa, Andre Bourdoux, Ilja Ocket, Francky Catthoor, Georges G. E. Gielen, "A 2-uJ, 12-class, 91% Accuracy Spiking Neural Network Approach For Radar Gesture Recognition", arXiv, 2021. https://arxiv.org/abs/2108.02669
- M. Arsalan et al., "Resource Efficient Gesture Sensing Based on FMCW Radar using Spiking Neural Networks", IEEE IMS, 2021. https://ieeexplore.ieee.org/document/9574994
- B. Vogginger et al., "Automotive Radar Processing With Spiking Neural Networks: Concepts and Challenges", Frontiers in Neuroscience, 2022. https://www.frontiersin.org/articles/10.3389/fnins.2022.851774
- Dong-Hee Paek, Seung-Hyun Kong, "SpikingRTNH: Spiking Neural Network for 4D Radar Object Detection", arXiv / IEEE IV, 2025. https://arxiv.org/abs/2502.00074
- S. Lopez-Randulfe et al., "Spiking Neural Network for Fourier Transform and Object Detection for Automotive Radar", Frontiers in Neurorobotics, 2021. https://www.frontiersin.org/articles/10.3389/fnbot.2021.688344/full
- Kai Zheng, Kun Qian, Timothy Woodford, Xinyu Zhang, "NeuroRadar: A Neuromorphic Radar Sensor for Low-Power IoT Systems", ACM SenSys 2023. https://dl.acm.org/doi/10.1145/3625687.3625795
- K-Radar / SpikingRTNH repository. https://github.com/kaist-avelab/K-Radar
- Y. Luo et al., "Spiking Neural Networks for Radar Emitter Recognition", Remote Sensing, 2024. https://www.mdpi.com/2072-4292/16/14/2680
- Q. Jiang, J. Sha, "RF Fingerprinting Identification Based on Spiking Neural Network for LEO-MIMO Systems", IEEE Wireless Communications Letters, 2023. https://doi.org/10.1109/LWC.2022.3223939

### SNN Methodology References

- E. O. Neftci, H. Mostafa, F. Zenke, "Surrogate Gradient Learning in Spiking Neural Networks", IEEE Signal Processing Magazine / arXiv, 2019. https://arxiv.org/abs/1901.09948
- A. Tavanaei et al., "Deep Learning in Spiking Neural Networks", Neural Networks, 2019. https://www.sciencedirect.com/science/article/pii/S0893608018303332
- D. Auge, J. Hille, E. Mueller, A. Knoll, "A Survey of Encoding Techniques for Signal Processing in Spiking Neural Networks", Neural Processing Letters / related indexing, 2021. https://www.sciencedirect.com/science/article/pii/S0925231221009722
- A. Taherkhani et al., "A Review of Learning in Biologically Plausible Spiking Neural Networks", Neural Networks, 2020. https://www.sciencedirect.com/science/article/pii/S0893608020303573

### Open-source Framework References

- snnTorch. https://github.com/jeshraghian/snntorch
- Norse. https://github.com/norse/norse
- SpikingJelly. https://github.com/fangwei123456/spikingjelly
- Lava. https://github.com/lava-nc/lava
- BindsNET. https://github.com/BindsNET/bindsnet
- Rockpool. https://github.com/synsense/rockpool
- Tonic. https://github.com/neuromorphs/tonic
- Brian2. https://github.com/brian-team/brian2

## 11. Requirement Traceability Checklist

| 요구사항 | 이 파일에서의 근거 |
|---|---|
| SNN methodology | Section 2 |
| UWB data representations | Section 3 |
| Spike encoding comparison | Section 4 |
| UWB direct papers | Section 5.1 and References |
| Radar/RF/wireless adjacent papers | Section 5.2 and References |
| General SNN methodology papers | Section 5.3 |
| Open-source/project survey | Section 6 |
| Task-by-task applicability | Section 7 |
| 3-5 recommended research directions | Section 8 |
| Reading priority list | Section 9 |
| References with links | Section 10 |
| Clear direct vs adjacent distinction | Sections 5.1 and 5.2 |
| No code implementation | This file is research-only |


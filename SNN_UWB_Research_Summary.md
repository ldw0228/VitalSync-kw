# SNN + UWB Research Summary

Last updated: 2026-05-12

## 1. Executive Summary

This note surveys how Spiking Neural Networks (SNNs) can be used for Ultra-Wideband (UWB) data processing. The scope is research and methodology only: no implementation is included.

Key conclusions:

- Direct SNN + UWB work exists, but it is sparse. The most relevant direct examples are UWB channel estimation, IR-UWB gesture recognition, and UWB radar medical classification.
- The strongest adjacent evidence comes from FMCW/mmWave/automotive radar and RF signal processing, where SNNs are used for gesture recognition, object detection, spectrum sensing, radar emitter recognition, and low-power signal pipelines.
- For UWB CIR/range-profile data, the most practical first approach is not raw waveform spiking. It is feature-first encoding: normalize CIR or range bins, then use rate, latency, or threshold/event encoding.
- For low-power or real-time sensing, SNNs are attractive because sparse spikes can reduce activity. For best accuracy in early experiments, surrogate-gradient SNNs or hybrid CNN-SNN models are more practical than pure STDP.
- Recommended first research path: LOS/NLOS classification or gesture/activity classification from CIR/range-profile sequences using a small LIF-based convolutional or recurrent SNN.

Direct UWB evidence and adjacent evidence are separated below. When a claim is an inference for UWB, it is labeled as such.

## 2. SNN Background and Methodologies

### 2.1 Neuron Models

| Model | Main idea | Strengths | Limitations | UWB relevance |
|---|---|---|---|---|
| IF | Integrate input until threshold, then spike | Simple, efficient | No leakage dynamics | Baseline for encoded CIR/range bins |
| LIF | Integrate-and-fire with membrane leak | Common, stable, hardware-friendly | Needs time constants tuned | Strong first choice for UWB sequences |
| ALIF | Adaptive threshold or adaptive current | Captures history and temporal adaptation | More parameters | Useful for multipath and temporal drift |
| Izhikevich | Biologically richer spiking dynamics | Diverse firing patterns | Heavier than LIF | Usually unnecessary for first UWB experiments |
| SRM | Spike Response Model using kernels | Flexible event response modeling | More design choices | Useful if UWB events are treated as temporal impulses |

Practical recommendation: start with LIF or ALIF. UWB data already has temporal structure from time-of-flight, multipath, and frame sequences, so a simple neuron with well-controlled time constants is easier to debug than a biologically richer model.

### 2.2 Network Families

| Family | Description | When to use for UWB |
|---|---|---|
| Feedforward SNN | Dense or MLP-like spiking layers | Small feature vectors, channel-state classification |
| Convolutional SNN | Spatial/temporal filters over bins or spectrograms | CIR, range profiles, time-frequency maps |
| Recurrent SNN | Temporal memory through recurrent state | Tracking, activity recognition, sequential localization |
| Hybrid ANN-SNN | ANN front-end or feature extractor plus spiking layers | Accuracy-first experiments, noisy UWB data |
| Liquid State Machine (LSM) | Fixed recurrent reservoir plus trainable readout | Channel estimation or time-series classification with limited labels |
| ANN-to-SNN conversion | Train ANN, convert activations to spikes | When an ANN baseline already works and latency is acceptable |

### 2.3 Learning Methods

| Method | Summary | Pros | Cons | UWB fit |
|---|---|---|---|---|
| Surrogate gradient learning | Backpropagation through non-differentiable spike functions using smooth approximations | Best practical supervised accuracy | Requires time-unrolled training | Strong default for LOS/NLOS, gesture, activity |
| STDP | Local spike-timing-dependent plasticity | Biologically plausible, unsupervised | Harder to optimize for target metrics | Useful for feature discovery or neuromorphic demos |
| ANN-to-SNN conversion | Converts trained ANN activations to firing rates | Leverages mature ANN training | Can require many time steps | Good if power/latency is secondary |
| Reservoir/LSM training | Fixed spiking reservoir, train readout | Efficient, temporal dynamics | Reservoir design matters | Promising for UWB channel estimation |
| Evolutionary topology search | Evolves SNN topology/weights | Can find compact models | Expensive and harder to reproduce | Seen in early UWB medical radar papers |

## 3. UWB Data Representations

| Representation | Description | Strengths | Risks | SNN conversion idea |
|---|---|---|---|---|
| Raw waveform | Time-domain received UWB pulse samples | Maximum information | High rate, noisy, difficult | Threshold-crossing or latency coding on peaks |
| CIR / Channel Impulse Response | Channel taps over delay | Directly captures multipath | Sensitive to synchronization and calibration | Rate/latency encoding per tap or per selected path |
| Range profile | Energy by range bin | Compact, interpretable | Loses phase/fine waveform details | Rate coding by bin energy or event coding by change |
| Time-of-flight features | First path, peak path, delay spread | Small feature set | Feature engineering may hide cues | Direct current injection to LIF/MLP-SNN |
| Spectrogram / time-frequency map | Frequency content over time | Works with CNN-like models | May be overkill for short pulses | Convolutional SNN over 2D map |
| Radar-like frame sequence | Range-Doppler, range-angle, or frame stack | Matches adjacent radar SNN literature | UWB may not always have Doppler/angle | ConvSNN or recurrent SNN over frame sequence |
| Event/spike representation | Events produced from amplitude changes or threshold crossings | Sparse, neuromorphic-friendly | Encoding design strongly affects results | Delta modulation, threshold crossing, TTFS |

Practical first choice: CIR or range profile. They preserve UWB-specific time-of-flight/multipath structure while avoiding the burden of modeling the full raw waveform.

## 4. Spike Encoding Methods for UWB-like Signals

| Encoding | How it works | Suitable UWB input | Advantages | Cautions |
|---|---|---|---|---|
| Rate coding | Larger value produces more spikes over a window | CIR amplitude, range energy, spectrogram magnitude | Simple, robust, widely supported | More spikes and longer latency |
| Temporal coding | Information encoded in spike timing pattern | CIR peaks, waveform peaks | Preserves timing structure | Sensitive to synchronization |
| Latency coding | Stronger value spikes earlier | Range bins, CIR taps | Efficient and interpretable | Needs normalization; weak features may vanish |
| Time-to-first-spike | Decision uses first spike timing | First path / strongest path features | Low latency | Can discard later multipath information |
| Population coding | Multiple neurons encode one variable | ToF, range, angle, channel statistics | Smooth representation | More neurons |
| Threshold/event encoding | Spike when signal crosses threshold or changes enough | Raw waveform, CIR changes over frames | Sparse, event-like | Threshold tuning is critical |
| Frequency/rate bin encoding | Map numeric features to firing frequencies | Channel-estimation feature vectors | Used in UWB channel-estimation work | May lose precise delay ordering |

For UWB waveform/CIR/range profile:

- Raw waveform: threshold/event encoding or temporal coding.
- CIR: latency coding for path delays, rate coding for tap strength, event coding for frame-to-frame changes.
- Range profile: rate or latency coding over range bins.
- Time-series UWB sensing: recurrent SNN or ConvSNN over a sequence of encoded frames.

## 5. Paper Survey

### 5.1 Direct UWB + SNN Papers

#### Paper D1 - Exploring the Potential of Spiking Neural Networks in UWB Channel Estimation

- Authors: Youdong Zhang, Xu He, Xiaolin Meng
- Year: 2025
- Venue/source: arXiv
- Original: https://arxiv.org/abs/2512.23975
- PDF: https://arxiv.org/pdf/2512.23975
- Category: Direct UWB
- Research problem: UWB channel estimation with SNNs.
- Data/signal: UWB channel data from eWINE/UWB-related channel models. The paper uses RF/channel features and channel impulse response information.
- Input representation: A 10-dimensional RF feature vector in one experiment and CIR vectors with 50 or 120 dimensions in another.
- Spike encoding: Rate/frequency-style encoding is used to convert channel features into spike trains.
- SNN structure: Liquid State Machine with 400 or 500 LIF neurons, plus a spiking self-organizing map component.
- Learning method: STDP-like unsupervised spiking learning with readout/evaluation for channel estimation tasks.
- Evaluation/result: Reported estimation performance is around 80 percent in the described settings.
- Limitations: The paper is recent and appears mainly exploratory; it does not yet establish a broad benchmark against mature non-spiking UWB estimators.
- UWB significance: Strongest directly relevant source for treating UWB channel features and CIR vectors as SNN inputs.

#### Paper D2 - Hand Gesture Recognition Using IR-UWB Radar with Spiking Neural Networks

- Authors: Shule Wang, Yulong Yan, Haoming Chu, Guangxi Hu, Zhi Zhang, Zhuo Zou, Lirong Zheng
- Year: 2022
- Venue/source: IEEE International Conference on Artificial Intelligence Circuits and Systems (AICAS)
- DOI: 10.1109/AICAS54282.2022.9870013
- Original/DOI page: https://ieeexplore.ieee.org/document/9870013
- Proceedings index: https://researchr.org/publication/aicas-2022
- Category: Direct UWB
- Research problem: Hand gesture recognition using impulse-radio UWB radar and SNNs.
- Data/signal: IR-UWB radar gesture signals.
- Input representation: Open metadata describes IR-UWB gesture recognition; full paper is needed to verify exact tensor/feature form.
- Spike encoding: Not fully visible from open metadata.
- SNN structure: Not fully visible from open metadata.
- Learning method: Not fully visible from open metadata.
- Evaluation/result: Open metadata states the method demonstrates higher gesture-recognition accuracy than the compared deep-learning baseline, but exact metrics need the full IEEE text.
- Limitations: IEEE access may be needed for full methods and numerical results.
- UWB significance: Directly aligned with UWB sensing. This should be a high-priority full-text read because it connects IR-UWB radar and SNNs for a concrete classification task.

#### Paper D3 - Spiking Neural Networks for Breast Cancer Classification in a Dielectrically Heterogeneous Breast

- Authors: Martin O'Halloran, Brendan McGinley, Elfed Lewis, Martin Glavin, Edward Jones
- Year: 2011
- Venue/source: Progress In Electromagnetics Research
- Original/PDF: https://www.jpier.org/issues/volume.html?paper=11071904
- Category: Direct UWB radar medical sensing
- Research problem: Breast cancer classification from UWB radar-derived target signatures.
- Data/signal: UWB radar target signatures from dielectrically heterogeneous breast models.
- Input representation: Radar target-signature features.
- Spike encoding: The paper uses SNN classifiers; exact encoding details should be checked in the PDF before reproduction.
- SNN structure: Fixed-topology SNN classifier.
- Learning method: Supervised SNN classification approach.
- Evaluation/result: Demonstrates feasibility of SNN-based classification for UWB radar medical signatures.
- Limitations: Medical simulation task, older SNN methodology, not directly transferable to indoor UWB sensing without adaptation.
- UWB significance: Important early evidence that UWB radar signatures can be mapped into SNN classifiers.

#### Paper D4 - Evolving Spiking Neural Networks for Breast Cancer Classification in a Dielectrically Heterogeneous Breast

- Authors: Brendan McGinley, Martin O'Halloran, Elfed Lewis, Martin Glavin, Edward Jones
- Year: 2011
- Venue/source: Progress In Electromagnetics Research Letters
- Original/PDF: https://www.jpier.org/issues/volume.html?paper=11100304
- Category: Direct UWB radar medical sensing
- Research problem: Improve UWB radar breast-cancer classification using evolved SNN topology.
- Data/signal: UWB radar target signatures from heterogeneous breast simulations.
- Input representation: Target-signature features.
- Spike encoding: SNN-based feature-to-spike classification; confirm exact encoding in full text.
- SNN structure: Evolved-topology SNN.
- Learning method: Evolutionary topology/parameter search.
- Evaluation/result: Demonstrates a compact alternative to fixed-topology SNNs.
- Limitations: Older medical radar setting; likely small/simulation-based compared with modern datasets.
- UWB significance: Shows that topology search can be considered if a hand-designed LIF network is too brittle.

#### Paper D5 - An Embedded Hardware Spiking Neural Network Targeted for Unsupervised UWB Radar-Based Bladder Volume Monitoring

- Authors: Irene Krewer, Damien Coyle, Barry McGinley, Martin O'Halloran, Martin Glavin, Edward Jones
- Year: 2013
- Venue/source: IEEE BioCAS poster/proceedings record
- Original/record: https://pure.ulster.ac.uk/en/publications/an-embedded-hardware-spiking-neural-network-targeted-for-unsupervi
- Category: Direct UWB radar medical sensing
- Research problem: Embedded unsupervised SNN for UWB radar-based bladder volume monitoring.
- Data/signal: UWB radar measurements related to bladder volume.
- Input representation: Radar-derived features.
- Spike encoding/SNN/learning: Unsupervised embedded SNN; full technical details require the source paper/poster.
- Evaluation/result: The source identifies the method and target application, but open metadata is limited.
- Limitations: Hard to assess reproducibility without the full artifact.
- UWB significance: Useful as a low-power embedded UWB-SNN application precedent.

### 5.2 Adjacent Radar, RF, and Wireless SNN Papers

#### Paper A1 - Radar-Based Hand Gesture Recognition Using Spiking Neural Networks

- Authors: S. H. Tsang et al.
- Year: 2021
- Venue/source: IEEE Sensors Journal
- Original/DOI page: https://ieeexplore.ieee.org/document/9420974
- Related open-source code: https://github.com/SoftwareImpacts/SIMPAC-2021-111
- Category: Adjacent radar sensing
- Research problem: Radar hand-gesture classification with SNNs.
- Data/signal: FMCW radar gesture data, commonly represented as time/range/range-Doppler-like tensors.
- Input representation: Preprocessed radar frames.
- Spike encoding: Implemented in a convolutional SNN pipeline; check paper for exact encoding.
- SNN structure: Convolutional SNN.
- Learning method: Supervised learning.
- Evaluation/result: Demonstrates practical radar gesture recognition with SNNs and public code.
- Limitations: FMCW radar is not UWB, so representation transfer must be adapted.
- UWB significance: Best open implementation reference for UWB gesture/activity classification architecture.

#### Paper A2 - Improving the Accuracy of Spiking Neural Networks for Radar Gesture Recognition through Preprocessing

- Authors: B. Safa et al.
- Year: 2021/2022
- Venue/source: IEEE-related publication; code/paper often referenced from the ConvSNN repository.
- Original/source hub: https://github.com/SoftwareImpacts/SIMPAC-2021-111
- Category: Adjacent radar sensing
- Research problem: Improve radar SNN gesture recognition with signal preprocessing.
- Data/signal: Radar gesture data.
- Input representation: Preprocessed radar features.
- Spike encoding: ConvSNN pipeline.
- SNN structure: Convolutional SNN.
- Learning method: Supervised SNN training.
- Evaluation/result: Shows preprocessing can matter as much as model choice.
- Limitations: Need to verify exact publication record and metrics before formal citation.
- UWB significance: Important warning: UWB preprocessing and normalization may dominate SNN performance.

#### Paper A3 - A 2-uJ, 12-Class, 91% Accuracy Spiking Neural Network Approach for Radar Gesture Recognition

- Authors: Ali Safa, Andre Bourdoux, Ilja Ocket, Francky Catthoor, Georges G. E. Gielen
- Year: 2021
- Venue/source: arXiv / Electronics-related radar SNN work
- Original: https://arxiv.org/abs/2108.02669
- Category: Adjacent low-power radar sensing
- Research problem: Low-energy radar gesture recognition with SNNs.
- Data/signal: Radar gesture data.
- Input representation: Radar feature frames.
- Spike encoding: SNN spike-based inference.
- SNN structure: Compact SNN.
- Learning method: Supervised training.
- Evaluation/result: Reports more than 91 percent accuracy on two radar gesture datasets and an estimated 2 microjoules per classification.
- Limitations: Verify hardware assumptions before using the energy number as a deployment claim.
- UWB significance: Supports the low-power motivation for SNN + UWB embedded sensing.

#### Paper A4 - Resource-Efficient Gesture Sensing Based on FMCW Radar Using Spiking Neural Networks

- Authors: M. Arsalan et al.
- Year: 2021
- Venue/source: IEEE MTT-S International Microwave Symposium (IMS)
- Original: https://ieeexplore.ieee.org/document/9574994
- Category: Adjacent radar sensing
- Research problem: Resource-efficient radar gesture recognition with SNNs.
- Data/signal: FMCW radar intermediate-frequency or radar-derived signals.
- Input representation: Radar signal representations suitable for SNN processing.
- Spike encoding: Spike-based temporal processing.
- SNN structure: SNN gesture classifier.
- Learning method: Supervised SNN approach.
- Evaluation/result: Demonstrates energy/resource-oriented radar gesture sensing from FMCW radar data.
- Limitations: FMCW assumptions differ from impulse UWB.
- UWB significance: Useful for designing UWB edge-sensing experiments where compute and power matter.

#### Paper A5 - Automotive Radar Processing With Spiking Neural Networks: Concepts and Challenges

- Authors: Bernhard Vogginger, Felix Kreutz, Javier Lopez-Randulfe, Chen Liu, Robin Dietrich, Hector A. Gonzalez, Daniel Scholz, Nico Reeb, Daniel Auge, Julian Hille, Muhammad Arsalan, Florian Mirus, Cyprian Grassmann, Alois Knoll, Christian Mayr, et al.
- Year: 2022
- Venue/source: Frontiers in Neuroscience
- Original: https://www.frontiersin.org/articles/10.3389/fnins.2022.851774
- Category: Adjacent radar methodology
- Research problem: Survey/discuss SNN use in radar processing.
- Data/signal: Radar pipelines broadly, including signal processing and perception tasks.
- Input representation: Multiple radar representations are discussed.
- Spike encoding: Discusses the challenge of mapping radar signals into spikes.
- SNN structure: General radar SNN concepts.
- Learning method: General.
- Evaluation/result: Conceptual and methodological rather than a single benchmark.
- Limitations: Not UWB-specific.
- UWB significance: Good background for understanding what makes radar-to-spike conversion hard.

#### Paper A6 - SpikingRTNH: Spiking Residual Transformer with Neural Heterogeneity for 4D Radar Object Detection

- Authors: Dong-Hee Paek, Seung-Hyun Kong
- Year: 2025
- Venue/source: arXiv / IEEE Intelligent Vehicles Symposium (IV) project listing
- Original: https://arxiv.org/abs/2502.00074
- Project/code: https://github.com/kaist-avelab/K-Radar
- Category: Adjacent 4D radar object detection
- Research problem: 4D radar object detection using a spiking transformer-style architecture.
- Data/signal: 4D radar tensor data.
- Input representation: Radar cube/tensor representation.
- Spike encoding: SNN-compatible radar tensor processing.
- SNN structure: LIF-based spiking replacement of the RTNH-style 4D radar detector, with biological top-down inference (BTI).
- Learning method: Supervised deep SNN training.
- Evaluation/result: Reports 78 percent energy reduction while maintaining comparable 3D/BEV detection performance to the ANN counterpart.
- Limitations: 4D radar detection is far from UWB CIR classification; model is much heavier than a first UWB experiment.
- UWB significance: Advanced reference for spiking architectures on high-dimensional radar tensors.

#### Paper A7 - Spiking Neural Network for Fourier Transform and Object Detection for Automotive Radar

- Authors: S. Lopez-Randulfe et al.
- Year: 2021/2022
- Venue/source: Frontiers in Neurorobotics / neuromorphic radar literature
- Original: https://www.frontiersin.org/articles/10.3389/fnbot.2021.688344/full
- Category: Adjacent radar signal processing
- Research problem: Use SNNs for Fourier-transform-like radar processing and object detection.
- Data/signal: Automotive radar signals.
- Input representation: Radar signal pipeline leading toward detection.
- Spike encoding: SNN-based signal processing components.
- SNN structure: SNN modules for transform and detection.
- Learning method: Task-specific spiking processing.
- Evaluation/result: Demonstrates SNNs can be used earlier in the radar processing chain, not only as final classifiers.
- Limitations: Automotive radar pipeline differs from UWB time-of-flight sensing.
- UWB significance: Suggests possible future work: spike-domain preprocessing for CIR/range extraction.

#### Paper A8 - NeuroRadar: A Neuromorphic Radar Sensor for Low-Power IoT Systems

- Authors: Kai Zheng, Kun Qian, Timothy Woodford, Xinyu Zhang
- Year: 2023; CACM research-highlight version appeared in 2025
- Venue/source: ACM SenSys / Communications of the ACM highlight
- Original/DOI page: https://dl.acm.org/doi/10.1145/3625687.3625795
- Open PDF: https://xyzhang.ucsd.edu/papers/Kai.Zheng_SenSys23_NeuroRadar.pdf
- Category: Adjacent neuromorphic radar hardware
- Research problem: Neuromorphic radar sensing for low-power IoT.
- Data/signal: Radar sensor data.
- Input representation: Event/neuromorphic radar representation.
- Spike encoding: Sensor-level neuromorphic/event processing.
- SNN structure: Neuromorphic processing pipeline.
- Learning method: System-level design rather than only model training.
- Evaluation/result: Reports gesture recognition and localization case studies, with much lower power than conventional radar pipelines.
- Limitations: Hardware/system assumptions may not transfer directly to commodity UWB.
- UWB significance: Useful for the long-term vision of event-driven UWB sensors.

#### Paper A9 - Spiking Neural Networks for Radar Emitter Recognition

- Authors: Y. Luo et al.
- Year: 2024
- Venue/source: MDPI Remote Sensing
- Original: https://www.mdpi.com/2072-4292/16/14/2680
- Category: Adjacent RF/radar classification
- Research problem: Classify radar emitters with SNNs.
- Data/signal: Radar emitter signal features.
- Input representation: Signal features or time-series representation for classification.
- Spike encoding: SNN-compatible encoding of radar emitter data.
- SNN structure: SNN classifier.
- Learning method: Supervised learning.
- Evaluation/result: Demonstrates SNN classification for non-image radar/RF signals.
- Limitations: Emitter recognition differs from UWB localization/sensing.
- UWB significance: Useful if UWB task is signal/channel state classification rather than geometric localization.

#### Paper A10 - RF Fingerprinting Identification Based on Spiking Neural Network for LEO-MIMO Systems

- Authors: Q. Jiang, J. Sha
- Year: 2023
- Venue/source: IEEE Wireless Communications Letters
- DOI/record: https://doi.org/10.1109/LWC.2022.3223939
- Category: Adjacent RF/wireless signal processing
- Research problem: Energy-efficient RF fingerprint identification for LEO-MIMO communication systems.
- Data/signal: OFDM/RF fingerprinting features.
- Input representation: Channel-independent RF fingerprint features with augmentation.
- Spike encoding: SNN-compatible RF feature encoding.
- SNN structure: SNN classifier for RF fingerprint identification.
- Learning method: Supervised learning.
- Evaluation/result: Reports up to 95.26 percent identification accuracy at 25 dB SNR and 63.3 percent power reduction compared with comparable models on FPGA.
- Limitations: Wireless device identification is not UWB ranging or radar sensing.
- UWB significance: Strong adjacent evidence for low-power RF feature classification using SNNs.

### 5.3 General SNN Methodology Papers to Read

#### Paper M1 - Surrogate Gradient Learning in Spiking Neural Networks

- Authors: E. O. Neftci, H. Mostafa, F. Zenke
- Year: 2019
- Original: https://arxiv.org/abs/1901.09948
- Role: Core training-method reference for modern supervised SNNs.
- UWB significance: Use this to justify surrogate gradients for CIR/range-profile classification.

#### Paper M2 - Deep Learning in Spiking Neural Networks

- Authors: A. Tavanaei, M. Ghodrati, S. R. Kheradpisheh, T. Masquelier, A. Maida
- Year: 2019
- Original: https://www.sciencedirect.com/science/article/pii/S0893608018303332
- Role: Broad overview of deep SNN architectures and learning.
- UWB significance: Good background for choosing between conversion, STDP, and surrogate learning.

#### Paper M3 - A Review of Encoding Techniques for Spiking Neural Networks

- Authors: D. Auge, J. Hille, E. Mueller, A. Knoll
- Year: 2021
- Original: https://www.sciencedirect.com/science/article/pii/S0925231221009722
- Role: Encoding-method reference.
- UWB significance: Use to justify rate, latency, temporal, and population coding choices.

#### Paper M4 - A Review of Learning in Biologically Plausible Spiking Neural Networks

- Authors: A. Taherkhani et al.
- Year: 2020
- Original: https://www.sciencedirect.com/science/article/pii/S0893608020303573
- Role: Review of STDP and biologically plausible learning.
- UWB significance: Useful if the research direction emphasizes unsupervised or neuromorphic learning.

## 6. Open-source and Project Survey

| Project | Link | Related paper/docs | Framework | Signal/data type | SNN/encoding support | Maintenance/usefulness | UWB significance |
|---|---|---|---|---|---|---|---|
| ConvSNN radar gesture code | https://github.com/SoftwareImpacts/SIMPAC-2021-111 | Radar gesture SNN papers by Tsang/Safa et al. | Python/PyTorch-like research code | Radar gesture data | Convolutional SNN for radar gestures | Valuable because it is task-adjacent and public | Best concrete starting reference for UWB gesture/activity recognition |
| K-Radar / SpikingRTNH | https://github.com/kaist-avelab/K-Radar | K-Radar and SpikingRTNH papers | PyTorch ecosystem | 4D radar tensors | Spiking residual transformer-style radar detection | Large, active research repo | Advanced reference for high-dimensional radar SNNs; too heavy for MVP |
| snnTorch | https://github.com/jeshraghian/snntorch | Docs: https://snntorch.readthedocs.io/ | PyTorch | General SNN | LIF, surrogate gradients, tutorials | Strong practical library | Recommended for first UWB classification experiments |
| Norse | https://github.com/norse/norse | Docs: https://norse.github.io/norse/ | PyTorch | General SNN | LIF/LSNN modules, surrogate training | Mature research library | Good for recurrent/temporal SNNs on UWB sequences |
| SpikingJelly | https://github.com/fangwei123456/spikingjelly | Docs: https://spikingjelly.readthedocs.io/ | PyTorch | General SNN | ANN-to-SNN, surrogate training, encoders | Broad feature set | Useful if comparing direct SNN training and ANN-to-SNN conversion |
| Lava | https://github.com/lava-nc/lava | Docs: https://lava-nc.org/ | Intel neuromorphic software stack | Neuromorphic algorithms and deployment | Process-based SNN modeling, Loihi-oriented workflows | Strong for neuromorphic deployment | Useful after model concept is stable and hardware mapping matters |
| BindsNET | https://github.com/BindsNET/bindsnet | Paper/docs in repo | PyTorch | General SNN, STDP experiments | STDP and biologically inspired models | Older but useful for STDP prototypes | Good if exploring unsupervised UWB feature learning |
| Rockpool | https://github.com/synsense/rockpool | Docs: https://rockpool.ai/ | Python/JAX/Torch ecosystem | Neuromorphic SNN and audio/event data | Spiking layers and deployment-oriented tooling | Useful for edge/neuromorphic workflows | Later-stage tool for efficient deployment ideas |
| Tonic | https://github.com/neuromorphs/tonic | Docs: https://tonic.readthedocs.io/ | Python | Event-based datasets | Dataset transforms, event handling | Useful support library | Helpful if UWB is converted into event streams |
| Brian2 | https://github.com/brian-team/brian2 | Docs: https://brian2.readthedocs.io/ | Python simulation | Biophysical/neural simulation | Flexible neuron modeling | Excellent for scientific simulation, less for deep learning | Useful for testing custom UWB spike encodings or neuron dynamics |

Open-source caution:

- Most SNN libraries are general-purpose; few include UWB-specific loaders.
- Radar SNN repositories are more useful for architecture and preprocessing patterns than for direct code reuse.
- For this project, snnTorch or Norse is likely the fastest path for reproducible experiments; Lava/Rockpool become relevant when neuromorphic hardware deployment is part of the goal.

## 7. Task-by-task Applicability

| UWB task | SNN suitability | Suggested input | Suggested encoding | Suggested model | Notes |
|---|---|---|---|---|---|
| Ranging | Medium | CIR peaks, first path, ToF features | Latency or TTFS | Small feedforward/recurrent LIF | Accuracy depends heavily on timing calibration |
| Indoor localization | Medium-high | CIR/range profile sequence | Rate + latency hybrid | ConvSNN or recurrent SNN | Good if labels and anchors are available |
| LOS/NLOS classification | High | CIR, delay spread, first-path/peak features | Rate or threshold/event | Small ConvSNN or MLP-SNN | Best first classification task |
| Gesture recognition | High | IR-UWB frame/range sequence | Rate, latency, event | ConvSNN/recurrent SNN | Direct IR-UWB SNN precedent exists |
| Human activity recognition | High | Range-time map or CIR sequence | Rate/event over frames | ConvSNN + recurrent SNN | Adjacent radar literature transfers well |
| Object/radar sensing | Medium | Radar-like frame/range-angle data | Rate/event | ConvSNN or spiking transformer | Needs richer UWB sensing setup |
| Channel state classification | High | CIR vectors or RF features | Frequency/rate coding | LSM or MLP-SNN | Direct UWB channel-estimation evidence exists |

## 8. Recommended Research Directions

### Direction 1 - LOS/NLOS Classification from CIR with LIF ConvSNN

- UWB data form: CIR vectors or range profiles.
- Encoding: Rate coding for amplitude plus optional latency coding for first-path timing.
- SNN structure: Shallow convolutional SNN with LIF neurons.
- Key references: Paper D1, Paper A1, Paper M1, Paper M3.
- Advantages: Clear labels, interpretable features, manageable model size.
- Expected difficulty: Dataset quality and synchronization.
- First experiment: Compare ANN baseline, rate-coded LIF MLP, and ConvSNN on the same CIR split.

### Direction 2 - IR-UWB Gesture Recognition with Radar-SNN Transfer

- UWB data form: Range-time maps or frame sequences from IR-UWB radar.
- Encoding: Rate or threshold/event encoding over frame changes.
- SNN structure: ConvSNN or ConvSNN plus recurrent LIF layer.
- Key references: Paper D2, Paper A1, Paper A2, Paper A4.
- Advantages: Direct UWB gesture precedent and public adjacent radar code.
- Expected difficulty: Matching preprocessing from FMCW examples to IR-UWB data.
- First experiment: Recreate a ConvSNN gesture pipeline with UWB range-time frames.

### Direction 3 - UWB Channel Estimation with Liquid State Machine

- UWB data form: RF feature vectors or CIR vectors.
- Encoding: Frequency/rate encoding.
- SNN structure: LSM with LIF reservoir and trainable readout.
- Key references: Paper D1, Paper M4, BindsNET/Norse.
- Advantages: Direct UWB channel-estimation paper exists.
- Expected difficulty: Benchmarking against classical estimators and deep ANN baselines.
- First experiment: Use fixed CIR vectors, train readout, compare against MLP/regression baseline.

### Direction 4 - Event-based UWB Motion or Activity Detection

- UWB data form: Frame-to-frame CIR or range-profile changes.
- Encoding: Threshold/event encoding on temporal differences.
- SNN structure: Recurrent LIF/ALIF or small ConvSNN.
- Key references: Paper A5, Paper A8, Paper M3.
- Advantages: Naturally sparse and low-power; good fit for activity sensing.
- Expected difficulty: Event thresholds and noise suppression.
- First experiment: Detect movement/no-movement or activity classes from delta-CIR spikes.

### Direction 5 - Hybrid ANN-SNN for Accuracy-first UWB Sensing

- UWB data form: CIR/range profile/spectrogram.
- Encoding: ANN feature extractor followed by SNN classifier, or ANN-to-SNN conversion.
- SNN structure: Hybrid CNN-SNN or converted ANN.
- Key references: Paper M2, SpikingJelly, snnTorch.
- Advantages: Better initial accuracy and easier debugging.
- Expected difficulty: Less biologically/event-driven purity; may reduce energy advantage.
- First experiment: Train CNN on UWB representation, then convert or replace classifier head with SNN.

## 9. Reading Priority List

1. Zhang, He, Meng - Exploring the Potential of Spiking Neural Networks in UWB Channel Estimation. Direct UWB + SNN and closest to channel/CIR processing.
2. Hand Gesture Recognition Using IR-UWB Radar with Spiking Neural Networks. Direct IR-UWB sensing task; obtain full IEEE text if possible.
3. Tsang/Safa radar gesture SNN papers plus ConvSNN code. Best practical implementation bridge from radar to UWB sensing.
4. Auge et al. - Review of Encoding Techniques for SNNs. Needed to justify spike encoding choices.
5. Neftci, Mostafa, Zenke - Surrogate Gradient Learning in SNNs. Needed for modern supervised SNN training.
6. O'Halloran/McGinley UWB medical radar SNN papers. Useful historical direct UWB precedents.
7. NeuroRadar and SNNs for Radar concepts/challenges. Useful for low-power/event-driven motivation.
8. SpikingRTNH/K-Radar. Read later for advanced radar tensor architecture ideas.

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

| Requirement | Evidence in this file |
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

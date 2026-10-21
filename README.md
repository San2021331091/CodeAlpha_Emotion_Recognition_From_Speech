# VocalSense AI — Speech Emotion Recognition

[![CodeAlpha Internship](https://img.shields.io/badge/CodeAlpha-Machine%20Learning%20Internship-6366f1.svg)](https://www.codealpha.tech/)
[![Task](https://img.shields.io/badge/Task%202-Emotion%20Recognition%20from%20Speech-emerald.svg)]()
[![Stack](https://img.shields.io/badge/Stack-Next.js%2015%20%7C%20NestJS%20%7C%20FastAPI-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-yellow.svg)]()
[![Node](https://img.shields.io/badge/Node.js-v20%2B-green.svg)]()

VocalSense AI is a deep learning speech signal processing and emotion recognition suite that classifies human emotional states from spoken audio utterances. Built with **Next.js 15 (Frontend)**, **NestJS (Enterprise API Gateway)**, and **FastAPI (ML Microservice)**.

---

## 📌 Project Overview

Speech emotion recognition (SER) extracts vocal tract and prosodic acoustic features from speech signals and classifies them into the six labels shared by RAVDESS, TESS, and EMO-DB:

- 😊 **Happy** — Elevated, buoyant melodic pitch contours with energetic timbral harmonics.
- 😡 **Angry** — Aggressive vocal tract strain, sharp initial attack, and high acoustic intensity.
- 😢 **Sad** — Subdued acoustic power, low fundamental frequency ($F_0$), and descending cadence.
- 😐 **Neutral** — Balanced spectral envelope, standard conversational pitch, and even rhythm.
- 😨 **Fearful** — Elevated pitch with rapid 7Hz micro-tremor and breathless aspiration noise.
- 🤢 **Disgust** — Vocal tract constriction, guttural pitch sagging, and negative emotional valence.

RAVDESS calm/surprised, TESS pleasant-surprise, and EMO-DB boredom recordings
are excluded because those labels are not shared consistently across all three
corpora.

Predictions are also projected into **Russell's Circumplex Model of Affect**, providing continuous 2D coordinates across **Valence** (pleasure vs displeasure) and **Arousal** (activation energy).

---

## 🧠 Machine Learning Methodology

### 1. Acoustic Signal Processing & Feature Engineering
Raw 22,050 Hz audio waveforms are transformed into a **231-dimensional acoustic feature representation**:

1. **Mel-Frequency Cepstral Coefficients (MFCCs)**:
   - 40 static MFCCs (mean & standard deviation across time frames)
   - 40 first-order $\Delta$ delta coefficients (velocity)
   - 40 second-order $\Delta^2$ delta-delta coefficients (acceleration)
   - Captures vocal tract resonance envelopes via discrete cosine transform:
     $$c_n = \sum_{k=0}^{K-1} \log(S_k) \cos\left[ \frac{\pi n}{K} \left(k + \frac{1}{2}\right) \right]$$
2. **12-TET Chroma Pitch Class Profile**:
   - 12 pitch classes (C, C#, D, D#, E, F, F#, G, G#, A, A#, B) measuring harmonic content.
3. **Spectral Contrast**:
   - Octave-based valley-to-peak decibel difference across 7 frequency sub-bands.
4. **Tonnetz (Tonal Centroid Features)**:
   - 6 harmonic space coordinates representing Fifth and Minor/Major third harmonic relations.
5. **Prosodic & Temporal Metrics**:
   - **Fundamental Frequency ($F_0$)**: Extracted via harmonic autocorrelation.
   - **Zero-Crossing Rate (ZCR)**: Quantifies unvoiced noisiness and breathiness.
   - **Root-Mean-Square Energy (RMS)**: Measures signal amplitude intensity envelope.
   - **Spectral Centroid & Rolloff**: Timbral brightness and 85% spectral energy boundary.

### 2. Deep Learning & Statistical Model Architectures
- **CRNN (CNN + Bi-LSTM Hybrid — Best Model)**:
  - 1D-CNN temporal convolution layers extract local formant and spectral patterns.
  - Bidirectional LSTM processes MFCC sequences forward and backward with mean pooling.
- **1D-CNN (Temporal Convolutional Network)**:
  - Two trainable convolutional blocks with batch normalization, ReLU, max pooling, and global average pooling.
- **Bi-LSTM (Bidirectional Recurrent Network)**:
  - Recurrent gate equations ($f_t, i_t, \tilde{c}_t, c_t, o_t$) with temporal mean pooling.
- **Deep MLP (Multi-Layer Perceptron)**:
  - 3 dense hidden layers (256, 128, 64) with Adam optimizer and early stopping on 231 acoustic features.
- **Random Forest & Support Vector Classifier (SVM)**:
  - 150-tree ensemble and RBF-kernel margin maximizer as rigorous statistical baselines.

---

## 📊 Model Performance Benchmark

Training combines **RAVDESS, TESS, and EMO-DB** on their six shared labels:
neutral, happy, sad, angry, fearful, and disgust. RAVDESS-only categories (calm
and surprised), TESS pleasant surprise, and EMO-DB boredom are excluded rather
than being treated as equivalent emotions. The RAVDESS audio-only speech
archive is sourced from [Livingstone & Russo](https://zenodo.org/records/1188976)
and licensed **CC BY-NC-SA 4.0**. TESS is available from the
[Borealis repository](https://doi.org/10.5683/SP2/E8H2MF) under **CC BY-NC 4.0**.
EMO-DB is available from its [official provider](http://emodb.bilderbar.info/)
for analysis with proper attribution. The datasets are stored locally for
training and are not bundled or redistributed with this project; follow each
dataset's own access, research, attribution, and redistribution terms.

Place the extracted datasets here:

- `ml-service/data/ravdess` — official RAVDESS actor folders; downloaded by the
  existing verified RAVDESS loader if missing.
- `ml-service/data/tess` — TESS folders such as `OAF_angry` and `YAF_neutral`,
  or set `TESS_DATA_DIR` to another extracted TESS folder.
- `ml-service/data/emo-db` — EMO-DB WAV files with names such as `03a01Fa.wav`,
  or set `EMODB_DATA_DIR` to another extracted EMO-DB folder.

The local TESS and EMO-DB downloads used for this workspace are in those
folders. To reproduce this setup elsewhere, download each dataset from its
authorized source and extract the files into the corresponding folder.

The loader normalizes the six shared labels and splits each corpus by speaker:
RAVDESS actors 01-16/17-20/21-24, TESS OAF/YAF into separate training/test
speakers, and EMO-DB speakers 60/20/20. TESS has only two speakers: OAF is used
for training, and separate YAF clips are used for validation and test. Training speakers
are excluded from evaluation. Training weights balance each dataset
and emotion class so TESS's larger file count does not dominate. Results report
test accuracy and macro F1 separately for each corpus, as well as overall;
cross-corpus results are essential because TESS/RAVDESS are English and EMO-DB
is German acted speech.

Model selection uses macro F1 balanced equally across datasets and emotion
classes, with balanced accuracy as the tie-breaker. Validation clips are
stratified into disjoint calibration and model-selection partitions. Class-bias
corrections are fitted on calibration clips only, while checkpoint selection
and voting weights use model-selection clips. Test recordings are used only
for reporting results.
The hybrid averages probabilities from histogram gradient boosting on acoustic
features and a Bi-LSTM on MFCC sequences. A second ensemble selects a pair of
models and their voting weight using model-selection recordings only. Every member
receives its training-time preprocessing. Model weights and preprocessing
statistics are learned from training speakers only. Model probabilities are
not calibrated confidence estimates, and acted-corpus results do not establish
accuracy on natural speech, other languages, or noisy recordings.

Run `python train_pipeline.py` from `ml-service` to train after placing TESS
and EMO-DB in their folders. The combined feature cache is
`data/multicorpus_features_v2.npz`; the legacy synthetic feature cache is not
used.

To stage a complete training run, use
`python train_pipeline.py --models-dir saved_models_balanced`, then
`python promote_model.py --source saved_models_balanced`. Promotion compares
balanced macro F1 on the model-selection partition and backs up replaced
artifacts. `balanced_training_evaluation.json` records the comparison and happy
precision/recall. The older `evaluate_ensemble.py` script supports legacy
accuracy-selected models and will not replace a model trained with the new
calibration split. Acted-speech results do not establish accuracy on everyday
microphone speech.

Silent, nearly silent, and shorter-than-0.25-second recordings are rejected
before amplitude normalization. Microphone capture requests disabled automatic
gain control, noise suppression, and echo cancellation to preserve vocal cues.

### Current deployed speech model

The default is **WavLM Speech Classifier**: a frozen Microsoft WavLM Base+
encoder with a linear SVM emotion head. It uses 16 kHz speech internally and
layer 8 mean and standard-deviation features. The head was trained on the
2,142 original training recordings, not the augmented corpus. The earlier
50,000-clip CNN experiments are retained separately.

Validation-selected held-out test accuracy is **79.55% RAVDESS** (176 clips),
**89.79% TESS** (960 clips), **83.65% EMO-DB** (104 clips), and **87.82% overall**.
Happy recall is 84.69%. Test audio was not used to select classifier settings.
The TESS validation/test speaker limitation described above still applies.
These dataset results do not guarantee recognition of every natural recording
or the mood of music. Long uploads are processed in bounded speech chunks.

```powershell
cd ml-service
python download_speech_encoder.py
python train_speech_encoder.py
python stage_speech_encoder.py
python promote_model.py --source saved_models_speech_encoder --report-file speech_encoder_promotion.json
```

Keep `.model_cache` for encoder inference and `data/wavlm_features_v2.*` for
reproducible evaluation and promotion comparisons. The model revision and
feature-cache record order are checked. `speech_encoder_promotion.json`
records the measured validation improvement before deployment.

### Expanded training with 50,000 audio clips

The three local datasets provide 3,910 eligible original recordings. The optional
balanced corpus contains **2,142 original training recordings and 47,858
augmented WAV clips**, or **50,000 total**. Four emotions have 8,333 clips and
two have 8,334; the count difference is at most one. It does not add
new speakers. The original 528 validation and 1,240 test recordings, and all
derivatives of those clips, are excluded from training augmentation.

Keep `data/training_50000` to repeat this expanded training. Standard training
and inference do not require it. Earlier balanced and regenerated corpora were
removed during cleanup; their models did not improve validation performance.

```powershell
cd ml-service
python prepare_training_audio.py --target-count 50000 --workers 4 --output-dir data/training_50000 --seed 20261022
python audit_training_audio.py --corpus-dir data/training_50000
python train_augmented.py --corpus-dir data/training_50000 --models-dir saved_models_50000 --epochs 40 --batch-size 128
python promote_model.py --source saved_models_50000 --report-file training_50000_evaluation.json
```

Preparation uses approximately 8 GB for PCM WAV files and a memory-mapped MFCC
cache in `data/training_50000`. Audio is organized as
`wav/neutral`, `wav/happy`, `wav/sad`, `wav/angry`, `wav/fearful`, and
`wav/disgust`, each containing 8,333 or 8,334 clips. Original recordings are linked or
copied into their class folders alongside newly generated variants. Obsolete
`data/training_regenerated_v2` and `data/training_balanced` corpora have been removed.
`manifest.jsonl` records the emotion, source,
speaker, original parent and random seed for each example; `metadata.json`
distinguishes original and augmented counts. Mild background noise, room echo,
microphone filtering and short boundary silence simulate recording variation
without changing pitch or speaking speed. Generator version 3 preserves the exact
requested count. It uses a new
seed, visits every source recording before reusing one from the same dataset
and emotion, and limits noise to 45–60 dB SNR, echo to 3%, microphone tilt to
3%, and boundary silence to 30 ms. The generation settings are versioned, so
existing audio from another configuration cannot silently be reused.
`quality_report.json` records header checks for every audio file, folder counts,
source-copy integrity, finite features, and training-only parent verification.
These inherited labels are not new
human annotations. Augmentation does not guarantee improved recognition.

The CNN trains on all 50,000 examples. Voting with the deployed model and
class-bias correction are compared on the original model-selection split.
Promotion backs up replaced artifacts and requires better balanced validation
macro F1. Test scores and happy recall are reported separately, and test audio
is not used to select a candidate.

RAVDESS contributes 704 original training recordings and 16,668 examples in the
50,000-clip corpus. To increase its influence without duplicating files or moving
held-out actors into training, use a separate weighted experiment:

```powershell
python train_augmented.py --corpus-dir data/training_50000 --models-dir saved_models_ravdess_weighted --epochs 40 --batch-size 128 --ravdess-weight 2
python promote_model.py --source saved_models_ravdess_weighted --report-file ravdess_weighted_evaluation.json
```

The factor changes the training loss, not the number of distinct recordings.
Validation selection remains balanced across datasets. Model Lab reports completed
expanded-run counts separately from the deployed model's original dataset split.

The Model Lab **Train 50,000-clip CNN** button starts a background job rather
than keeping one HTTP request open for the whole run. `POST /api/train` returns
202 with a job ID; `GET /api/train/status` reports queued, running, completed,
or failed. The gateway exposes these as `/speech-emotion/retrain` and
`/speech-emotion/retrain/status`. The page polls status and displays the actual
error or validation-retention outcome. A web-service restart does not stop the
separate worker; duplicate starts reuse the active job. Job state is stored in
`training_job.json` and logs in `.logs/training-job.log`.

---

## 📂 Project Structure

```
CodeAlpha_Emotion_Recognition_From_Speech/
├── ml-service/                 # FastAPI Python ML Microservice (Port 8000)
│   ├── app/
│   │   ├── api/routes.py       # Prediction, audio streaming, metrics endpoints
│   │   ├── ml/
│   │   │   ├── feature_extractor.py  # Librosa MFCC & chroma extraction
│   │   │   ├── data_generator.py     # Source-filter emotional speech synthesizer
│   │   │   ├── deep_models.py        # CRNN, 1D-CNN, Bi-LSTM, Deep MLP models
│   │   │   ├── trainer.py            # Training pipeline & evaluation engine
│   │   │   └── predictor.py          # Real-time inference & explainability
│   │   ├── models/schemas.py   # Pydantic data schemas
│   │   └── main.py             # FastAPI entry point & CORS
│   ├── data/                   # Benchmark WAV audios & cached features
│   ├── saved_models/           # Serialized models (.joblib) & metrics.json
│   ├── train_pipeline.py       # Standalone training script
│   ├── requirements.txt
│   └── Dockerfile
│
├── backend-gateway/            # NestJS TypeScript API Gateway (Port 3001)
│   ├── src/
│   │   ├── speech-emotion/     # Audio upload proxy & ML service client
│   │   ├── emotion-history/    # Session history, ground-truth audit & CSV export
│   │   ├── analytics/          # KPI metrics & session summary statistics
│   │   ├── app.module.ts
│   │   └── main.ts
│   ├── package.json
│   ├── tsconfig.json
│   └── Dockerfile
│
├── frontend/                   # Next.js 15 + Tailwind CSS Frontend (Port 3000)
│   ├── src/
│   │   ├── app/                # App Router (layout.tsx, page.tsx, globals.css)
│   │   ├── components/         # AudioRecorder, BenchmarkSamples, EmotionResultCard, etc.
│   │   └── lib/                # Emotion constants, circumplex coordinates, utils
│   ├── package.json
│   ├── tailwind.config.ts
│   └── Dockerfile
│
├── docker-compose.yml          # Multi-container container orchestration
├── start_services.ps1          # 1-click Windows PowerShell launcher script
├── presentation_slides.html    # Interactive slide deck for LinkedIn video
└── README.md
```

---

## 🚀 Quickstart Guide

### Option 1: 1-Click Launch (Windows PowerShell)
```powershell
.\start_services.ps1
```

### Option 2: Run via Docker Compose
```bash
docker-compose up --build
```

### Option 3: Manual Step-by-Step Setup

#### Step 1: Start ML Microservice
```bash
cd ml-service
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Install the CPU build used by this project:
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python train_pipeline.py
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
*API Docs available at: `http://127.0.0.1:8000/docs`*

#### Step 2: Start NestJS Backend Gateway
```bash
cd ../backend-gateway
npm install
npm run build
npm start
```
*Gateway running at: `http://localhost:3001`*

#### Step 3: Start Next.js Frontend
```bash
cd ../frontend
npm install
npm run dev
```
*Open `http://localhost:3000` in your browser.*

---

## 🧪 Interactive Features Walkthrough

1. **Live Microphone Voice Studio**:
   - Record spoken utterances directly through your web browser with a real-time animated HTML5 canvas oscilloscope.
   - Listen to recorded audio, view recording duration, and select your preferred model architecture.
2. **Benchmark Speech Utterances**:
   - 8 real RAVDESS speech recordings representing ground-truth states (Happy, Angry, Sad, Calm, Fearful, Disgust, Surprised, Neutral).
   - In-browser playback and 1-click evaluation.
3. **Affect Circumplex (Valence-Arousal Grid)**:
   - Visual 2D coordinate plot dynamically displaying emotional arousal and valence quadrants.
4. **Acoustic Feature Extraction Spectrum**:
   - Visual inspection of 13 MFCC coefficients, 12-TET chroma pitch class energy, fundamental pitch ($F_0$), RMS intensity, and Zero-Crossing Rate.
5. **Model Performance Lab**:
   - Cross-model benchmark table comparing Accuracy, Precision, Recall, F1, and Latency.
   - Interactive 8x8 Confusion Matrix heatmap.
   - Per-emotion F1-score breakdown bars and feature importance ranking.
6. **Session Voice Log & CSV Export**:
   - Full history of analyzed utterances with human verification tagging.
   - 1-click export to `speech_emotion_history.csv`.

---

## 👨‍💻 Author & Acknowledgements
- Developed by: **Santosh Saha**
- Role: Machine Learning Intern at [CodeAlpha](https://www.codealpha.tech/)
- Task-2 — Speech Emotion Recognition
- Program: CodeAlpha Machine Learning Internship
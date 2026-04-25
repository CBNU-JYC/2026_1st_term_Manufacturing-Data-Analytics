# 7th MIMII Sound Practice

이 폴더는 MIMII 밸브 사운드 데이터로 오토인코더 기반 이상 탐지를 실습하는 코드입니다.

## 실행 순서

```bash
cd /Users/jeong-yongcheol/Desktop/00_CBNU_AI/My_project/ManDA_Lecture
python3 7th_MIMII_sound_0414/step1_eda_audio.py
python3 7th_MIMII_sound_0414/step2_train_audio_ae.py
python3 7th_MIMII_sound_0414/step3_eval_audio_ae.py
python3 7th_MIMII_sound_0414/step4_inference_audio_ae.py
```

## 주요 파일

- `step1_eda_audio.py`: waveform, spectrogram, mel-spectrogram EDA
- `step2_train_audio_ae.py`: 오토인코더 학습 및 모델 저장
- `step3_eval_audio_ae.py`: 복원 오차 분포, ROC-AUC, 최적 임계값, 재구성 비교
- `step4_inference_audio_ae.py`: 시간대별 이상 점수 추론

## 자동 저장 결과

- `visualizations/`
  - `step1_waveform.png`
  - `step1_spectrogram.png`
  - `step1_mel_spectrogram.png`
  - `step2_training_loss.png`
  - `step3_error_distribution.png`
  - `step3_roc_curve.png`
  - `step3_confusion_matrix.png`
  - `step3_reconstruction_*.png`
  - `step4_inference_scores.png`
- `results/`
  - `step4_inference_scores.csv`

## 참고

- `sounddevice`가 없으면 오디오 재생만 건너뛰고 나머지 분석은 계속 진행됩니다.
- 그래프는 화면에 표시될 뿐 아니라 자동으로 이미지 파일로도 저장됩니다.

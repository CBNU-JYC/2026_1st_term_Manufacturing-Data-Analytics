# 7th KAMP Sound Practice

이 폴더는 팬(FAN) 소리 데이터를 이용해 정상/이상 여부를 분류하는 실습 코드입니다.

## 실행 순서

```bash
cd /Users/jeong-yongcheol/Desktop/00_CBNU_AI/My_project/ManDA_Lecture
python3 7th_KAMP_sound_0414/step1_eda.py
python3 7th_KAMP_sound_0414/step2_train.py
python3 7th_KAMP_sound_0414/step3_evaluate.py
python3 7th_KAMP_sound_0414/step4_inference.py
```

## 주요 파일

- `step1_eda.py`: 파형, Half Spectrum, MFCC, 상관관계 히트맵 생성
- `step2_train.py`: 특징 추출 후 의사결정나무 학습 및 모델 저장
- `step3_evaluate.py`: 정확도, 재현율, 정밀도, F1-Score 평가 및 오분류 분석
- `step4_inference.py`: 새로운 사운드 파일 추론

## 자동 저장 결과

- `visualizations/`
  - `step1_waveform.png`
  - `step1_half_spectrum.png`
  - `step1_mfcc.png`
  - `step1_feature_correlation_heatmap.png`
  - `step3_misclassified_sample_*.png`
- `results/`
  - `step4_inference_results.csv`

## 참고

- 권장 실행 환경은 현재 터미널의 `python3`입니다.
- `/usr/local/bin/python3`를 사용할 경우 필요한 패키지를 별도로 설치해야 할 수 있습니다.

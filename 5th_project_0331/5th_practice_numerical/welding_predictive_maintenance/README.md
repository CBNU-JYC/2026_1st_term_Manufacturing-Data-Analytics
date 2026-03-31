# Welding Predictive Maintenance

이 폴더는 용접 공정 데이터를 이용해 시계열 기반 이상 탐지를 수행하는 예제입니다.

## 파일 구성

- `step1_data_preparation.py`
  원본 CSV를 읽어 시계열 시퀀스를 만들고 `processed_dataset.pt`를 생성합니다.
- `step2_model_training.py`
  LSTM AutoEncoder를 학습하고 가장 좋은 모델을 `best_lstm_ae.pth`로 저장합니다.
- `step3_evaluation.py`
  검증셋으로 임곗값을 찾고 테스트셋 평가 및 시각화 이미지를 생성합니다.
- `z_explanation_*.py`
  각 코드 줄의 의미를 쉽게 이해할 수 있도록 한국어 설명 주석을 추가한 학습용 파일입니다.

## 실행 순서

아래 순서대로 실행하면 됩니다.

```bash
cd /Users/jeong-yongcheol/Desktop/00_CBNU_AI/My_project/ManDA_Lecture/5th_project_0331/5th_practice_numerical/welding_predictive_maintenance

/usr/local/bin/python3 step1_data_preparation.py
/usr/local/bin/python3 -u step2_model_training.py
/usr/local/bin/python3 -u step3_evaluation.py
```

## 생성 결과물

- `processed_dataset.pt`
  전처리된 학습/검증/테스트 텐서 묶음
- `best_lstm_ae.pth`
  학습된 LSTM AutoEncoder 가중치
- `precision_recall_threshold.png`
  임곗값 선택용 정밀도/재현율 그래프
- `reconstruction_error_scatter.png`
  정상/이상 샘플의 재구성 오차 산점도
- `reconstruction_error_distribution.png`
  정상/이상 샘플의 재구성 오차 분포
- `confusion_matrix.png`
  최종 분류 결과 혼동행렬

## 참고 사항

- 스크립트들은 현재 파일이 있는 폴더 기준으로 데이터와 결과 파일을 읽고 쓰도록 수정되어 있습니다.
- `step2_model_training.py`는 CPU 환경에서는 시간이 조금 걸릴 수 있습니다.
- `step3_evaluation.py`는 화면 창 대신 PNG 파일을 저장하도록 바뀌어 있어 터미널에서 바로 종료됩니다.

## 최근 확인 결과

- 최적 임곗값: `0.4040`
- 정확도: `100.00%`
- F1-Score: `1.0000`

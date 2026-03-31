# AI4I 2020 Predictive Maintenance

이 폴더는 AI4I 2020 예지보전 데이터를 사용해 고장 여부를 분류하는 MLP 기반 실습 예제입니다.

## 파일 구성

- `step1_eda.py`
  데이터 구조, 센서 분포, 상관관계, 타깃 분포를 확인하고 EDA 이미지를 저장합니다.
- `step2_data_prep.py`
  원본 CSV를 전처리하고 학습/테스트셋, 스케일러, DataLoader를 준비합니다.
- `step3_train_model.py`
  MLP 모델을 학습하고 평가 후 모델, 스케일러, 요약 이미지를 저장합니다.
- `step4_inference.py`
  저장된 모델과 스케일러를 사용해 새 센서 입력 1건에 대해 고장 여부를 추론합니다.
- `z_explanation_*.py`
  각 코드 줄의 의미를 쉽게 이해할 수 있도록 한국어 설명 주석을 추가한 학습용 파일입니다.

## 실행 순서

```bash
cd /Users/jeong-yongcheol/Desktop/00_CBNU_AI/My_project/ManDA_Lecture/5th_project_0331/5th_practice_numerical/ai4i_2020_predictive_maintenance

/usr/local/bin/python3 step1_eda.py
/usr/local/bin/python3 step2_data_prep.py
/usr/local/bin/python3 -u step3_train_model.py
/usr/local/bin/python3 -u step4_inference.py
```

## 생성 결과물

- `eda_sensor_distributions.png`
  주요 센서 변수 분포 그래프
- `eda_correlation_matrix.png`
  센서 변수 간 상관계수 히트맵
- `eda_target_distribution.png`
  정상/고장 타깃 분포 그래프
- `models/fault_diagnosis_mlp.pth`
  학습된 MLP 가중치
- `models/sensor_scaler.pkl`
  학습 시 사용한 표준화 스케일러
- `training_evaluation_summary.png`
  혼동행렬, 평가 지표 막대그래프, ROC Curve 요약 이미지
- `runs/fault_diagnosis_experiment/`
  TensorBoard 로그 디렉토리

## 참고 사항

- 스크립트들은 현재 파일이 있는 폴더 기준으로 데이터와 결과 파일을 읽고 쓰도록 수정되어 있습니다.
- `step1_eda.py`와 `step3_train_model.py`는 터미널 환경에서도 멈추지 않도록 그래프를 PNG 파일로 저장합니다.
- `step2_data_prep.py`는 로컬 `ai4i2020.csv`를 읽도록 정리되어 있어 인터넷 연결 없이도 실행할 수 있습니다.
- 학습 결과는 초기값과 샘플링에 따라 조금씩 달라질 수 있습니다.

## 최근 확인 결과

- 정확도: `0.9215`
- 정밀도: `0.2911`
- 재현율: `0.9118`
- F1-Score: `0.4413`
- ROC-AUC: `0.9813`

## 추론 예시

`step4_inference.py`의 현재 샘플 입력 기준으로는 결함 발생 확률이 매우 높게 예측되며, 최근 확인 결과는 `99.99%`였습니다.

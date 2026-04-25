# 6th Practice Image Study Guide

이 문서는 `6th_practice_image_0407` 폴더 안의 이미지 실습 코드를 초보자 기준으로 빠르게 복습할 수 있게 정리한 학습용 안내서입니다.

원본 코드는 그대로 두고, 설명용 파이썬 파일은 모두 `z_explanation_*.py` 이름으로 따로 만들어 두었습니다.

## 1. 폴더 구성

### `cromate_cnn_anomaly`

- 목표: 이미지를 보고 `정상`과 `불량`을 분류하는 CNN 모델 만들기
- 문제 종류: 분류(Classification)
- 핵심 아이디어: 이미지를 CNN에 넣고 두 클래스 중 하나를 맞히게 학습

중요 파일:

- `step1_data_analysis.py`
  - 데이터 개수와 샘플 이미지를 확인하는 파일
- `step2_train.py`
  - CNN 모델을 학습하는 파일
- `step3_evaluation.py`
  - 학습된 모델 성능을 평가하는 파일
- `step4_inference.py`
  - 한 장의 이미지를 예측하고 Grad-CAM으로 설명하는 파일
- `simple_vision.py`
  - `torchvision` 없이 이미지 전처리를 하기 위한 도구 파일
- `plot_utils.py`
  - 그래프에서 한글 폰트를 설정하는 파일

설명용 파일:

- `z_explanation_cnn_anomaly.py`
- `z_explanation_plot_utils.py`
- `z_explanation_simple_vision.py`
- `z_explanation_step1_data_analysis.py`
- `z_explanation_step2_train.py`
- `z_explanation_step3_evaluation.py`
- `z_explanation_step4_inference.py`

### `mvtec`

- 목표: 정상 이미지만 학습한 뒤, 테스트 이미지가 정상인지 불량인지 이상 탐지하기
- 문제 종류: 이상 탐지(Anomaly Detection)
- 핵심 아이디어: 정상 이미지를 잘 복원하는 오토인코더를 만든 뒤, 복원 오차가 크면 불량으로 판단

중요 파일:

- `step1_data_eda.py`
  - MVTec 데이터셋 구조와 샘플 이미지를 확인하는 파일
- `step2_train.py`
  - 오토인코더 모델을 학습하는 파일
- `step3_evaluate.py`
  - 복원 오차로 이상 점수를 계산하고 시각화하는 파일
- `simple_vision.py`
  - `torchvision` 없이 이미지 전처리를 하는 파일

설명용 파일:

- `z_explanation_simple_vision.py`
- `z_explanation_step1_data_eda.py`
- `z_explanation_step2_train.py`
- `z_explanation_step3_evaluate.py`

## 2. 두 실습의 차이

### CNN 분류 실습

- 입력: 정상/불량 라벨이 붙은 이미지
- 학습 목표: 이미지를 보고 바로 정상인지 불량인지 맞히기
- 출력: 클래스 예측 결과
- 예시 질문: "이 사진은 정상일까, 불량일까?"

### MVTec 이상 탐지 실습

- 입력: 주로 정상 이미지
- 학습 목표: 정상 이미지를 잘 복원하는 방법 배우기
- 출력: 복원 오차 기반 이상 점수
- 예시 질문: "이 사진은 평소 정상 이미지와 얼마나 다를까?"

## 3. 추천 학습 순서

### `cromate_cnn_anomaly`

1. `z_explanation_step1_data_analysis.py`
2. `z_explanation_simple_vision.py`
3. `z_explanation_step2_train.py`
4. `z_explanation_step3_evaluation.py`
5. `z_explanation_step4_inference.py`
6. `z_explanation_cnn_anomaly.py`

이 순서가 좋은 이유:

- 먼저 데이터를 이해해야 뒤 코드가 쉬워집니다.
- 전처리 도구를 먼저 보면 이미지가 어떻게 모델 입력으로 바뀌는지 이해할 수 있습니다.
- 학습, 평가, 추론 순서로 보면 전체 흐름이 자연스럽게 이어집니다.

### `mvtec`

1. `z_explanation_step1_data_eda.py`
2. `z_explanation_simple_vision.py`
3. `z_explanation_step2_train.py`
4. `z_explanation_step3_evaluate.py`

이 순서가 좋은 이유:

- MVTec은 데이터 구조가 조금 다르기 때문에 먼저 폴더 구조를 이해하는 것이 중요합니다.
- 그다음 오토인코더가 무엇을 배우는지 보고,
- 마지막에 복원 오차로 이상을 어떻게 찾는지 보면 이해가 잘 됩니다.

## 4. 자주 나오는 핵심 개념

### `DataLoader`

- 데이터를 한 장씩이 아니라 여러 장씩 묶어서 꺼내 주는 도구입니다.
- 학습 속도를 높이고 코드를 깔끔하게 해 줍니다.

### `Transform`

- 이미지를 모델이 먹기 좋은 형태로 바꾸는 준비 과정입니다.
- 예: 크기 맞추기, 숫자 텐서로 바꾸기

### `Tensor`

- 딥러닝에서 사용하는 숫자 상자입니다.
- 이미지도 결국 숫자로 바꿔서 모델에 넣습니다.

### `Loss`

- 모델이 얼마나 틀렸는지 알려 주는 점수입니다.
- 학습은 이 점수를 줄이는 방향으로 진행됩니다.

### `Optimizer`

- 손실을 줄이도록 모델 가중치를 업데이트하는 도구입니다.

### `Epoch`

- 전체 학습 데이터를 한 바퀴 다 보는 것을 1에폭이라고 합니다.

### `Grad-CAM`

- 모델이 이미지의 어느 부분을 중요하게 봤는지 색으로 보여 주는 설명 도구입니다.

### `Autoencoder`

- 입력 이미지를 압축했다가 다시 복원하는 모델입니다.
- 정상 이미지를 잘 복원하도록 학습하면,
- 이상 이미지에서는 복원을 잘 못해서 오차가 커질 수 있습니다.

## 5. 실행 순서 요약

### CNN 분류 실습 실행

```bash
python step1_data_analysis.py
python step2_train.py
python step3_evaluation.py
python step4_inference.py
```

### MVTec 이상 탐지 실습 실행

```bash
python step1_data_eda.py
python step2_train.py
python step3_evaluate.py
```

## 6. 초보자용 공부 팁

- 처음에는 모델 수식보다 "입력 -> 처리 -> 출력" 흐름을 먼저 보세요.
- `print()`가 어디서 무엇을 보여 주는지 먼저 따라가면 훨씬 덜 어렵습니다.
- 함수 이름과 변수 이름만 읽어도 코드의 절반은 이해할 수 있습니다.
- 설명 파일을 먼저 읽고, 그 다음 원본 파일을 보면 훨씬 쉬워집니다.
- 한 번에 전체를 이해하려 하지 말고, 파일 하나씩 끊어서 보는 것이 좋습니다.

## 7. 내가 먼저 보면 좋은 파일

정말 처음 시작한다면 아래 두 파일부터 보는 것을 추천합니다.

- `cromate_cnn_anomaly/z_explanation_step2_train.py`
- `mvtec/z_explanation_step2_train.py`

이유:

- 두 파일 모두 "모델이 어떻게 배우는지"가 가장 잘 드러납니다.
- CNN 분류와 오토인코더 이상 탐지의 차이도 비교하기 쉽습니다.

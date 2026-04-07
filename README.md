# 2026 1학기 제조 데이터 분석과 최적화

![GitHub repo size](https://img.shields.io/github/repo-size/CBNU-JYC/2026_1st_term_Manufacturing-Data-Analytics)
![GitHub last commit](https://img.shields.io/github/last-commit/CBNU-JYC/2026_1st_term_Manufacturing-Data-Analytics)
![Python](https://img.shields.io/badge/Python-Anaconda%20Environment-3776AB?logo=python&logoColor=white)
![VS Code](https://img.shields.io/badge/Editor-VS%20Code-007ACC?logo=visualstudiocode&logoColor=white)
![Course](https://img.shields.io/badge/Course-Manufacturing%20Data%20Analytics-0A7E8C)

충북대학교 산업인공지능학 전공 `제조 데이터 분석과 최적화` 강의의 실습 코드, 과제, 설명용 스크립트, 생성 데이터셋을 정리한 저장소입니다.

이 저장소는 제조 데이터 품질 정제, OPC-UA 통신 실습, 데이터 파이프라인 구축, 정보 모델 실습, 예지보전 분류 과제, MVTec 이상 탐지 프로젝트를 주차별로 관리하는 것을 목표로 합니다.

## 저장소 정보

- 저장소: <https://github.com/CBNU-JYC/2026_1st_term_Manufacturing-Data-Analytics.git>
- 로컬 경로: `/Users/jeong-yongcheol/Desktop/00_CBNU_AI/My_project/ManDA_Lecture`
- 개발 환경: `Anaconda Navigator + VSCode`
- 언어/도구: `Python`, `pandas`, `numpy`, `OPC-UA`

## 교과목 정보

- 강의명: `제조 데이터 분석과 최적화`
- 개설연도-학기: `2026년 1학기`
- 개설학과: `산업인공지능학`
- 교과목번호-분반번호: `8884024-01`
- 이수구분: `전공심화`
- 학점/시수: `3-3-0`
- 강의시간/강의실: `화 11, 12, 13 [901-A601]`
- 담당교수: `김한진`

## 핵심 학습 주제

- 제조 데이터 품질 진단과 정제 기준 수립
- 결측치, 중복, 이상치, 상수 센서 제거 실습
- OPC-UA 서버/클라이언트 기반 제조 센서 데이터 수집
- 데이터 수집-정제-검증 파이프라인 구현
- OPC UA 정보 모델 심화 실습
- SECOM 반도체 공정 데이터셋 품질 확보 과제
- AI4I 2020 기반 제조 설비 고장 분류 모델 개선
- MVTec AD 기반 오토인코더 이상 탐지 프로젝트

## 프로젝트 구성

| 폴더 | 내용 | 주요 파일 | 산출물 |
|---|---|---|---|
| `3rd_project_0317` | 제조 데이터 정제 및 품질 검증 실습 | `clean_dataset.py`, `Explain_clean_dataset.py`, `expected_qa.md` | `labeled_data_clean.csv` |
| `3rd_homework_0331` | SECOM 데이터셋 품질 확보 과제 본 제출본 | `secom_quality_clean.py`, `z_explanation_secom_quality_clean.py` | `cleaned_secom_data.csv` |
| `4th_project_0324/4th_practice/opcua_basic` | OPC-UA 기본 통신 및 제조 센서 수집 | `opc_server.py`, `opc_client.py`, `opc_server_mfg.py`, `opc_client_mfg.py` | `manufacturing_sensor_data.csv` |
| `4th_project_0324/4th_practice/data_pipeline` | 제조 데이터 수집-정제-검증 파이프라인 | `data_pipeline.py`, `data_pipeline_check.py`, `opc_server_mfg.py` | `raw_sensor_dataset.csv`, `verified_sensor_dataset.csv`, `lifecycle_optimized_dataset.csv` |
| `4th_project_0324/4th_practice/information_model` | OPC UA 정보 모델 심화 실습 | `advanced_server.py`, `advanced_client.py` | 정보 모델 실습 로그 및 호출 결과 |
| `5th_homework_0407` | AI4I 2020 예지보전 데이터셋 기반 개선 분류 과제 | `improved_model.py`, `z_explanation_improved_model.py` | `evaluation_result.png`, `results.json`, `models/` |
| `5th_project_0331/5th_practice_numerical` | 수치형 제조 데이터 기반 예지보전 실습 | `fault_diagnosis_mlp.pth`, `best_lstm_ae.pth` 외 실습 결과물 | 학습 모델, 임곗값 분석 그래프, 재구성 오차 시각화 |
| `6th_project_0407/mvtec` | MVTec AD bottle 클래스 기반 오토인코더 이상 탐지 프로젝트 | `step1_data_eda.py`, `step2_train.py`, `step3_evaluate.py` | `autoencoder_model.pth`, 평가용 시각화 및 데이터셋 |

## 주요 과제 및 실습 요약

### 1. `3rd_project_0317`

- 사출성형 공정 데이터를 대상으로 유일성, 완전성, 유효성, 일관성, 정확성 관점의 정제를 수행합니다.
- `clean_dataset.py`는 중복 제거, 핵심 센서 결측 제거, 물리적으로 불가능한 시간값 제거, 논리 모순 제거, 비정상 온도 제거를 수행합니다.
- `Explain_clean_dataset.py`는 코드 이해를 돕기 위한 설명용 버전입니다.

### 2. `3rd_homework_0331`

- UCI SECOM 반도체 공정 데이터셋을 대상으로 데이터 품질 확보 과제를 수행합니다.
- `secom_quality_clean.py`는 다음 절차를 포함합니다.
  - 센서 기준 중복 행 제거
  - 결측률 50% 초과 센서 제거
  - 남은 결측치 중앙값 보간
  - 분산 0 센서 제거
  - IQR 기반 이상치 클리핑
  - `Pass/Fail` 레이블 유효성 검증
  - MDQI 관점의 품질 점수 계산 및 출력
- `z_explanation_secom_quality_clean.py`는 같은 로직을 학습용 주석과 함께 정리한 설명 버전입니다.

### 3. `4th_project_0324`

- `opcua_basic`: OPC-UA 서버/클라이언트 연결과 제조 센서 데이터 수집 실습
- `data_pipeline`: 원천 데이터 수집부터 검증/최적화 데이터셋 생성까지의 파이프라인 실습
- `information_model`: 고급 OPC UA 정보 모델 서버/클라이언트 실습

### 4. `5th_homework_0407`

- AI4I 2020 Predictive Maintenance 데이터셋을 사용해 제조 설비 고장 여부를 분류하는 개선 모델 과제입니다.
- `improved_model.py`는 데이터 전처리, 클래스 불균형 대응, MLP 학습, 최적 threshold 탐색, 성능 평가를 포함합니다.
- `z_explanation_improved_model.py`는 동일 로직을 학습용 설명과 함께 정리한 주석 강화 버전입니다.
- 결과물로 `evaluation_result.png`, `results.json`, `models/improved_mlp.pth`, `models/improved_scaler.pkl`, `models/optimal_threshold.pkl`을 저장합니다.

### 5. `6th_project_0407`

- MVTec AD `bottle` 클래스 데이터를 사용해 비지도 이상 탐지 오토인코더를 학습하고 평가하는 프로젝트입니다.
- `step1_data_eda.py`는 데이터 구조 확인과 이미지 분포 탐색을 담당합니다.
- `step2_train.py`는 정상 이미지를 기반으로 오토인코더를 학습하고 `autoencoder_model.pth`를 저장합니다.
- `step3_evaluate.py`는 재구성 오차를 이용해 이상 여부를 판별하고 평가를 수행합니다.

## 빠른 시작

### 저장소 열기

```bash
cd /Users/jeong-yongcheol/Desktop/00_CBNU_AI/My_project/ManDA_Lecture
```

### 권장 환경

```bash
python --version
```

- `Anaconda Navigator`에서 사용하는 Python 환경을 활성화한 뒤 실행하는 것을 권장합니다.
- `VSCode`에서 프로젝트 폴더를 그대로 열어 실습할 수 있습니다.

## 실행 예시

### 1. 사출 공정 데이터 정제 실습

```bash
cd 3rd_project_0317
python clean_dataset.py
```

### 2. SECOM 품질 확보 과제 실행

```bash
cd 3rd_homework_0331
python secom_quality_clean.py
```

실행 결과:

- `uci-secom.csv`를 읽어 정제 수행
- `cleaned_secom_data.csv` 저장
- 각 품질 지표와 MDQI 계산 결과 출력

### 3. OPC-UA 기본 실습

서버 실행:

```bash
cd 4th_project_0324/4th_practice/opcua_basic
python opc_server.py
```

클라이언트 실행:

```bash
cd 4th_project_0324/4th_practice/opcua_basic
python opc_client.py
```

### 4. 제조 센서 수집 실습

서버 실행:

```bash
cd 4th_project_0324/4th_practice/opcua_basic
python opc_server_mfg.py
```

클라이언트 실행:

```bash
cd 4th_project_0324/4th_practice/opcua_basic
python opc_client_mfg.py
```

### 5. 데이터 파이프라인 실습

서버 실행:

```bash
cd 4th_project_0324/4th_practice/data_pipeline
python opc_server_mfg.py
```

파이프라인 실행:

```bash
cd 4th_project_0324/4th_practice/data_pipeline
python data_pipeline.py
```

검증 스크립트 실행:

```bash
cd 4th_project_0324/4th_practice/data_pipeline
python data_pipeline_check.py
```

### 6. AI4I 2020 개선 모델 과제 실행

```bash
cd 5th_homework_0407
python improved_model.py
```

실행 결과:

- `ai4i2020.csv`를 읽어 전처리 및 학습 수행
- 최적 threshold 기반 평가 결과 계산
- `evaluation_result.png`, `results.json`, `models/` 산출물 저장

### 7. MVTec 오토인코더 프로젝트 실행

데이터 탐색:

```bash
cd 6th_project_0407/mvtec
python step1_data_eda.py
```

학습:

```bash
cd 6th_project_0407/mvtec
python step2_train.py
```

평가:

```bash
cd 6th_project_0407/mvtec
python step3_evaluate.py
```

## 현재 저장소 구조

```text
ManDA_Lecture/
├── 3rd_project_0317/
│   ├── clean_dataset.py
│   ├── Explain_clean_dataset.py
│   ├── labeled_data.csv
│   ├── labeled_data_clean.csv
│   ├── environment.yml
│   ├── requirements.txt
│   ├── expected_qa.md
│   ├── note.md
│   ├── note.txt
│   └── presentation_summary.md
├── 3rd_homework_0331/
│   ├── secom_quality_clean.py
│   ├── z_explanation_secom_quality_clean.py
│   ├── uci-secom.csv
│   └── cleaned_secom_data.csv
├── 4th_project_0324/
│   └── 4th_practice/
│       ├── opcua_basic/
│       ├── data_pipeline/
│       └── information_model/
├── 5th_homework_0407/
│   ├── improved_model.py
│   ├── z_explanation_improved_model.py
│   ├── ai4i2020.csv
│   ├── evaluation_result.png
│   ├── results.json
│   └── models/
├── 5th_project_0331/
│   └── 5th_practice_numerical/
├── 6th_project_0407/
│   └── mvtec/
│       ├── step1_data_eda.py
│       ├── step2_train.py
│       ├── step3_evaluate.py
│       └── autoencoder_model.pth
├── secom_clean.csv
├── .gitignore
└── README.md
```

## 참고 사항

- 설명용 파일(`Explain_*.py`, `z_explanation_*.py`)은 원본 코드 학습을 돕기 위한 주석 강화 버전입니다.
- 생성 CSV는 실습 결과 확인과 후속 분석을 위해 저장소에 함께 보관합니다.
- Python 캐시와 개발 환경 파일(`__pycache__`, `*.pyc`, `.cache/`, `.matplotlib/`, `.vscode/`, `.idea/`)은 버전 관리 대상에서 제외합니다.

## 설명용 코드 작성 지침

초보자용 파이썬 코드 설명 파일은 원본 파일을 유지한 채, 파일명 앞에 `z_explanation_`을 붙여 별도 파일로 작성합니다.

요구사항:

1. 각 줄 또는 주요 구문마다 `#` 주석으로 "이 코드가 무슨 일을 하는지" 초등학생도 이해할 수 있게 설명합니다.
2. 함수 위에는 함수 역할, 매개변수 설명, 반환값 설명을 포함한 docstring을 작성합니다.
3. 프로그램 전체 흐름을 파일 맨 위에서 간단히 설명합니다.
4. 복잡한 부분은 왜 이렇게 작성했는지 이유도 함께 설명합니다.
5. 주석은 PEP 8 스타일에 맞게 `#` 뒤를 한 칸 띄워 정렬합니다.

응답 형식:

- 주석만 추가한 완성 코드 형태로 작성합니다.
- 설명 파일은 반드시 `z_explanation_파일명.py` 형식으로 저장합니다.

## 강의 자료 링크

- 충북대학교 EISN: <https://eisn.cbnu.ac.kr/nxui/index.html?OBSC_YN=0&LNG=ko#main>

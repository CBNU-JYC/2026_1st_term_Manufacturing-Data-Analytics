# 2026 1학기 제조 데이터 분석과 최적화

![GitHub repo size](https://img.shields.io/github/repo-size/CBNU-JYC/2026_1st_term_Manufacturing-Data-Analytics)
![GitHub last commit](https://img.shields.io/github/last-commit/CBNU-JYC/2026_1st_term_Manufacturing-Data-Analytics)
![Python](https://img.shields.io/badge/Python-Anaconda%20Environment-3776AB?logo=python&logoColor=white)
![VS Code](https://img.shields.io/badge/Editor-VS%20Code-007ACC?logo=visualstudiocode&logoColor=white)
![Course](https://img.shields.io/badge/Course-Manufacturing%20Data%20Analytics-0A7E8C)

충북대학교 산업인공지능학 전공 `제조 데이터 분석과 최적화` 강의의 실습 코드, 과제, 설명용 스크립트, 생성 데이터셋을 정리한 저장소입니다.

이 저장소는 제조 데이터 품질 정제, OPC-UA 통신 실습, 데이터 파이프라인 구축, 정보 모델 실습, SECOM 반도체 공정 데이터 과제를 주차별로 관리하는 것을 목표로 합니다.

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

## 프로젝트 구성

| 폴더 | 내용 | 주요 파일 | 산출물 |
|---|---|---|---|
| `3rd_project_0317` | 제조 데이터 정제 및 품질 검증 실습 | `clean_dataset.py`, `Explain_clean_dataset.py`, `expected_qa.md` | `labeled_data_clean.csv` |
| `3rd_homework_0331` | SECOM 데이터셋 품질 확보 과제 본 제출본 | `secom_quality_clean.py`, `z_explanation_secom_quality_clean.py` | `cleaned_secom_data.csv` |
| `z_3rd_homework_0331` | SECOM 데이터셋 보조 정제/설명 버전 | `clean_uci-secom_dataset.py`, `z_explanation_clean_uci-secom_dataset.py`, `z_explanation2_clean_uci-secom_dataset.py` | `cleaned_secom_data.csv` |
| `4th_project_0324/4th_practice/opcua_basic` | OPC-UA 기본 통신 및 제조 센서 수집 | `opc_server.py`, `opc_client.py`, `opc_server_mfg.py`, `opc_client_mfg.py` | `manufacturing_sensor_data.csv` |
| `4th_project_0324/4th_practice/data_pipeline` | 제조 데이터 수집-정제-검증 파이프라인 | `data_pipeline.py`, `data_pipeline_check.py`, `opc_server_mfg.py` | `raw_sensor_dataset.csv`, `verified_sensor_dataset.csv`, `lifecycle_optimized_dataset.csv` |
| `4th_project_0324/4th_practice/information_model` | OPC UA 정보 모델 심화 실습 | `advanced_server.py`, `advanced_client.py` | 정보 모델 실습 로그 및 호출 결과 |

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

### 3. `z_3rd_homework_0331`

- SECOM 데이터셋을 조금 더 간단한 형태로 정제한 보조 스크립트 모음입니다.
- `clean_uci-secom_dataset.py`는 결측률 기반 컬럼 제거, 중앙값 보간, 상수 센서 제거, 라벨 포맷 정리, IQR 기반 이상치 보정을 수행합니다.
- 설명 파일 두 개는 같은 과정을 단계별로 이해할 수 있도록 정리한 참고용 버전입니다.

### 4. `4th_project_0324`

- `opcua_basic`: OPC-UA 서버/클라이언트 연결과 제조 센서 데이터 수집 실습
- `data_pipeline`: 원천 데이터 수집부터 검증/최적화 데이터셋 생성까지의 파이프라인 실습
- `information_model`: 고급 OPC UA 정보 모델 서버/클라이언트 실습

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
├── z_3rd_homework_0331/
│   ├── clean_uci-secom_dataset.py
│   ├── z_explanation_clean_uci-secom_dataset.py
│   ├── z_explanation2_clean_uci-secom_dataset.py
│   ├── uci-secom.csv
│   └── cleaned_secom_data.csv
├── 4th_project_0324/
│   └── 4th_practice/
│       ├── opcua_basic/
│       ├── data_pipeline/
│       └── information_model/
├── secom_clean.csv
├── .gitignore
└── README.md
```

## 참고 사항

- 설명용 파일(`Explain_*.py`, `z_explanation_*.py`)은 원본 코드 학습을 돕기 위한 주석 강화 버전입니다.
- 생성 CSV는 실습 결과 확인과 후속 분석을 위해 저장소에 함께 보관합니다.
- Python 캐시 파일(`__pycache__`, `*.pyc`)은 버전 관리 대상에서 제외합니다.

## 강의 자료 링크

- 충북대학교 EISN: <https://eisn.cbnu.ac.kr/nxui/index.html?OBSC_YN=0&LNG=ko#main>


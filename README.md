# 2026 1학기 제조 데이터 분석과 최적화

충북대학교 산업인공지능학 전공 `제조 데이터 분석과 최적화` 강의의 실습 코드, 설명 파일, 실행 결과물, 환경 설정을 정리한 저장소입니다.

이 저장소는 강의 실습 내용을 주차별 프로젝트 형태로 정리하고, Python 실습 코드와 설명용 파이썬 파일, 생성된 데이터셋 결과물까지 함께 관리하는 것을 목표로 합니다.

## 한눈에 보기

- 강의명: `제조 데이터 분석과 최적화`
- 학기: `2026년 1학기`
- 저장소: <https://github.com/CBNU-JYC/2026_1st_term_Manufacturing-Data-Analytics.git>
- 개발 환경: `Anaconda Navigator + VSCode`
- 실습 핵심 주제:
  - 데이터 정제 및 품질 개선
  - OPC-UA 기반 제조 데이터 수집
  - 데이터 파이프라인 구축
  - OPC UA 정보 모델 심화 실습

## 1. 저장소 정보

- 저장소 이름: `2026_1st_term_Manufacturing-Data-Analytics`
- GitHub 주소: <https://github.com/CBNU-JYC/2026_1st_term_Manufacturing-Data-Analytics.git>
- GitHub 계정: `draonfe73@chungbuk.ac.kr`

## 2. 강의 자료 링크

- 충북대학교 EISN 강의 자료 링크:
  <https://eisn.cbnu.ac.kr/nxui/index.html?OBSC_YN=0&LNG=ko#main>

## 3. 교과목 정보

- 개설연도-학기: `2026년 1학기`
- 개설학과: `산업인공지능학`
- 교과목번호-분반번호: `8884024-01`
- 교과목명: `제조 데이터 분석과 최적화`
- 이수구분: `전공심화`
- 학점/시수: `3-3-0`
- 강의시간/강의실: `화 11, 12, 13 [901-A601]`
- 담당교수: `김한진(초빙교원)`
- 전화: `043-249-1466`
- 이메일: `gks359@cbnu.ac.kr`

## 4. 교과목 개요

### 강의 개요

본 교과목은 스마트팩토리 및 자율 제조 시대를 맞이하여, 제조 현장에서 발생하는 다양한 데이터의 특성을 이해하고 이를 분석 및 최적화하는 핵심 역량을 배양하는 것을 목표로 합니다.

수강생은 제조데이터에 대한 이해와 제조데이터를 활용하여 AI를 개발하는 방법을 실습을 통해 학습합니다. 나아가, 공정 효율을 극대화하기 위한 생산 스케줄링 최적화 기법을 심도 있게 다룹니다.

### 학습목표

1. 제조 데이터의 특성을 이해하고 데이터 품질 평가 체계를 수립할 수 있다.
2. 수치, 이미지, 소리 등 다양한 형태의 제조 데이터를 활용하여 AI 모델을 직접 개발하고 성능을 검증할 수 있다.
3. 다양한 제조 환경을 이해하고, 상황에 맞는 최적의 스케줄링 해법을 도출할 수 있다.

## 5. 개발 및 실행 환경

- Python 환경: `Anaconda Navigator` 기반의 `VSCode` 가상환경
- 프로젝트 열기 경로:
  `/Users/jeong-yongcheol/Desktop/00_CBNU_AI/My_project/ManDA_Lecture/`
- 실습 중 `Codex AI`의 도움을 받아 코드 설명 파일과 실행/정리 작업을 보조함

## 6. 빠른 시작

### 저장소 열기

```bash
cd /Users/jeong-yongcheol/Desktop/00_CBNU_AI/My_project/ManDA_Lecture
```

### 권장 실행 환경

- `Anaconda Navigator`에서 사용하는 Python 환경을 활성화한 뒤 실행
- `VSCode`에서 위 프로젝트 경로를 그대로 열어 실습

### OPC-UA 기본 실습 예시

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

### 제조 센서 수집 실습 예시

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

### 데이터 파이프라인 실습 예시

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

## 7. 강의 노트 및 원본 자료 위치

- 맥북 로컬 참고 경로:
  `desktop/00_CBNU_AI/1_(대학원, 오창 A704)제조 데이터 분석과 최적화 (8884024-01), 김한진교수`

## 8. 주차별 학습 정리

| 주차/폴더 | 핵심 주제 | 주요 파일 | 결과물 |
|---|---|---|---|
| `3rd_project_0317` | 데이터 정제, 라벨링, 품질 개선 | `clean_dataset.py`, `Explain_clean_dataset.py` | `labeled_data_clean.csv` |
| `4th_project_0324/4th_practice/opcua_basic` | OPC-UA 기본 서버/클라이언트 통신 | `opc_server.py`, `opc_client.py`, `opc_server_mfg.py`, `opc_client_mfg.py` | `manufacturing_sensor_data.csv` |
| `4th_project_0324/4th_practice/data_pipeline` | 제조 데이터 수집-정제-라벨링-검증 파이프라인 | `data_pipeline.py`, `data_pipeline_check.py`, `opc_server_mfg.py` | `raw_sensor_dataset.csv`, `verified_sensor_dataset.csv`, `lifecycle_optimized_dataset.csv` |
| `4th_project_0324/4th_practice/information_model` | OPC UA 정보 모델 심화, DA/HA/AC/Prog | `advanced_server.py`, `advanced_client.py` | 실시간 이벤트/이력/메서드 호출 실습 |

## 9. 현재 저장소 구조

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
├── 4th_project_0324/
│   └── 4th_practice/
│       ├── opcua_basic/
│       │   ├── opc_server.py
│       │   ├── opc_client.py
│       │   ├── opc_server_mfg.py
│       │   ├── opc_client_mfg.py
│       │   ├── z_explation_*.py
│       │   ├── __pycache__/
│       │   └── manufacturing_sensor_data.csv
│       ├── data_pipeline/
│       │   ├── data_pipeline.py
│       │   ├── data_pipeline_check.py
│       │   ├── opc_server_mfg.py
│       │   ├── z_explation_*.py
│       │   ├── raw_sensor_dataset.csv
│       │   ├── verified_sensor_dataset.csv
│       │   └── lifecycle_optimized_dataset.csv
│       └── information_model/
│           ├── advanced_server.py
│           ├── advanced_client.py
│           └── z_explanation_*.py
├── .gitignore
└── README.md
```

## 10. 디렉토리별 설명

### `3rd_project_0317`

- 데이터 정제와 라벨링 관련 실습 자료가 들어 있습니다.
- `clean_dataset.py`는 데이터 정제 실습용 스크립트입니다.
- `Explain_clean_dataset.py`는 원본 코드를 이해하기 쉽게 설명한 파일입니다.
- `labeled_data.csv`, `labeled_data_clean.csv`는 데이터셋 및 정제 결과 파일입니다.
- `environment.yml`, `requirements.txt`는 실행 환경 복원용 의존성 파일입니다.

### `4th_project_0324/4th_practice/opcua_basic`

- OPC-UA 서버와 클라이언트의 기본 연결 구조를 실습하는 폴더입니다.
- `opc_server.py`, `opc_client.py`는 가장 기본적인 OPC-UA 통신 예제입니다.
- `opc_server_mfg.py`, `opc_client_mfg.py`는 제조 설비 센서 시나리오를 반영한 예제입니다.
- `z_explation_*.py` 파일은 각 원본 코드를 한글 주석 중심으로 상세 설명한 버전입니다.
- `__pycache__`는 실행 과정에서 생성된 Python 캐시 파일입니다.
- `manufacturing_sensor_data.csv`는 실습 실행 결과 저장된 센서 데이터셋입니다.

### `4th_project_0324/4th_practice/data_pipeline`

- 제조 센서 데이터를 수집, 정제, 라벨링, 검증하는 데이터 파이프라인 실습 폴더입니다.
- `data_pipeline.py`는 수집부터 정제, 라벨링, 저장까지 이어지는 기본 파이프라인입니다.
- `data_pipeline_check.py`는 데이터 품질 검증 항목을 강화한 버전입니다.
- `opc_server_mfg.py`는 센서값을 공급하는 제조 설비 서버 예제입니다.
- `z_explation_*.py` 파일은 코드 설명용 파이썬 파일입니다.
- `raw_sensor_dataset.csv`, `verified_sensor_dataset.csv`, `lifecycle_optimized_dataset.csv`는 실습 결과물입니다.

### `4th_project_0324/4th_practice/information_model`

- OPC UA의 심화 기능을 실습하는 폴더입니다.
- `advanced_server.py`는 실시간 데이터(DA), 이력 저장(HA), 이벤트(AC), 원격 메서드 호출(Prog)을 포함한 서버 예제입니다.
- `advanced_client.py`는 심화 서버의 기능을 읽고 사용하는 클라이언트 예제입니다.
- `z_explanation_*.py` 파일은 각 파일을 쉽게 이해할 수 있도록 설명을 추가한 버전입니다.

## 11. 포함 파일 안내

- 이 저장소에는 코드 파일뿐 아니라 실습 중 생성된 CSV 결과물도 함께 포함되어 있습니다.
- 일부 폴더에는 실행 중 자동 생성된 `__pycache__`와 편집기 설정 파일이 포함되어 있습니다.
- 설명용 파일은 강의 복습과 코드 이해를 돕기 위해 별도로 작성한 버전입니다.
- 설명 파일 이름은 실습 당시 생성된 이름(`z_explation_*`, `z_explanation_*`)을 그대로 유지했습니다.

## 12. GitHub 업로드 관련 메모

- 참고 유튜브:
  <https://youtu.be/AOn6UUscqQw?si=eo5WK_GSbmj-4LSi>
- 주제: `기존 프로젝트 Github(Remote Repository)에 올리기`

체크리스트:

- [ ] 준비물: `git scm`, `pycharm`, `github.com id`
- [ ] pycharm에서 git에 올릴 프로젝트 불러오기(새 project 만들기)
- [ ] github.com에서 새 repository 만들기
- [ ] `git init` 하기
- [ ] `.py` 파일 하나 만들기
- [ ] `git add` 하기
- [ ] `git commit` 하기
- [ ] `git remote`에 GitHub 새 저장소 주소 등록하기
- [ ] `git push` 하기
- [ ] 업로드 확인하기
- [ ] collaborator 추가하기

## 13. 참고 사항

- 강의 실습 환경과 로컬 경로 설명은 작성 당시 사용 환경을 기준으로 기록했습니다.
- 이후 주차 실습이 추가되면 README의 저장소 구조와 주차별 학습 정리 표도 함께 갱신하는 것을 권장합니다.

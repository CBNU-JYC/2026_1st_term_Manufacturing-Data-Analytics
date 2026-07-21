"""
프로그램 전체 흐름 설명:
1. Step 2에서 저장한 모델과 테스트 데이터를 불러옵니다.
2. 모델이 테스트 데이터를 얼마나 잘 맞히는지 정확도, 재현율, 정밀도, F1 점수로 확인합니다.
3. 의사결정나무가 어떤 질문을 거쳐 정상/이상을 판단했는지 샘플별로 출력합니다.
4. 틀린 샘플이 있으면 특징값을 그래프로 비교해 왜 헷갈렸는지 살펴봅니다.
5. 평가 결과와 예측 결과를 results 폴더에 저장합니다.
"""

import os  # 파일과 폴더 경로를 만들 때 사용합니다.
import json  # 평가 결과를 json 파일로 저장할 때 사용합니다.
import pandas as pd  # CSV 파일을 읽고 표를 다룰 때 사용합니다.
import numpy as np  # 배열 비교와 숫자 계산에 사용합니다.
import joblib  # 저장된 모델 파일을 불러올 때 사용합니다.
from sklearn import metrics  # 정확도, F1 점수, 오차 행렬 같은 평가 도구입니다.


def explain_decision_path(model, sample_X, feature_names):
    """
    샘플 하나가 의사결정나무 안에서 어떤 조건을 지나 예측됐는지 출력합니다.

    Args:
        model: 학습된 DecisionTreeClassifier 모델입니다.
        sample_X: 예측 과정을 확인할 샘플 1개의 특징 표입니다.
        feature_names: 특징 이름 목록입니다.

    Returns:
        None: 판단 과정을 화면에 출력만 합니다.
    """

    node_indicator = model.decision_path(sample_X)  # 샘플이 지나간 나무 노드들을 찾습니다.
    leaf_id = model.apply(sample_X)[0]  # 샘플이 마지막에 도착한 잎 노드 번호입니다.

    feature = model.tree_.feature  # 각 노드가 어떤 특징을 질문하는지 담긴 배열입니다.
    threshold = model.tree_.threshold  # 각 노드의 기준값이 담긴 배열입니다.

    node_index = node_indicator.indices[
        node_indicator.indptr[0]:node_indicator.indptr[1]
    ]  # 지나간 노드 번호만 잘라냅니다.

    print(f"▶ 분석 대상 샘플의 특성 값: {sample_X.iloc[0].to_dict()}")  # 샘플 값을 보여줍니다.
    print("-" * 50)  # 출력 구분선을 그립니다.
    print("🚦 [AI의 의사결정 흐름]")  # 판단 흐름 제목입니다.

    for node_id in node_index:  # 지나간 노드를 하나씩 확인합니다.
        if leaf_id == node_id:  # 마지막 도착지라면 더 이상 질문하지 않습니다.
            pred_class = model.classes_[model.predict(sample_X)[0]]  # 모델의 최종 예측값입니다.
            status = "이상(Error)" if pred_class == 1 else "정상(OK)"  # 숫자 예측을 사람이 읽는 글자로 바꿉니다.
            print(f" └── [최종 판별] 리프 노드({node_id}) 도달 ➔ 예측 결과: {status}")  # 최종 결과를 출력합니다.
            continue  # 다음 반복으로 넘어갑니다.

        sample_value = sample_X.iloc[0, feature[node_id]]  # 현재 질문에 쓰인 샘플의 실제 값입니다.
        node_feature_name = feature_names[feature[node_id]]  # 현재 질문에 쓰인 특징 이름입니다.
        node_threshold = threshold[node_id]  # 현재 질문의 기준값입니다.

        if sample_value <= node_threshold:  # 샘플 값이 기준보다 작거나 같으면 왼쪽으로 갑니다.
            threshold_sign = "<="  # 비교 결과를 글자로 저장합니다.
            direction = "왼쪽(True)"  # 이동 방향을 저장합니다.
        else:  # 샘플 값이 기준보다 크면 오른쪽으로 갑니다.
            threshold_sign = ">"  # 비교 결과를 글자로 저장합니다.
            direction = "오른쪽(False)"  # 이동 방향을 저장합니다.

        print(f" ├── [노드 {node_id}] 조건: {node_feature_name} <= {node_threshold:.4f}")  # 질문 조건을 출력합니다.
        print(
            f" │   ↳ 샘플의 값({sample_value:.4f})은 "
            f"{node_threshold:.4f} {threshold_sign} 이므로 {direction} 가지로 이동"
        )  # 왜 그 방향으로 갔는지 출력합니다.

    print("-" * 50)  # 출력 구분선을 그립니다.


print("저장된 모델과 테스트 데이터를 불러옵니다...")  # 파일 로드 시작을 알립니다.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 코드 파일이 있는 폴더입니다.
VIS_DIR = os.path.join(BASE_DIR, "0_result")  # 시각화 이미지를 저장할 폴더입니다.
RESULTS_DIR = os.path.join(BASE_DIR, "0_result")  # 평가 결과 파일을 저장할 폴더입니다.
os.makedirs(VIS_DIR, exist_ok=True)  # 이미지 폴더가 없으면 만듭니다.
os.makedirs(RESULTS_DIR, exist_ok=True)  # 결과 폴더가 없으면 만듭니다.

try:  # 필요한 파일들이 있는지 확인하며 불러옵니다.
    Dtc_loaded = joblib.load(os.path.join(RESULTS_DIR, 'dtc_sound_model.pkl'))  # 학습된 모델을 불러옵니다.
    X_test = pd.read_csv(os.path.join(RESULTS_DIR, 'X_test.csv'))  # 테스트 입력 데이터를 읽습니다.
    y_test = pd.read_csv(os.path.join(RESULTS_DIR, 'y_test.csv'))['NG']  # 테스트 정답만 읽습니다.

    test_filepaths = None  # 테스트 파일 경로가 없을 수도 있으므로 처음에는 비워둡니다.
    test_filepaths_path = os.path.join(RESULTS_DIR, 'test_filepaths.csv')  # 파일 경로 CSV 위치입니다.
    if os.path.exists(test_filepaths_path):  # 파일 경로 CSV가 있으면 읽습니다.
        test_filepaths = pd.read_csv(test_filepaths_path)['filepath']  # 원본 오디오 경로 열입니다.

    print("불러오기 완료!\n")  # 로드 성공을 알립니다.
except FileNotFoundError:  # 모델이나 테스트 데이터가 없으면 Step 2를 먼저 해야 합니다.
    print("오류: 모델 파일이나 데이터 파일을 찾을 수 없습니다. 학습 코드를 먼저 실행해주세요.")
    exit()  # 더 진행하지 않고 프로그램을 끝냅니다.

print("=== [모델 전체 성능 평가 지표] ===")  # 평가 결과 제목입니다.
y_pred = Dtc_loaded.predict(X_test)  # 테스트 입력에 대한 모델 예측입니다.

accuracy = metrics.accuracy_score(y_test, y_pred)  # 전체 중 맞힌 비율입니다.
recall = metrics.recall_score(y_test, y_pred)  # 실제 이상 중 이상이라고 잘 찾은 비율입니다.
precision = metrics.precision_score(y_test, y_pred)  # 이상이라고 예측한 것 중 진짜 이상의 비율입니다.
f1_score = metrics.f1_score(y_test, y_pred)  # 정밀도와 재현율을 함께 본 점수입니다.

print(f"테스트 정확도(Accuracy): {accuracy:.2f}")  # 정확도를 출력합니다.
print(f"재현율(Recall): {recall:.2f}")  # 재현율을 출력합니다.
print(f"정밀도(Precision): {precision:.2f}")  # 정밀도를 출력합니다.
print(f"F1 Score: {f1_score:.2f}")  # F1 점수를 출력합니다.

cm = metrics.confusion_matrix(y_test, y_pred)  # 실제/예측 조합을 세어 오차 행렬을 만듭니다.
cm_df = pd.DataFrame(cm).rename(
    index={0: '실제값(정상:N)', 1: '실제값(이상:P)'},
    columns={0: '예측값(정상:N)', 1: '예측값(이상:P)'}
)  # 숫자 행렬에 읽기 쉬운 이름을 붙입니다.
print("\n=== [오차 행렬 (Confusion Matrix)] ===")  # 오차 행렬 제목입니다.
print(cm_df)  # 오차 행렬을 출력합니다.
with open(os.path.join(RESULTS_DIR, "step3_metrics.json"), "w", encoding="utf-8") as f:  # 평가 json 파일을 엽니다.
    json.dump(
        {
            "accuracy": float(accuracy),  # 정확도입니다.
            "recall": float(recall),  # 재현율입니다.
            "precision": float(precision),  # 정밀도입니다.
            "f1_score": float(f1_score),  # F1 점수입니다.
            "confusion_matrix": cm.tolist(),  # 오차 행렬입니다.
        },
        f,
        ensure_ascii=False,
        indent=2,
    )  # 평가 결과를 저장합니다.
cm_df.to_csv(os.path.join(RESULTS_DIR, "step3_confusion_matrix.csv"), index=True)  # 오차 행렬을 CSV로 저장합니다.

print("\n\n=== [개별 샘플 판단 근거 추적 실습 (Explainable AI)] ===")  # 설명 가능한 AI 실습 제목입니다.
feature_names = X_test.columns.tolist()  # 특징 이름 목록을 만듭니다.

normal_sample_idx = y_test[y_test == 0].index[0]  # 실제 정상인 첫 번째 샘플 번호입니다.
sample_normal_X = X_test.loc[[normal_sample_idx]]  # 그 샘플의 입력 특징만 가져옵니다.

print("\n🔵 [Case 1] 실제 '정상(OK)'인 샘플의 판별 과정 추적")  # 정상 샘플 분석 제목입니다.
explain_decision_path(Dtc_loaded, sample_normal_X, feature_names)  # 정상 샘플의 판단 경로를 출력합니다.

if len(y_test[y_test == 1]) > 0:  # 테스트 데이터에 이상 샘플이 있는지 확인합니다.
    error_sample_idx = y_test[y_test == 1].index[0]  # 실제 이상인 첫 번째 샘플 번호입니다.
    sample_error_X = X_test.loc[[error_sample_idx]]  # 그 샘플의 입력 특징만 가져옵니다.

    print("\n[Case 2] 실제 '이상(Error)'인 샘플의 판별 과정 추적")  # 이상 샘플 분석 제목입니다.
    explain_decision_path(Dtc_loaded, sample_error_X, feature_names)  # 이상 샘플의 판단 경로를 출력합니다.
else:  # 이상 샘플이 없으면 분석을 건너뜁니다.
    print("\n테스트 데이터셋에 '이상(Error)' 샘플이 존재하지 않아 Case 2 추적은 생략합니다.")

import matplotlib.pyplot as plt  # 틀린 샘플의 특징 비교 그래프를 그리기 위해 사용합니다.

print("\n\n=== [오분류 샘플 원인 분석 (왜 AI가 헷갈렸을까?)] ===")  # 오분류 분석 제목입니다.
misclassified_mask = y_test.values != y_pred  # 정답과 예측이 다른 위치를 True로 표시합니다.
misclassified_indices = y_test[misclassified_mask].index  # 틀린 샘플의 인덱스만 가져옵니다.
pd.DataFrame(
    {
        "index": X_test.index.tolist(),  # 테스트 데이터 인덱스입니다.
        "y_true": y_test.tolist(),  # 실제 정답입니다.
        "y_pred": y_pred.tolist(),  # 모델 예측입니다.
        "is_misclassified": misclassified_mask.tolist(),  # 틀렸는지 여부입니다.
    }
).to_csv(os.path.join(RESULTS_DIR, "step3_predictions.csv"), index=False)  # 모든 예측 결과를 CSV로 저장합니다.

if len(misclassified_indices) == 0:  # 틀린 샘플이 하나도 없으면 분석할 필요가 없습니다.
    print("모든 테스트 데이터를 정확하게 분류했습니다! 오분류된 샘플이 없습니다.")
else:  # 틀린 샘플이 있으면 왜 틀렸는지 살펴봅니다.
    print(f"총 {len(misclassified_indices)}개의 오분류 샘플이 발견되었습니다.")  # 틀린 개수를 알려줍니다.

    mean_normal = X_test[y_test.values == 0].mean()  # 정상 샘플들의 평균 특징입니다.
    mean_error = X_test[y_test.values == 1].mean() if len(y_test[y_test == 1]) > 0 else X_test.mean()  # 이상 평균입니다.

    for i, idx in enumerate(misclassified_indices[:2]):  # 출력이 너무 길어지지 않게 최대 2개만 봅니다.
        actual = "이상(Error)" if y_test.loc[idx] == 1 else "정상(OK)"  # 실제 정답을 글자로 바꿉니다.
        predicted = "이상(Error)" if y_pred[X_test.index.get_loc(idx)] == 1 else "정상(OK)"  # 예측을 글자로 바꿉니다.

        sample_X = X_test.loc[[idx]]  # 틀린 샘플 하나의 특징입니다.

        print(f"\n[분석 {i + 1}] 인덱스 {idx} 샘플 (실제: {actual} ➔ AI 예측: {predicted})")  # 분석 대상을 출력합니다.

        if test_filepaths is not None:  # 원본 파일 경로가 저장되어 있으면 보여줍니다.
            pos_idx = X_test.index.get_loc(idx)  # 현재 인덱스가 테스트 배열에서 몇 번째인지 찾습니다.
            original_wav_path = test_filepaths.iloc[pos_idx]  # 원본 wav 파일 경로입니다.
            print(f" 🎵 [원본 오디오 경로]: {original_wav_path}")  # 사람이 직접 들어볼 수 있게 경로를 보여줍니다.
            try:  # Windows 환경이면 자동 재생을 시도합니다.
                import winsound  # Windows 전용 소리 재생 도구입니다.
                print("[자동 재생]: 소리가 재생됩니다. (Windows 전용)")  # 재생 안내입니다.
                winsound.PlaySound(original_wav_path, winsound.SND_FILENAME)  # wav 파일을 재생합니다.
            except Exception:
                pass  # Windows가 아니거나 재생이 실패해도 분석은 계속합니다.

        explain_decision_path(Dtc_loaded, sample_X, feature_names)  # 틀린 샘플의 판단 경로를 출력합니다.

        plt.figure(figsize=(10, 5))  # 막대그래프 그림판을 만듭니다.
        x = np.arange(len(feature_names))  # 특징 개수만큼 x 위치를 만듭니다.
        width = 0.25  # 막대 너비입니다.

        plt.bar(x - width, mean_normal, width, label='Mean of Normal', color='blue', alpha=0.5)  # 정상 평균 막대입니다.
        plt.bar(x, mean_error, width, label='Mean of Error', color='red', alpha=0.5)  # 이상 평균 막대입니다.
        plt.bar(
            x + width,
            sample_X.iloc[0],
            width,
            label=f'This Sample (Actual: {actual})',
            color='orange',
            edgecolor='black',
            linewidth=2,
        )  # 틀린 샘플의 특징 막대입니다.

        plt.ylabel('Feature Value')  # y축 이름입니다.
        plt.title(f'Feature Comparison for Misclassified Sample #{idx}')  # 그래프 제목입니다.
        plt.xticks(x, feature_names)  # x축에 특징 이름을 붙입니다.
        plt.legend()  # 색깔 설명을 보여줍니다.

        save_path = os.path.join(VIS_DIR, f"step3_misclassified_sample_{idx}.png")  # 저장할 이미지 경로입니다.
        plt.savefig(save_path, dpi=200, bbox_inches='tight')  # 그래프를 저장합니다.
        plt.close()  # 그림을 닫아 메모리를 아낍니다.

        print(f"[시사점] 오분류된 샘플의 특징값 시각화 결과를 '{save_path}'에 저장했습니다.")  # 저장 위치를 알려줍니다.
        print("    ➔ 해당 샘플이 실제 정답보다 AI가 예측한 클래스의 평균과 더 비슷했는지 볼 수 있습니다.")  # 해석 방법입니다.
        print("    ➔ 원본 WAV 파일을 직접 들어보면 노이즈가 있거나 정상 소리와 비슷할 수 있습니다.")  # 왜 틀릴 수 있는지 설명합니다.

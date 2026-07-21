import pandas as pd
import numpy as np
import joblib
from sklearn import metrics
import os
import json

# =====================================================================
# 1. 샘플별 의사결정나무 판단 근거(Decision Path) 추적 함수 정의
# =====================================================================
def explain_decision_path(model, sample_X, feature_names):
    """
    단일 샘플이 의사결정나무의 어떤 조건들을 거쳐 최종 예측에 도달했는지 출력합니다.
    """
    # 샘플의 판단 경로와 각 노드의 정보 추출
    node_indicator = model.decision_path(sample_X)
    leaf_id = model.apply(sample_X)[0]
    
    # scikit-learn 트리 내부 구조체 접근
    feature = model.tree_.feature
    threshold = model.tree_.threshold
    
    # 경로에 포함된 노드 인덱스 추출
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    print(f"▶ 분석 대상 샘플의 특성 값: {sample_X.iloc[0].to_dict()}")
    print("-" * 50)
    print("🚦 [AI의 의사결정 흐름]")
    
    for node_id in node_index:
        # 리프 노드(최종 목적지)에 도달한 경우
        if leaf_id == node_id:
            pred_class = model.classes_[model.predict(sample_X)[0]]
            status = "이상(Error)" if pred_class == 1 else "정상(OK)"
            print(f" └── [최종 판별] 리프 노드({node_id}) 도달 ➔ 예측 결과: {status}")
            continue

        # 현재 노드의 분기 조건과 샘플의 실제 값 비교
        sample_value = sample_X.iloc[0, feature[node_id]]
        node_feature_name = feature_names[feature[node_id]]
        node_threshold = threshold[node_id]

        if sample_value <= node_threshold:
            threshold_sign = "<="
            direction = "왼쪽(True)"
        else:
            threshold_sign = ">"
            direction = "오른쪽(False)"
            
        print(f" ├── [노드 {node_id}] 조건: {node_feature_name} <= {node_threshold:.4f}")
        print(f" │   ↳ 샘플의 값({sample_value:.4f})은 {node_threshold:.4f} {threshold_sign} 이므로 {direction} 가지로 이동")

    print("-" * 50)


# =====================================================================
# 2. 저장된 모델 및 테스트 데이터 불러오기
# =====================================================================
print("저장된 모델과 테스트 데이터를 불러옵니다...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIS_DIR = os.path.join(BASE_DIR, "0_result")
RESULTS_DIR = os.path.join(BASE_DIR, "0_result")
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

try:
    Dtc_loaded = joblib.load(os.path.join(RESULTS_DIR, 'dtc_sound_model.pkl'))
    X_test = pd.read_csv(os.path.join(RESULTS_DIR, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(RESULTS_DIR, 'y_test.csv'))['NG']

    test_filepaths = None
    test_filepaths_path = os.path.join(RESULTS_DIR, 'test_filepaths.csv')
    if os.path.exists(test_filepaths_path):
        test_filepaths = pd.read_csv(test_filepaths_path)['filepath']
        
    print("불러오기 완료!\n")
except FileNotFoundError:
    print("오류: 모델 파일이나 데이터 파일을 찾을 수 없습니다. 학습 코드를 먼저 실행해주세요.")
    exit()

# =====================================================================
# 3. 테스트 데이터셋 전체 성능 평가
# =====================================================================
print("=== [모델 전체 성능 평가 지표] ===")
y_pred = Dtc_loaded.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
f1_score = metrics.f1_score(y_test, y_pred)

print(f"테스트 정확도(Accuracy): {accuracy:.2f}")
print(f"재현율(Recall): {recall:.2f}")
print(f"정밀도(Precision): {precision:.2f}")
print(f"F1 Score: {f1_score:.2f}")

cm = metrics.confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm).rename(
    index={0: '실제값(정상:N)', 1: '실제값(이상:P)'}, 
    columns={0: '예측값(정상:N)', 1: '예측값(이상:P)'}
)
print("\n=== [오차 행렬 (Confusion Matrix)] ===")
print(cm_df)
with open(os.path.join(RESULTS_DIR, "step3_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "accuracy": float(accuracy),
            "recall": float(recall),
            "precision": float(precision),
            "f1_score": float(f1_score),
            "confusion_matrix": cm.tolist(),
        },
        f,
        ensure_ascii=False,
        indent=2,
    )
cm_df.to_csv(os.path.join(RESULTS_DIR, "step3_confusion_matrix.csv"), index=True)


# =====================================================================
# 4. 개별 샘플에 대한 판단 근거(XAI) 확인 실습
# =====================================================================
print("\n\n=== [개별 샘플 판단 근거 추적 실습 (Explainable AI)] ===")
feature_names = X_test.columns.tolist()

# 실제 정답이 '정상(0)'인 샘플 중 하나를 무작위로 추출
normal_sample_idx = y_test[y_test == 0].index[0]
sample_normal_X = X_test.loc[[normal_sample_idx]]

print("\n🔵 [Case 1] 실제 '정상(OK)'인 샘플의 판별 과정 추적")
explain_decision_path(Dtc_loaded, sample_normal_X, feature_names)


# 실제 정답이 '이상(1)'인 샘플 중 하나를 무작위로 추출
# (만약 테스트 셋에 이상 샘플이 없다면 에러가 날 수 있으므로 예외 처리)
if len(y_test[y_test == 1]) > 0:
    error_sample_idx = y_test[y_test == 1].index[0]
    sample_error_X = X_test.loc[[error_sample_idx]]
    
    print("\n[Case 2] 실제 '이상(Error)'인 샘플의 판별 과정 추적")
    explain_decision_path(Dtc_loaded, sample_error_X, feature_names)
else:
    print("\n테스트 데이터셋에 '이상(Error)' 샘플이 존재하지 않아 Case 2 추적은 생략합니다.")


# =====================================================================
# 5. 오분류(못맞춘) 샘플 원인 심층 분석 및 시각화
# =====================================================================
import matplotlib.pyplot as plt

print("\n\n=== [오분류 샘플 원인 분석 (왜 AI가 헷갈렸을까?)] ===")
# 오분류된 인덱스 찾기: y_test와 y_pred가 다른 경우
# y_test는 Series, y_pred는 numpy array
misclassified_mask = y_test.values != y_pred
misclassified_indices = y_test[misclassified_mask].index
pd.DataFrame(
    {
        "index": X_test.index.tolist(),
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "is_misclassified": misclassified_mask.tolist(),
    }
).to_csv(os.path.join(RESULTS_DIR, "step3_predictions.csv"), index=False)

if len(misclassified_indices) == 0:
    print("모든 테스트 데이터를 정확하게 분류했습니다! 오분류된 샘플이 없습니다.")
else:
    print(f"총 {len(misclassified_indices)}개의 오분류 샘플이 발견되었습니다.")
    
    # 정상 데이터와 이상 데이터의 특성 평균(분포) 계산 (비교용)
    mean_normal = X_test[y_test.values == 0].mean()
    mean_error = X_test[y_test.values == 1].mean() if len(y_test[y_test == 1]) > 0 else X_test.mean()

    # 오분류된 샘플 중 최대 2개만 분석 (과도한 출력 방지)
    for i, idx in enumerate(misclassified_indices[:2]):
        actual = "이상(Error)" if y_test.loc[idx] == 1 else "정상(OK)"
        predicted = "이상(Error)" if y_pred[X_test.index.get_loc(idx)] == 1 else "정상(OK)"
        
        sample_X = X_test.loc[[idx]]
        
        print(f"\n[분석 {i+1}] 인덱스 {idx} 샘플 (실제: {actual} ➔ AI 예측: {predicted})")
        
        if test_filepaths is not None:
            pos_idx = X_test.index.get_loc(idx)
            original_wav_path = test_filepaths.iloc[pos_idx]
            print(f" 🎵 [원본 오디오 경로]: {original_wav_path}")
            try:
                import winsound
                print(f"[자동 재생]: 소리가 재생됩니다. (Windows 전용)")
                winsound.PlaySound(original_wav_path, winsound.SND_FILENAME)
            except Exception as e:
                pass
            
        # 1) 어떤 경로에서 잘못 판단했는지 추적
        explain_decision_path(Dtc_loaded, sample_X, feature_names)
        
        # 2) 시각화로 '왜 헷갈렸는지' 확인 (데이터 분포와 비교)
        # 특징 벡터(Feature Vector)가 정상 평균에 가까운지, 이상 평균에 가까운지 Bar 그래프로 시각화
        plt.figure(figsize=(10, 5))
        x = np.arange(len(feature_names))
        width = 0.25
        
        plt.bar(x - width, mean_normal, width, label='Mean of Normal', color='blue', alpha=0.5)
        plt.bar(x, mean_error, width, label='Mean of Error', color='red', alpha=0.5)
        plt.bar(x + width, sample_X.iloc[0], width, label=f'This Sample (Actual: {actual})', color='orange', edgecolor='black', linewidth=2)
        
        plt.ylabel('Feature Value')
        plt.title(f'Feature Comparison for Misclassified Sample #{idx}')
        plt.xticks(x, feature_names)
        plt.legend()
        
        save_path = os.path.join(VIS_DIR, f"step3_misclassified_sample_{idx}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"[시사점] 오분류된 샘플의 특징값 시각화 결과를 '{save_path}'에 저장했습니다.")
        print(f"    ➔ 위 이미지를 확인해보면, 해당 샘플의 오디오 특성이 실제 정답보다 AI가 예측한 클래스의 특성과 얼마나 비슷했는지 시각적으로 확인할 수 있습니다.")
        print(f"    ➔ (참고) 원본 WAV 파일의 소리를 직접 들어보면 노이즈가 끼어 있거나, 정상 소리와 매우 유사하게 들릴 가능성이 높습니다.")

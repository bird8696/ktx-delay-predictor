import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import sys

sys.path.append(".")
from api.korail import get_train_plan, get_train_info
from pipeline.preprocess import TrainPreprocessor


# ── 1. 피처 & 타겟 정의 ───────────────────────────────────
# ML 모델에 입력할 피처 컬럼 목록
FEATURE_COLS = [
    "plan_arvl_min",        # 예정 도착 시각 (분)
    "plan_dptre_min",       # 예정 출발 시각 (분)
    "travel_min",           # 예정 소요 시간 (분)
    "time_slot",            # 시간대 (0=새벽 ~ 5=심야)
    "day_of_week",          # 요일 (0=월요일 ~ 6=일요일)
    "is_weekend",           # 주말 여부 (1=주말, 0=평일)
    "arvl_stn_nm_enc",      # 도착역 이름 (LabelEncoder 인코딩)
    "dptre_stn_nm_enc",     # 출발역 이름 (LabelEncoder 인코딩)
    "mrnt_nm_enc",          # 노선명 (LabelEncoder 인코딩)
    "uppln_dn_se_cd_enc",   # 상행/하행 구분 (LabelEncoder 인코딩)
]

# 예측할 타겟 컬럼
# 0 = 정상 (5분 미만)
# 1 = 소지연 (5~15분)
# 2 = 대지연 (15분 이상)
TARGET_COL = "delay_label"


# ── 2. 데이터 준비 ────────────────────────────────────────
def prepare_data() -> tuple:
    """
    API에서 데이터 수집 후 전처리까지 완료된 X, y 반환
    - X: 피처 DataFrame
    - y: 타겟(지연 레이블) Series
    - preprocessor: 저장된 전처리기 객체
    """
    print("=== 데이터 수집 중 ===")
    # 운행계획 (예정 시각) + 운행정보 (실제 시각) 수집
    df_plan = get_train_plan()
    df_info = get_train_info()

    print("\n=== 전처리 중 ===")
    preprocessor = TrainPreprocessor()
    # fit=True: 스케일러, 인코더를 이 데이터 기준으로 학습
    df = preprocessor.run(df_plan, df_info, fit=True)

    # 정의한 피처 중 실제로 DataFrame에 존재하는 것만 사용
    # (API 응답에 따라 일부 컬럼이 없을 수 있음)
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    print(f"\n[피처] 사용할 피처 수: {len(available_features)}")
    print(f"[피처] 목록: {available_features}")

    X = df[available_features]
    y = df[TARGET_COL]

    # 전처리기 저장 (예측 시 동일한 기준으로 변환하기 위해)
    preprocessor.save()

    return X, y, preprocessor


# ── 3. RandomForest 학습 ──────────────────────────────────
def train_random_forest(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """
    RandomForestClassifier로 열차 지연 예측 모델 학습
    - 여러 개의 결정 트리를 앙상블해서 예측 성능 향상
    - 과적합에 강하고 피처 중요도 확인 가능
    - 클래스 불균형 (정상이 압도적으로 많음) 자동 보정
    """
    print("\n=== RandomForest 학습 ===")

    # 학습/테스트 데이터 8:2 비율로 분리
    # stratify=y: 분리 후에도 클래스 비율 유지
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,    # 동일한 결과 재현을 위한 시드값 고정
        stratify=y
    )
    print(f"[분리] 학습: {len(X_train)}행 / 테스트: {len(X_test)}행")

    # 모델 정의
    model = RandomForestClassifier(
        n_estimators=200,       # 트리 수 증가
        max_depth=15,           # 깊이 증가
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,     # 과적합 방지
        n_jobs=-1               # 병렬 처리로 속도 향상
    )

    # 학습 실행
    model.fit(X_train, y_train)

    # 테스트 데이터로 성능 평가
    y_pred = model.predict(X_test)

    # 실제 데이터에 존재하는 클래스만 target_names에 포함
    # (테스트셋에 "대지연" 클래스가 없을 수도 있어서 동적으로 생성)
    label_map = {0: "정상", 1: "소지연", 2: "대지연"}
    target_names = [label_map[i] for i in sorted(y.unique())]

    print("\n=== 모델 성능 평가 ===")
    print(classification_report(
        y_test, y_pred,
        target_names=target_names,
        zero_division=0         # 예측값이 없는 클래스는 0으로 처리
    ))

    # 피처 중요도 출력 (높을수록 예측에 중요한 피처)
    feature_importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\n=== 피처 중요도 ===")
    print(feature_importance)

    return model


# ── 4. Isolation Forest 이상치 탐지 ──────────────────────
def train_isolation_forest(X: pd.DataFrame) -> IsolationForest:
    """
    IsolationForest로 비정상적인 운행 패턴 탐지
    - 레이블 없이 비지도학습으로 이상치 탐지 (비지도학습 활용)
    - 정상 패턴과 다른 열차를 자동으로 감지
    - 예: 특정 구간에서 갑작스럽게 지연이 폭발적으로 늘어나는 경우
    """
    print("\n=== Isolation Forest 학습 ===")

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # 전체 데이터 중 5%를 이상치로 가정
        random_state=42
    )
    model.fit(X)

    # 이상치 예측 결과 확인
    # 1 = 정상, -1 = 이상치
    predictions = model.predict(X)
    anomaly_count = (predictions == -1).sum()
    print(f"[이상치 탐지] 전체 {len(X)}행 중 {anomaly_count}개 이상치 감지")

    return model


# ── 5. 모델 저장 ──────────────────────────────────────────
def save_models(rf_model, if_model, save_dir: str = "models/saved"):
    """
    학습된 모델을 joblib 파일로 저장
    - 나중에 Streamlit 대시보드에서 불러와서 실시간 예측에 사용
    """
    os.makedirs(save_dir, exist_ok=True)

    # RandomForest 저장
    rf_path = os.path.join(save_dir, "random_forest.joblib")
    joblib.dump(rf_model, rf_path)
    print(f"[저장] RandomForest → {rf_path}")

    # Isolation Forest 저장
    if_path = os.path.join(save_dir, "isolation_forest.joblib")
    joblib.dump(if_model, if_path)
    print(f"[저장] IsolationForest → {if_path}")


# ── 6. 모델 로드 & 예측 ───────────────────────────────────
def load_and_predict(X_new: pd.DataFrame) -> dict:
    """
    저장된 모델 불러와서 새로운 데이터 예측
    Streamlit 대시보드에서 실시간 예측할 때 이 함수를 호출

    반환값:
    - delay_label: 지연 예측 결과 텍스트 리스트
    - delay_proba: 각 클래스별 확률 (정상/소지연/대지연)
    - is_anomaly: 이상치 여부 리스트
    """
    # 저장된 모델 불러오기
    rf_model = joblib.load("models/saved/random_forest.joblib")
    if_model = joblib.load("models/saved/isolation_forest.joblib")

    # 지연 예측 (클래스 + 확률)
    delay_pred = rf_model.predict(X_new)
    delay_proba = rf_model.predict_proba(X_new)

    # 이상치 탐지 (1=정상, -1=이상치)
    anomaly_pred = if_model.predict(X_new)

    # 숫자 레이블 → 텍스트 변환
    label_map = {0: "정상", 1: "소지연 (5~15분)", 2: "대지연 (15분 이상)"}

    return {
        "delay_label": [label_map[p] for p in delay_pred],
        "delay_proba": delay_proba.tolist(),
        "is_anomaly": [p == -1 for p in anomaly_pred]
    }


# ── 메인 실행 ─────────────────────────────────────────────
# 이 파일을 직접 실행할 때만 아래 코드 동작
# Streamlit 등에서 import할 때는 실행 안 됨
if __name__ == "__main__":
    # 데이터 준비 (수집 + 전처리)
    X, y, preprocessor = prepare_data()

    # 모델 학습
    rf_model = train_random_forest(X, y)
    if_model = train_isolation_forest(X)

    # 모델 저장
    save_models(rf_model, if_model)

    print("\n=== 완료 ===")
    print("models/saved/ 폴더에 모델 저장됨")
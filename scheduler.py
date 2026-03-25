import pandas as pd
import os
import sys
from datetime import datetime, timedelta

sys.path.append(".")
from api.korail import get_train_plan, get_train_info
from pipeline.preprocess import TrainPreprocessor
from models.train_model import FEATURE_COLS, TARGET_COL, train_random_forest, train_isolation_forest, save_models

# 누적 데이터 저장 경로
DATA_PATH = "data/train_history.csv"


# ── 1. 전날 운행 완료 데이터 수집 ─────────────────────────
def collect_yesterday_data() -> pd.DataFrame:
    """
    최근 7일치 데이터 수집
    API 호출 7회로 과거 데이터 한번에 수집
    """
    all_dfs = []

    for i in range(1, 8):  # 1일 전 ~ 7일 전
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        print(f"[수집] {date} 데이터 수집 중...")

        df_plan = get_train_plan(dep_date=date)
        df_info = get_train_info(dep_date=date)

        if df_plan.empty or df_info.empty:
            print(f"[수집] {date} 데이터 없음, 건너뜀")
            continue

        # 전처리 실행
        preprocessor = TrainPreprocessor()
        df = preprocessor.run(df_plan, df_info, fit=True)

        # 실제 도착 시각 있는 것만 유효 데이터
        if "real_arvl_min" in df.columns:
            df = df[df["real_arvl_min"].notna()]

        # 수집 날짜 컬럼 추가
        df["collected_date"] = date
        all_dfs.append(df)
        print(f"[수집] {date} → {len(df)}행")

    if not all_dfs:
        print("[수집] 수집된 데이터 없음")
        return pd.DataFrame()

    df_combined = pd.concat(all_dfs, ignore_index=True)
    print(f"[수집] 전체 {len(df_combined)}행 수집 완료")
    return df_combined


# ── 2. 누적 데이터에 추가 저장 ────────────────────────────
def save_to_history(df_new: pd.DataFrame):
    """
    새로 수집한 데이터를 기존 누적 CSV에 추가
    파일 없으면 새로 생성
    """
    os.makedirs("data", exist_ok=True)

    if df_new.empty:
        print("[저장] 저장할 데이터 없음")
        return

    if os.path.exists(DATA_PATH):
        # 기존 데이터 불러와서 합치기
        df_existing = pd.read_csv(DATA_PATH)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)

        # 중복 제거 (같은 날짜 + 열차번호)
        if "trn_no" in df_combined.columns and "collected_date" in df_combined.columns:
            df_combined = df_combined.drop_duplicates(
                subset=["trn_no", "collected_date"]
            ).reset_index(drop=True)
    else:
        df_combined = df_new

    df_combined.to_csv(DATA_PATH, index=False)
    print(f"[저장] 누적 데이터: {len(df_combined)}행 → {DATA_PATH}")


# ── 3. 누적 데이터로 모델 재학습 ──────────────────────────
def retrain_from_history():
    """
    누적된 CSV 데이터로 모델 재학습
    데이터가 적으면 학습 건너뜀 (최소 50행 필요)
    """
    if not os.path.exists(DATA_PATH):
        print("[재학습] 누적 데이터 없음, 건너뜀")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"[재학습] 누적 데이터 {len(df)}행으로 재학습 시작")

    # 최소 데이터 기준 미달 시 건너뜀
    if len(df) < 50:
        print(f"[재학습] 데이터 부족 ({len(df)}행 < 50행), 건너뜀")
        return

    # 피처 & 타겟 준비
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    if TARGET_COL not in df.columns:
        print("[재학습] 타겟 컬럼 없음, 건너뜀")
        return

    X = df[available_features]
    y = df[TARGET_COL]

    print(f"[재학습] 피처: {available_features}")
    print(f"[재학습] 타겟 분포:\n{y.value_counts()}")

    # 모델 재학습 & 저장
    rf_model = train_random_forest(X, y)
    if_model = train_isolation_forest(X)
    save_models(rf_model, if_model)

    # 전처리기도 재저장
    preprocessor = TrainPreprocessor()
    preprocessor.save()
    print("[재학습] 완료")


# ── 메인 실행 ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"=== 스케줄러 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    # 1. 전날 데이터 수집
    df_new = collect_yesterday_data()

    # 2. 누적 저장
    save_to_history(df_new)

    # 3. 재학습
    retrain_from_history()

    print(f"=== 스케줄러 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
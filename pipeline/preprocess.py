import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
import joblib
import os


class TrainPreprocessor:
    def __init__(self):
        # 수치형 컬럼 정규화용 스케일러 (평균 0, 표준편차 1로 변환)
        self.scaler = StandardScaler()
        # 범주형 컬럼 인코딩용 딕셔너리 (컬럼별로 따로 LabelEncoder 관리)
        self.label_encoders = {}

    # ── 1. 두 API 데이터 병합 ──────────────────────────────
    def merge_plan_and_info(self, df_plan: pd.DataFrame, df_info: pd.DataFrame) -> pd.DataFrame:
        """
        운행계획(예정 시각) + 운행정보(실제 시각)를
        열차번호(trn_no) 기준으로 병합 → 지연 시간 계산 가능하게 만듦
        - 두 API가 서로 다른 날짜 데이터를 반환하는 경우가 있어서
          run_ymd 조건 제거하고 trn_no만으로 매칭
        """
        df = pd.merge(
            df_plan,
            df_info,
            on="trn_no",        # 열차번호 기준으로 병합
            how="inner",        # 양쪽에 모두 있는 데이터만 유지
            suffixes=("_plan", "_info")  # 같은 컬럼명 충돌 시 구분자 추가
        )
        print(f"[병합] 운행계획 {len(df_plan)}행 + 운행정보 {len(df_info)}행 → {len(df)}행")
        return df

    # ── 2. 시각 컬럼 파싱 ─────────────────────────────────
    def parse_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        날짜/시각 문자열 → datetime 변환 후 분 단위 정수 파생 컬럼 생성
        예) '2026-03-24 05:13:00.0' → 313 (분)
        분 단위로 변환하는 이유: ML 모델이 시각 데이터를 숫자로 처리해야 하기 때문
        """
        # 변환할 시각 컬럼 목록과 파생 컬럼명 매핑
        time_cols = {
            "trn_plan_arvl_dt": "plan_arvl_min",   # 예정 도착 시각 (분)
            "trn_plan_dptre_dt": "plan_dptre_min",  # 예정 출발 시각 (분)
            "trn_arvl_dt": "real_arvl_min",         # 실제 도착 시각 (분)
            "trn_dptre_dt": "real_dptre_min",       # 실제 출발 시각 (분)
        }

        for col, new_col in time_cols.items():
            if col in df.columns:
                # 문자열 → datetime (변환 실패 시 NaT으로 처리)
                df[col] = pd.to_datetime(df[col], errors="coerce")
                # 시각을 분 단위로 변환 (예: 05:13 → 313분)
                df[new_col] = df[col].dt.hour * 60 + df[col].dt.minute

        # 운행 날짜 컬럼 파싱
        # 병합 후 run_ymd가 run_ymd_plan으로 바뀌므로 존재하는 컬럼 사용
        run_col = "run_ymd_plan" if "run_ymd_plan" in df.columns else "run_ymd"
        df[run_col] = pd.to_datetime(df[run_col], format="%Y%m%d", errors="coerce")

        # 요일 추출 (0=월요일 ~ 6=일요일) → 주말/평일 패턴 분석에 사용
        df["day_of_week"] = df[run_col].dt.dayofweek

        print(f"[시각 파싱] 완료")
        return df

    # ── 3. 지연 시간 계산 ─────────────────────────────────
    def calculate_delay(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        실제 도착 시각 - 예정 도착 시각 = 지연 시간(분)
        지연 분류 레이블도 함께 생성 (ML 모델의 타겟 변수)
        """
        # 지연 시간 계산 (음수면 일찍 도착한 것)
        df["delay_min"] = df["real_arvl_min"] - df["plan_arvl_min"]

        # 지연 분류 레이블 생성
        # 0: 정상 (5분 미만)
        # 1: 소지연 (5~15분)
        # 2: 대지연 (15분 이상)
        conditions = [
            df["delay_min"] < 5,
            df["delay_min"].between(5, 15),
            df["delay_min"] > 15
        ]
        df["delay_label"] = np.select(conditions, [0, 1, 2], default=0)

        print(f"[지연 계산] 평균 지연: {df['delay_min'].mean():.1f}분")
        print(f"[지연 분포]\n{df['delay_label'].value_counts()}")
        return df

    # ── 4. 결측값 처리 ────────────────────────────────────
    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        컬럼 특성에 맞게 결측값 처리
        - 시각 컬럼: 선형 보간 (앞뒤 값의 중간값으로 채움)
        - 지연 시간: KNN Imputer (비슷한 패턴의 열차 데이터로 채움)
        - 역명 등 범주형: 최빈값 (가장 많이 등장하는 값으로 채움)
        """
        print(f"[결측값] 처리 전:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

        # 시각 관련 수치 컬럼 → 선형 보간
        time_min_cols = ["plan_arvl_min", "plan_dptre_min", "real_arvl_min", "real_dptre_min"]
        # 실제로 존재하는 컬럼만 필터링
        time_min_cols = [c for c in time_min_cols if c in df.columns]
        df[time_min_cols] = df[time_min_cols].interpolate(method="linear", limit_direction="both")

        # 지연 시간 → KNN Imputer (주변 열차 패턴 참고해서 채움)
        if "delay_min" in df.columns and df["delay_min"].isnull().sum() > 0:
            imputer = KNNImputer(n_neighbors=5)  # 가장 유사한 5개 열차 참고
            df[["delay_min"]] = imputer.fit_transform(df[["delay_min"]])

        # 역명 등 범주형 컬럼 → 최빈값으로 대체
        cat_cols = ["arvl_stn_nm", "dptre_stn_nm", "stn_nm", "mrnt_nm"]
        for col in [c for c in cat_cols if c in df.columns]:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else "UNKNOWN")

        print(f"[결측값] 처리 후 잔여: {df.isnull().sum().sum()}")
        return df

    # ── 5. 이상치 제거 ────────────────────────────────────
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        IQR 방식으로 지연 시간 이상치 제거
        IQR = Q3 - Q1 (중간 50% 범위)
        정상 범위: Q1 - 1.5*IQR ~ Q3 + 1.5*IQR
        이 범위 밖의 극단적 지연 데이터는 모델 학습에 악영향을 주므로 제거
        """
        before = len(df)

        # 사분위수 계산
        Q1 = df["delay_min"].quantile(0.25)   # 하위 25%
        Q3 = df["delay_min"].quantile(0.75)   # 상위 75%
        IQR = Q3 - Q1                          # 중간 50% 범위

        # 정상 범위 경계값
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # 정상 범위 데이터만 유지
        df = df[df["delay_min"].between(lower, upper)].reset_index(drop=True)
        print(f"[이상치] {before - len(df)}행 제거 → {len(df)}행 남음 (범위: {lower:.1f} ~ {upper:.1f}분)")
        return df

    # ── 6. 피처 엔지니어링 ────────────────────────────────
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ML 모델 성능 향상을 위한 파생 변수 생성
        원본 데이터에서 의미 있는 새로운 특징을 추출
        """
        # 예정 소요 시간 (도착 예정 - 출발 예정)
        if "plan_arvl_min" in df.columns and "plan_dptre_min" in df.columns:
            df["travel_min"] = df["plan_arvl_min"] - df["plan_dptre_min"]

        # 시간대 구분 (출발 시각 기준)
        # 0=새벽, 1=출근, 2=오전, 3=오후, 4=저녁, 5=심야
        if "plan_dptre_min" in df.columns:
            hour = (df["plan_dptre_min"] // 60).astype(int) % 24
            df["time_slot"] = pd.cut(
                hour,
                bins=[-1, 6, 9, 12, 18, 21, 24],
                labels=[0, 1, 2, 3, 4, 5]
            ).astype(int)

        # 주말 여부 (1=주말, 0=평일) → 주말에 지연이 더 많은지 파악
        if "day_of_week" in df.columns:
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        print(f"[피처 엔지니어링] 완료 → 최종 컬럼 수: {len(df.columns)}")
        return df

    # ── 7. 범주형 인코딩 & 수치형 정규화 ──────────────────
    def encode_and_scale(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        ML 모델은 문자열을 직접 처리 못하므로 숫자로 변환 필요
        - 범주형(역명 등) → LabelEncoder로 숫자 변환 (서울=0, 부산=1 등)
        - 수치형(시각 등) → StandardScaler로 정규화 (평균 0, 표준편차 1)
        fit=True: 학습 데이터로 기준 학습 후 변환
        fit=False: 기존 기준으로만 변환 (예측 시 사용)
        """
        # 범주형 컬럼 인코딩
        cat_cols = ["arvl_stn_nm", "dptre_stn_nm", "mrnt_nm", "uppln_dn_se_cd"]
        cat_cols = [c for c in cat_cols if c in df.columns]

        for col in cat_cols:
            # 컬럼별 LabelEncoder 없으면 새로 생성
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()

            if fit:
                # 학습 시: 레이블 학습 후 변환
                df[col + "_enc"] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # 예측 시: 학습 때 못 본 새로운 역명은 UNKNOWN으로 처리
                known = set(self.label_encoders[col].classes_)
                df[col] = df[col].apply(lambda x: x if x in known else "UNKNOWN")
                df[col + "_enc"] = self.label_encoders[col].transform(df[col].astype(str))

        # 수치형 컬럼 정규화
        scale_cols = ["plan_arvl_min", "plan_dptre_min", "travel_min"]
        scale_cols = [c for c in scale_cols if c in df.columns]

        if scale_cols:
            if fit:
                df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
            else:
                df[scale_cols] = self.scaler.transform(df[scale_cols])

        print(f"[인코딩/정규화] 완료")
        return df

    # ── 8. 전체 파이프라인 실행 ───────────────────────────
    def run(self, df_plan: pd.DataFrame, df_info: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        1~7단계 전체를 순서대로 실행하는 메인 함수
        fit=True: 학습 데이터 처리 시
        fit=False: 새로운 데이터 예측 시
        """
        df = self.merge_plan_and_info(df_plan, df_info)
        df = self.parse_datetime_columns(df)
        df = self.calculate_delay(df)
        df = self.handle_missing(df)
        df = self.remove_outliers(df)
        df = self.engineer_features(df)
        df = self.encode_and_scale(df, fit=fit)
        print(f"\n[전처리 완료] 최종 데이터: {df.shape}")
        return df

    # ── 9. 전처리기 저장/로드 ─────────────────────────────
    def save(self, path: str = "models/saved/preprocessor.joblib"):
        """
        학습된 스케일러, 인코더를 파일로 저장
        나중에 예측할 때 동일한 기준으로 변환하기 위해 필요
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[저장] {path}")

    @staticmethod
    def load(path: str = "models/saved/preprocessor.joblib"):
        """저장된 전처리기 불러오기"""
        return joblib.load(path)


# 이 파일을 직접 실행할 때만 아래 코드 동작
# 다른 파일에서 import할 때는 실행 안 됨
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from api.korail import get_train_plan, get_train_info

    # API에서 데이터 수집
    df_plan = get_train_plan()
    df_info = get_train_info()

    # 전처리 파이프라인 실행
    preprocessor = TrainPreprocessor()
    df_final = preprocessor.run(df_plan, df_info)

    print("\n=== 최종 데이터 샘플 ===")
    print(df_final.head(3))
    print(f"\n=== 사용 가능한 피처 ===")
    print(df_final.columns.tolist())
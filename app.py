import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sys
import time
from datetime import datetime

sys.path.append(".")
from api.korail import get_train_plan, get_train_info
from pipeline.preprocess import TrainPreprocessor
from models.train_model import FEATURE_COLS, load_and_predict

# ── 페이지 기본 설정 ──────────────────────────────────────
st.set_page_config(
    page_title="KTX 지연 예측 대시보드",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 모델 & 전처리기 로드 ──────────────────────────────────
@st.cache_resource
def load_models():
    """
    저장된 모델과 전처리기를 한 번만 로드해서 캐싱
    @st.cache_resource: 앱이 실행되는 동안 한 번만 로드 (매 요청마다 로드 방지)
    """
    try:
        rf_model = joblib.load("models/saved/random_forest.joblib")
        if_model = joblib.load("models/saved/isolation_forest.joblib")
        preprocessor = TrainPreprocessor.load("models/saved/preprocessor.joblib")
        return rf_model, if_model, preprocessor
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None, None, None


# ── 데이터 수집 & 전처리 ──────────────────────────────────
@st.cache_data(ttl=30)
def fetch_data():
    """
    API에서 실시간 데이터 수집 후 전처리 완료
    @st.cache_data(ttl=30): 30초마다 캐시 자동 갱신 (준실시간 효과)
    저장된 전처리기를 불러와서 학습 때와 동일한 기준으로 변환
    """
    try:
        df_plan = get_train_plan()
        df_info = get_train_info()

        try:
            # 저장된 전처리기 사용 (학습 때와 동일한 인코더/스케일러 기준 적용)
            preprocessor = TrainPreprocessor.load("models/saved/preprocessor.joblib")
            df = preprocessor.run(df_plan, df_info, fit=False)
        except Exception:
            # 저장된 전처리기 없으면 새로 fit (첫 실행 시)
            preprocessor = TrainPreprocessor()
            df = preprocessor.run(df_plan, df_info, fit=True)

        return df, df_plan, df_info

    except Exception as e:
        st.error(f"데이터 수집 실패: {e}")
        return None, None, None


# ── 지연 레이블 변환 헬퍼 함수 ───────────────────────────
def label_to_text(label: int) -> str:
    """숫자 레이블 → 이모지 텍스트 변환"""
    return {0: "✅ 정상", 1: "⚠️ 소지연", 2: "🚨 대지연"}.get(label, "알 수 없음")

def label_to_color(label: int) -> str:
    """숫자 레이블 → 색상 변환"""
    return {0: "green", 1: "orange", 2: "red"}.get(label, "gray")


# ── 사이드바 ──────────────────────────────────────────────
with st.sidebar:
    st.title("🚄 KTX 지연 예측")
    st.markdown("---")

    # 자동 갱신 토글 (30초마다 자동으로 데이터 갱신)
    auto_refresh = st.toggle("자동 갱신 (30초)", value=False)
    if auto_refresh:
        st.success("자동 갱신 활성화")

    st.markdown("---")

    # 수동 갱신 버튼 (캐시 초기화 후 즉시 갱신)
    if st.button("🔄 지금 갱신", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    from datetime import timezone, timedelta
    KST = timezone(timedelta(hours=9))
    st.caption(f"마지막 갱신: {datetime.now(KST).strftime('%H:%M:%S')} (KST)")


# ── 메인 타이틀 ───────────────────────────────────────────
st.title("🚄 KTX 열차 지연 예측 대시보드")
st.markdown("코레일 실시간 데이터 기반 ML 지연 예측 시스템")
st.markdown("---")


# ── 데이터 & 모델 로드 ────────────────────────────────────
rf_model, if_model, preprocessor = load_models()
df, df_plan, df_info = fetch_data()

# 데이터 또는 모델 로드 실패 시 안내 메시지 출력 후 중단
if df is None or rf_model is None:
    st.error("데이터 또는 모델을 불러올 수 없어요. 모델 학습을 먼저 실행해주세요.")
    st.code("python models/train_model.py")
    st.stop()


# ── 피처 준비 & 예측 실행 ─────────────────────────────────
# 정의된 피처 중 실제로 DataFrame에 존재하는 컬럼만 사용
available_features = [c for c in FEATURE_COLS if c in df.columns]
X = df[available_features]

# RandomForest로 지연 예측 (0=정상, 1=소지연, 2=대지연)
delay_pred = rf_model.predict(X)
delay_proba = rf_model.predict_proba(X)

# Isolation Forest로 이상치 탐지 (1=정상, -1=이상치)
anomaly_pred = if_model.predict(X)

# 예측 결과를 DataFrame에 컬럼으로 추가
df["예측_지연"] = delay_pred
df["이상치"] = anomaly_pred == -1


# ── 상단 요약 카드 (4개) ──────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

total   = len(df)
normal  = (delay_pred == 0).sum()
delay   = (delay_pred >= 1).sum()
anomaly = (anomaly_pred == -1).sum()

with col1:
    st.metric("전체 열차", f"{total}개")
with col2:
    st.metric("정상 운행", f"{normal}개", delta=f"{normal/total*100:.0f}%")
with col3:
    st.metric("지연 예측", f"{delay}개",
              delta=f"-{delay}개" if delay > 0 else "0개",
              delta_color="inverse")
with col4:
    st.metric("이상치 탐지", f"{anomaly}개", delta_color="inverse")

st.markdown("---")


# ── 탭 구성 ───────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📋 열차 현황", "📊 분석", "🔍 이상치", "🤖 AI 질의"])


# ── 탭 1: 열차 현황 테이블 ────────────────────────────────
with tab1:
    st.subheader("실시간 열차 현황")

    # 표시할 컬럼 한글명 매핑 (지연(분) 제거)
    col_map = {
        "trn_no":            "열차번호",
        "dptre_stn_nm":      "출발역",
        "arvl_stn_nm":       "도착역",
        "trn_plan_dptre_dt": "예정 출발",
        "trn_plan_arvl_dt":  "예정 도착",
        "예측_지연":          "지연 예측",
        "이상치":             "이상치"
    }

    # 실제로 존재하는 컬럼만 필터링해서 표시
    exist_cols = {k: v for k, v in col_map.items() if k in df.columns}
    df_display = df[list(exist_cols.keys())].copy()
    df_display.columns = list(exist_cols.values())

    # 숫자 레이블 → 이모지 텍스트 변환
    df_display["지연 예측"] = df_display["지연 예측"].map(
        {0: "✅ 정상", 1: "⚠️ 소지연", 2: "🚨 대지연"}
    )
    df_display["이상치"] = df_display["이상치"].map(
        {True: "🚨 이상", False: "-"}
    )

    st.dataframe(df_display, use_container_width=True, height=400)


# ── 탭 2: 분석 차트 ───────────────────────────────────────
with tab2:
    st.subheader("지연 예측 분포")

    col_a, col_b = st.columns(2)

    with col_a:
        # 지연 예측 분포 파이차트
        label_counts = pd.Series(delay_pred).map(
            {0: "정상", 1: "소지연", 2: "대지연"}
        ).value_counts()

        fig_pie = px.pie(
            values=label_counts.values,
            names=label_counts.index,
            title="지연 예측 분포",
            color_discrete_map={
                "정상": "#2ecc71",
                "소지연": "#f39c12",
                "대지연": "#e74c3c"
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        # 시간대별 평균 지연 예측 막대차트
        if "time_slot" in df.columns:
            slot_map = {0: "새벽", 1: "출근", 2: "오전", 3: "오후", 4: "저녁", 5: "심야"}
            df["시간대"] = df["time_slot"].map(slot_map)
            slot_delay = df.groupby("시간대")["예측_지연"].mean().reset_index()

            fig_bar = px.bar(
                slot_delay,
                x="시간대",
                y="예측_지연",
                title="시간대별 평균 지연 예측",
                color="예측_지연",
                color_continuous_scale="RdYlGn_r"
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# ── 탭 3: 이상치 탐지 ─────────────────────────────────────
with tab3:
    st.subheader("이상치 탐지 결과")

    # 이상치로 탐지된 열차만 필터링
    anomaly_df = df[df["이상치"] == True]

    if len(anomaly_df) == 0:
        st.info("현재 탐지된 이상치가 없어요.")
    else:
        st.warning(f"⚠️ {len(anomaly_df)}개 이상치 탐지됨")

        # 이상치 열차 테이블 출력
        exist_cols_a = {k: v for k, v in col_map.items() if k in anomaly_df.columns}
        df_anomaly_display = anomaly_df[list(exist_cols_a.keys())].copy()
        df_anomaly_display.columns = list(exist_cols_a.values())
        st.dataframe(df_anomaly_display, use_container_width=True)

    # 이상치 스코어 분포 (정상/이상치 색상 구분)
    anomaly_scores = if_model.decision_function(X)

    fig_score = go.Figure()

    # 정상 구간 (스코어 >= 0)
    normal_scores = anomaly_scores[anomaly_scores >= 0]
    if len(normal_scores) > 0:
        fig_score.add_trace(go.Histogram(
            x=normal_scores,
            name="정상",
            marker_color="#9b59b6",
            nbinsx=20,
            opacity=0.85
        ))

    # 이상치 구간 (스코어 < 0)
    anomaly_scores_neg = anomaly_scores[anomaly_scores < 0]
    if len(anomaly_scores_neg) > 0:
        fig_score.add_trace(go.Histogram(
            x=anomaly_scores_neg,
            name="이상치",
            marker_color="#e74c3c",
            nbinsx=20,
            opacity=0.85
        ))

    fig_score.update_layout(
        title="이상치 스코어 분포",
        xaxis_title="이상치 스코어 (0 미만 = 이상치)",
        yaxis_title="열차 수",
        barmode="overlay",
        legend=dict(orientation="h", y=1.1)
    )

    # 경계선 (0 기준)
    fig_score.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text="⬅ 이상치  |  정상 ➡",
        annotation_position="top"
    )

    st.plotly_chart(fig_score, use_container_width=True)


# ── 탭 4: Claude AI 자연어 질의 ───────────────────────────
with tab4:
    st.subheader("🤖 AI에게 열차 정보 물어보기")
    st.caption("예: '오늘 지연 예측 요약해줘', '소지연 열차 몇 개야?', '이상치 열차 설명해줘'")

    # 대화 기록을 세션 상태로 관리 (페이지 새로고침해도 유지)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 기존 대화 내역 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 사용자 입력창
    if prompt := st.chat_input("열차 정보에 대해 질문해보세요..."):

        # 사용자 메시지 저장 및 출력
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 현재 데이터 요약을 Claude에게 컨텍스트로 전달
        context = f"""
현재 열차 운행 데이터 요약:
- 전체 열차: {total}개
- 정상 운행: {normal}개 ({normal/total*100:.0f}%)
- 지연 예측: {delay}개 ({delay/total*100:.0f}%)
- 이상치 탐지: {anomaly}개
- 데이터 기준 시각: {datetime.now(KST).strftime('%Y-%m-%d %H:%M')} (KST)
"""
        # Claude API 호출
        with st.chat_message("assistant"):
            with st.spinner("분석 중..."):
                try:
                    import anthropic
                    from dotenv import load_dotenv
                    import os
                    load_dotenv()

                    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                    response = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=1000,
                        system="당신은 KTX 열차 운행 분석 전문가입니다. 제공된 데이터를 바탕으로 친절하고 명확하게 답변해주세요. 한국어로 답변해주세요.",
                        messages=[
                            {
                                "role": "user",
                                "content": f"{context}\n\n사용자 질문: {prompt}"
                            }
                        ]
                    )
                    answer = response.content[0].text
                    st.markdown(answer)
                    # AI 응답도 대화 기록에 저장
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"AI 응답 실패: {e}")


# ── 자동 갱신 ─────────────────────────────────────────────
if auto_refresh:
    # 30초 대기 후 캐시 초기화 + 페이지 재실행
    time.sleep(30)
    st.cache_data.clear()
    st.rerun()
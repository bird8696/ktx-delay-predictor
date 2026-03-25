import requests
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime
from urllib.parse import unquote

# .env 파일에서 환경변수 로드
load_dotenv()
# URL 인코딩 문제 방지를 위해 unquote()로 디코딩
API_KEY = unquote(os.getenv("KORAIL_API_KEY"))
# 코레일 열차운행정보 API 기본 주소
BASE_URL = "https://apis.data.go.kr/B551457/run/v2"


def parse_response(response) -> pd.DataFrame:
    """API 응답 JSON → DataFrame 변환"""
    try:
        # JSON 응답을 파이썬 딕셔너리로 변환
        data = response.json()

        # 실제 데이터는 response.body.items.item 안에 있음
        items = data["response"]["body"]["items"]["item"]

        # 공공데이터 API 특성상 결과가 1개면 dict, 여러 개면 list로 옴
        # 통일성을 위해 항상 list로 변환
        if isinstance(items, dict):
            items = [items]

        # list → DataFrame 변환
        df = pd.DataFrame(items)
        print(f"[파싱 완료] {len(df)}행 {len(df.columns)}열")
        return df

    except Exception as e:
        print(f"[파싱 실패] {e}")
        print(f"[원본 응답] {response.text[:300]}")
        # 실패 시 빈 DataFrame 반환 (에러 전파 방지)
        return pd.DataFrame()


def get_train_plan(dep_date: str = None) -> pd.DataFrame:
    """
    여객열차 운행계획 조회 (예정 시각)
    - 열차번호, 출발역, 도착역, 계획 출발/도착 시각 포함
    - dep_date: 조회할 날짜 (YYYYMMDD), 기본값은 오늘
    """
    # 날짜 미입력 시 오늘 날짜로 자동 설정 (예: 20260324)
    if dep_date is None:
        dep_date = datetime.now().strftime("%Y%m%d")

    # params 대신 URL 직접 조합 (requests의 이중 인코딩 문제 방지)
    url = (
        f"{BASE_URL}/travelerTrainRunPlan2"
        f"?serviceKey={API_KEY}"
        f"&numOfRows=500"   # 한 번에 최대 500개 조회
        f"&pageNo=1"        # 첫 번째 페이지
        f"&_type=json"      # JSON 형식으로 응답 요청
        f"&runDt={dep_date}"
    )

    response = requests.get(url, timeout=10)
    print(f"[운행계획] 상태코드: {response.status_code}")
    return parse_response(response)


def get_train_info(dep_date: str = None) -> pd.DataFrame:
    """
    여객열차 실제 운행정보 조회 (실제 출발/도착 + 지연)
    - 실제 도착 시각, 지연 여부 포함 → ML 지연 예측의 핵심 데이터
    - dep_date: 조회할 날짜 (YYYYMMDD), 기본값은 오늘
    """
    if dep_date is None:
        dep_date = datetime.now().strftime("%Y%m%d")

    url = (
        f"{BASE_URL}/travelerTrainRunInfo2"
        f"?serviceKey={API_KEY}"
        f"&numOfRows=500"
        f"&pageNo=1"
        f"&_type=json"
        f"&runDt={dep_date}"
    )

    response = requests.get(url, timeout=10)
    print(f"[운행정보] 상태코드: {response.status_code}")
    return parse_response(response)


# 이 파일을 직접 실행할 때만 아래 코드 동작
# 다른 파일에서 import할 때는 실행 안 됨
if __name__ == "__main__":
    print("=== 운행계획 ===")
    df_plan = get_train_plan()
    print(df_plan.head(3))
    print(f"컬럼 목록: {df_plan.columns.tolist()}")

    print("\n=== 운행정보 ===")
    df_info = get_train_info()
    print(df_info.head(3))
    print(f"컬럼 목록: {df_info.columns.tolist()}")
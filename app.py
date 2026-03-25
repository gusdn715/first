"""
한반도 방어 전력 시뮬레이션 대시보드
-------------------------------------
스택: Streamlit, Folium, streamlit-folium, Pandas
"""

import math
import numpy as np
import pandas as pd
import folium
import streamlit as st
from streamlit_folium import st_folium

# ═══════════════════════════════════════════════════════════════
# 페이지 설정
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="한반도 방어 전력 시뮬레이션",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# 상수 정의
# ═══════════════════════════════════════════════════════════════
KP_LAT_MIN, KP_LAT_MAX = 33.0, 43.0
KP_LON_MIN, KP_LON_MAX = 124.0, 131.0
KP_CENTER = [38.5, 127.5]

# 기본 미사일 종류 (요격 확률 매트릭스 열 헤더용)
DEFAULT_MISSILE_TYPES = [
    "단거리 탄도미사일 (SRBM)",
    "중거리 탄도미사일 (MRBM)",
    "순항미사일 (CM)",
    "극초음속 미사일 (HGV)",
]

# ═══════════════════════════════════════════════════════════════
# 유틸리티 함수
# ═══════════════════════════════════════════════════════════════

def is_valid_coords(lat: float, lon: float) -> bool:
    """입력 좌표가 한반도 유효 범위 내에 있는지 검사"""
    return KP_LAT_MIN <= lat <= KP_LAT_MAX and KP_LON_MIN <= lon <= KP_LON_MAX


def get_threat_color(level: int) -> tuple[str, str]:
    """
    위협도 수치(1~10)에 따라 (채우기 색상, 테두리 색상) 반환.
      1~3 : 녹색  (낮은 위협)
      4~6 : 주황  (중간 위협)
      7~8 : 빨강  (높은 위협)
      9~10: 자주  (극도 위협)
    """
    if level <= 3:
        return "#2ecc71", "#27ae60"
    elif level <= 6:
        return "#f39c12", "#d68910"
    elif level <= 8:
        return "#e74c3c", "#c0392b"
    else:
        return "#9b59b6", "#7d3c98"


def generate_radar_sector(
    center_lat: float,
    center_lon: float,
    range_km: float,
    azimuth_deg: float,
    detection_angle_deg: float,
    num_points: int = 72,
) -> list[list[float]]:
    """
    레이더 탐지 부채꼴(Sector)을 나타내는 Polygon 좌표 리스트 생성.

    [좌표 변환 원리]
    ─────────────────────────────────────────────────────────────
    1. 위도 1도 ≈ 111.32 km  (위도에 관계없이 거의 일정)
    2. 경도 1도 ≈ 111.32 × cos(위도_rad) km
       → 위도가 높아질수록 경도선 간격이 좁아지므로 보정 필요

    [방위각 → 수학 각도 변환]
    ─────────────────────────────────────────────────────────────
    방위각(Azimuth)  : 북쪽(0°)을 기준으로 시계 방향으로 증가
    수학적 각도      : 동쪽(0°)을 기준으로 반시계 방향으로 증가
    변환식           : math_angle_rad = (90° - azimuth°) × π/180

      azimuth=0   (북)  → math_angle=90°  → (dx=0,  dy=+R) ✓ 북쪽
      azimuth=90  (동)  → math_angle=0°   → (dx=+R, dy=0 ) ✓ 동쪽
      azimuth=180 (남)  → math_angle=-90° → (dx=0,  dy=-R) ✓ 남쪽
      azimuth=270 (서)  → math_angle=180° → (dx=-R, dy=0 ) ✓ 서쪽

    [부채꼴 꼭짓점 생성 순서]
    ─────────────────────────────────────────────────────────────
    중심점 → (방위각 - 탐지각/2) 방향 호 시작점 → 호 중간점들
           → (방위각 + 탐지각/2) 방향 호 끝점 → 중심점 (닫기)

    Parameters
    ----------
    center_lat        : 레이더 중심 위도 (도)
    center_lon        : 레이더 중심 경도 (도)
    range_km          : 레이더 탐지 최대 거리 (km)
    azimuth_deg       : 레이더가 지향하는 방위각 (도, 북=0, 시계방향)
    detection_angle_deg: 레이더 총 탐지 각도 (도)
    num_points        : 호를 구성하는 점의 수 (많을수록 부드러운 곡선)

    Returns
    -------
    [[lat, lon], ...] 형태의 좌표 목록 (Folium Polygon에 전달 가능)
    """
    half_angle = detection_angle_deg / 2.0

    # 탐지 범위 시작·끝 방위각
    start_az = azimuth_deg - half_angle
    end_az = azimuth_deg + half_angle

    # 중심 위도를 라디안으로 변환 (경도 보정에 사용)
    lat_rad = math.radians(center_lat)

    coords = [[center_lat, center_lon]]  # ① 중심점(꼭짓점)

    for i in range(num_points + 1):
        # ② 현재 방위각을 선형 보간으로 계산
        current_az = start_az + (end_az - start_az) * i / num_points

        # ③ 방위각 → 수학적 라디안 변환
        math_rad = math.radians(90.0 - current_az)

        # ④ km 단위 동서(dx) / 남북(dy) 이동량
        dx_km = range_km * math.cos(math_rad)   # 동(+) / 서(-) km
        dy_km = range_km * math.sin(math_rad)   # 북(+) / 남(-) km

        # ⑤ km → 위경도 변환
        #    위도 변화: 1° = 111.32 km → Δlat = dy_km / 111.32
        d_lat = dy_km / 111.32

        #    경도 변화: 1° = 111.32 × cos(lat) km → Δlon = dx_km / (111.32 × cos(lat))
        #    cos(lat) ≠ 0 이므로 안전하나, 극지방 근처에서는 오차 발생 가능
        d_lon = dx_km / (111.32 * math.cos(lat_rad))

        coords.append([center_lat + d_lat, center_lon + d_lon])

    coords.append([center_lat, center_lon])  # ⑥ 중심점으로 복귀 (다각형 닫기)
    return coords


# ═══════════════════════════════════════════════════════════════
# 세션 상태 초기화
# ═══════════════════════════════════════════════════════════════
if "missiles" not in st.session_state:
    st.session_state.missiles: list[dict] = []

if "defense_assets" not in st.session_state:
    st.session_state.defense_assets: list[dict] = []

# {(row_name, col_name): 요격확률(float)} — DataFrame 대신 dict로 관리하여 구조 변경과 무관하게 값 유지
if "intercept_probs" not in st.session_state:
    st.session_state.intercept_probs: dict = {}


def _unique_label(label: str, existing: list[str]) -> str:
    """중복 이름에 #2, #3 접미사를 붙여 고유한 레이블 반환"""
    if label not in existing:
        return label
    count = sum(1 for e in existing if e == label or e.startswith(label + " #"))
    return f"{label} #{count + 1}"


def build_prob_matrix() -> pd.DataFrame:
    """현재 defense_assets(행) × missiles(열) 기반으로 매트릭스를 매 렌더마다 새로 생성"""
    rows = [a["row_name"] for a in st.session_state.defense_assets]
    cols = [m["col_name"] for m in st.session_state.missiles]
    if not rows or not cols:
        return pd.DataFrame()
    data = {
        col: [
            st.session_state.intercept_probs.get((row, col), 50.0)
            for row in rows
        ]
        for col in cols
    }
    return pd.DataFrame(data, index=rows, dtype=float)


def save_prob_matrix(df: pd.DataFrame) -> None:
    """data_editor 결과를 intercept_probs dict에 저장"""
    for row in df.index:
        for col in df.columns:
            st.session_state.intercept_probs[(row, col)] = float(df.loc[row, col])


# ═══════════════════════════════════════════════════════════════
# 사이드바 – 전력 자산 관리
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ 전력 자산 관리")
    st.caption(f"유효 범위 | 위도 {KP_LAT_MIN}~{KP_LAT_MAX}° / 경도 {KP_LON_MIN}~{KP_LON_MAX}°")
    st.divider()

    tab_m, tab_d = st.tabs(["🚀 적 미사일 전력", "🛡️ 아군 방어 자산"])

    # ── 탭 1 : 적 미사일 ──────────────────────────────────────
    with tab_m:
        st.subheader("적 미사일 추가")
        with st.form("form_missile", clear_on_submit=True):
            m_name = st.text_input("미사일 명칭", value="KN-23", placeholder="예: KN-25")
            c1, c2 = st.columns(2)
            with c1:
                m_lat = st.number_input(
                    "위도 (°N)", min_value=28.0, max_value=48.0,
                    value=39.0, step=0.1, format="%.2f",
                )
            with c2:
                m_lon = st.number_input(
                    "경도 (°E)", min_value=118.0, max_value=138.0,
                    value=125.5, step=0.1, format="%.2f",
                )
            m_range = st.slider("최대 사거리 (km)", 50, 5000, 500, 50)
            m_threat = st.slider("위협도 (1 ~ 10)", 1, 10, 7)
            add_missile = st.form_submit_button("🚀 미사일 추가", use_container_width=True)

        if add_missile:
            if not is_valid_coords(m_lat, m_lon):
                st.error(
                    f"⚠️ 좌표({m_lat:.2f}°N, {m_lon:.2f}°E)가 한반도 범위를 벗어났습니다.\n\n"
                    f"위도 {KP_LAT_MIN}–{KP_LAT_MAX}° / 경도 {KP_LON_MIN}–{KP_LON_MAX}° 내에서 입력하세요."
                )
            else:
                existing_col_names = [m["col_name"] for m in st.session_state.missiles]
                col_label = _unique_label(m_name, existing_col_names)
                st.session_state.missiles.append(
                    dict(name=m_name, col_name=col_label,
                         lat=m_lat, lon=m_lon, range_km=m_range, threat=m_threat)
                )
                st.success(f"✅ '{m_name}' 추가 완료!")
                st.rerun()

        # 등록된 미사일 목록
        if st.session_state.missiles:
            st.divider()
            st.markdown("**등록된 미사일 목록**")
            for i, mis in enumerate(st.session_state.missiles):
                fill, border = get_threat_color(mis["threat"])
                badge = f'<span style="background:{fill};color:#fff;padding:1px 6px;border-radius:4px;font-size:11px;">위협도 {mis["threat"]}</span>'
                ca, cb = st.columns([4, 1])
                with ca:
                    st.markdown(
                        f"**{mis['name']}** {badge}  \n"
                        f"📍 {mis['lat']:.2f}°N, {mis['lon']:.2f}°E | 사거리 {mis['range_km']} km",
                        unsafe_allow_html=True,
                    )
                with cb:
                    if st.button("✕", key=f"del_m_{i}", help="삭제"):
                        removed_m = st.session_state.missiles.pop(i)
                        dead_col = removed_m["col_name"]
                        st.session_state.intercept_probs = {
                            k: v for k, v in st.session_state.intercept_probs.items()
                            if k[1] != dead_col
                        }
                        st.rerun()

    # ── 탭 2 : 아군 방어 자산 ─────────────────────────────────
    with tab_d:
        st.subheader("방어 자산 추가")
        with st.form("form_defense", clear_on_submit=True):
            d_name = st.text_input("자산 명칭", value="패트리엇 PAC-3", placeholder="예: THAAD")
            c1, c2 = st.columns(2)
            with c1:
                d_lat = st.number_input(
                    "위도 (°N)", min_value=28.0, max_value=48.0,
                    value=37.5, step=0.1, format="%.2f", key="def_lat",
                )
            with c2:
                d_lon = st.number_input(
                    "경도 (°E)", min_value=118.0, max_value=138.0,
                    value=127.0, step=0.1, format="%.2f", key="def_lon",
                )
            d_intercept = st.slider("요격 사거리 (km)", 10, 800, 100, 10)
            d_radar_rng = st.slider("레이더 탐지 거리 (km)", 50, 2000, 400, 10)
            d_azimuth   = st.slider("레이더 지향 방위각 (°, 북=0)", 0, 360, 0, 5)
            d_angle     = st.slider("레이더 탐지 각도 (°)", 10, 360, 90, 5)
            add_defense = st.form_submit_button("🛡️ 방어 자산 추가", use_container_width=True)

        if add_defense:
            if not is_valid_coords(d_lat, d_lon):
                st.error(
                    f"⚠️ 좌표({d_lat:.2f}°N, {d_lon:.2f}°E)가 한반도 범위를 벗어났습니다.\n\n"
                    f"위도 {KP_LAT_MIN}–{KP_LAT_MAX}° / 경도 {KP_LON_MIN}–{KP_LON_MAX}° 내에서 입력하세요."
                )
            else:
                existing_row_names = [a["row_name"] for a in st.session_state.defense_assets]
                row_label = _unique_label(d_name, existing_row_names)
                st.session_state.defense_assets.append(
                    dict(name=d_name, row_name=row_label,
                         lat=d_lat, lon=d_lon, intercept_km=d_intercept,
                         radar_km=d_radar_rng, azimuth=d_azimuth, angle=d_angle)
                )
                st.success(f"✅ '{d_name}' 추가 완료!")
                st.rerun()

        # 등록된 방어 자산 목록
        if st.session_state.defense_assets:
            st.divider()
            st.markdown("**등록된 방어 자산 목록**")
            for i, ast in enumerate(st.session_state.defense_assets):
                ca, cb = st.columns([4, 1])
                with ca:
                    st.markdown(
                        f"**{ast['name']}**  \n"
                        f"📍 {ast['lat']:.2f}°N, {ast['lon']:.2f}°E  \n"
                        f"요격 {ast['intercept_km']} km | 레이더 {ast['radar_km']} km  \n"
                        f"방위각 {ast['azimuth']}° / 탐지각 {ast['angle']}°"
                    )
                with cb:
                    if st.button("✕", key=f"del_d_{i}", help="삭제"):
                        removed = st.session_state.defense_assets.pop(i)
                        dead_row = removed["row_name"]
                        st.session_state.intercept_probs = {
                            k: v for k, v in st.session_state.intercept_probs.items()
                            if k[0] != dead_row
                        }
                        st.rerun()


# ═══════════════════════════════════════════════════════════════
# 메인 화면 – 헤더 및 요약 지표
# ═══════════════════════════════════════════════════════════════
st.markdown("# 🇰🇷 한반도 방어 전력 시뮬레이션 대시보드")

m_count = len(st.session_state.missiles)
d_count = len(st.session_state.defense_assets)
avg_threat = (
    float(np.mean([m["threat"] for m in st.session_state.missiles]))
    if st.session_state.missiles else 0.0
)
high_threat_count = sum(1 for m in st.session_state.missiles if m["threat"] >= 7)

col1, col2, col3, col4 = st.columns(4)
col1.metric("🚀 적 미사일 전력", f"{m_count} 기")
col2.metric("⚠️ 고위협 (7+)", f"{high_threat_count} 기")
col3.metric("🛡️ 아군 방어 자산", f"{d_count} 기")
col4.metric("📊 평균 위협도", f"{avg_threat:.1f} / 10")

st.divider()


# ═══════════════════════════════════════════════════════════════
# 메인 화면 – Folium 지도
# ═══════════════════════════════════════════════════════════════
st.subheader("🗺️ 전장 상황도")

# 범례 표시 (컬럼 활용)
leg1, leg2, leg3, leg4, leg5 = st.columns(5)
leg1.markdown('<span style="color:#27ae60;">●</span> 위협도 1~3 (낮음)', unsafe_allow_html=True)
leg2.markdown('<span style="color:#d68910;">●</span> 위협도 4~6 (중간)', unsafe_allow_html=True)
leg3.markdown('<span style="color:#c0392b;">●</span> 위협도 7~8 (높음)', unsafe_allow_html=True)
leg4.markdown('<span style="color:#7d3c98;">●</span> 위협도 9~10 (극도)', unsafe_allow_html=True)
leg5.markdown('<span style="color:#2980b9;">●</span> 아군 방어 / <span style="color:#1abc9c;">●</span> 레이더', unsafe_allow_html=True)

# ── Folium 지도 생성 ─────────────────────────────────────────
fmap = folium.Map(
    location=KP_CENTER,
    zoom_start=6,
    tiles="CartoDB positron",
    control_scale=True,
)

# 한반도 경계 참고선 (점선 사각형)
folium.Rectangle(
    bounds=[[KP_LAT_MIN, KP_LON_MIN], [KP_LAT_MAX, KP_LON_MAX]],
    color="#95a5a6",
    fill=False,
    weight=1,
    dash_array="6 4",
    tooltip="한반도 유효 범위",
).add_to(fmap)

# ── 적 미사일 시각화 ─────────────────────────────────────────
for mis in st.session_state.missiles:
    fill_c, border_c = get_threat_color(mis["threat"])

    # 사거리 원 (반투명)
    folium.Circle(
        location=[mis["lat"], mis["lon"]],
        radius=mis["range_km"] * 1000,   # km → m
        color=border_c,
        fill=True,
        fill_color=fill_c,
        fill_opacity=0.12,
        weight=2,
        tooltip=(
            f"🚀 {mis['name']}  |  "
            f"사거리: {mis['range_km']} km  |  "
            f"위협도: {mis['threat']}/10"
        ),
    ).add_to(fmap)

    # 발사 위치 마커
    folium.CircleMarker(
        location=[mis["lat"], mis["lon"]],
        radius=7,
        color=border_c,
        fill=True,
        fill_color=fill_c,
        fill_opacity=0.95,
        tooltip=f"🚀 {mis['name']}",
    ).add_to(fmap)

    # 라벨
    folium.Marker(
        location=[mis["lat"], mis["lon"]],
        icon=folium.DivIcon(
            html=(
                f'<div style="font-size:11px;font-weight:bold;color:{border_c};'
                f'white-space:nowrap;margin-top:-22px;margin-left:10px;">'
                f'{mis["name"]}</div>'
            ),
            icon_size=(120, 20),
            icon_anchor=(0, 0),
        ),
    ).add_to(fmap)

# ── 아군 방어 자산 시각화 ────────────────────────────────────
for ast in st.session_state.defense_assets:

    # 요격 사거리 원 (파란색 점선)
    folium.Circle(
        location=[ast["lat"], ast["lon"]],
        radius=ast["intercept_km"] * 1000,
        color="#2980b9",
        fill=True,
        fill_color="#3498db",
        fill_opacity=0.10,
        weight=2,
        dash_array="6 4",
        tooltip=(
            f"🛡️ {ast['name']}  |  "
            f"요격 사거리: {ast['intercept_km']} km"
        ),
    ).add_to(fmap)

    # 레이더 탐지 부채꼴
    sector_coords = generate_radar_sector(
        center_lat=ast["lat"],
        center_lon=ast["lon"],
        range_km=ast["radar_km"],
        azimuth_deg=ast["azimuth"],
        detection_angle_deg=ast["angle"],
    )
    folium.Polygon(
        locations=sector_coords,
        color="#1abc9c",
        fill=True,
        fill_color="#1abc9c",
        fill_opacity=0.18,
        weight=1.5,
        tooltip=(
            f"📡 {ast['name']} 레이더  |  "
            f"방위각: {ast['azimuth']}°  |  "
            f"탐지각: {ast['angle']}°  |  "
            f"탐지 거리: {ast['radar_km']} km"
        ),
    ).add_to(fmap)

    # 자산 위치 마커
    folium.CircleMarker(
        location=[ast["lat"], ast["lon"]],
        radius=8,
        color="#1a5276",
        fill=True,
        fill_color="#2980b9",
        fill_opacity=0.95,
        tooltip=f"🛡️ {ast['name']}",
    ).add_to(fmap)

    # 라벨
    folium.Marker(
        location=[ast["lat"], ast["lon"]],
        icon=folium.DivIcon(
            html=(
                f'<div style="font-size:11px;font-weight:bold;color:#1a5276;'
                f'white-space:nowrap;margin-top:-22px;margin-left:10px;">'
                f'{ast["name"]}</div>'
            ),
            icon_size=(130, 20),
            icon_anchor=(0, 0),
        ),
    ).add_to(fmap)

# 지도 렌더링
st_folium(fmap, use_container_width=True, height=600, returned_objects=[])


# ═══════════════════════════════════════════════════════════════
# 메인 화면 – 교전 요격 확률 매트릭스
# ═══════════════════════════════════════════════════════════════
st.divider()
st.subheader("📊 교전 요격 확률 매트릭스 (%)")
st.caption(
    "행: 아군 방어 자산  |  열: 적 미사일 종류  |  "
    "셀을 직접 클릭하여 요격 성공 확률(%)을 수정할 수 있습니다."
)

# 매 렌더마다 현재 missiles(열) × defense_assets(행)으로 매트릭스를 새로 구성
current_matrix = build_prob_matrix()

if current_matrix.empty:
    if not st.session_state.defense_assets and not st.session_state.missiles:
        st.info("사이드바에서 아군 방어 자산과 적 미사일을 추가하면 매트릭스가 자동으로 생성됩니다.")
    elif not st.session_state.missiles:
        st.info("사이드바에서 적 미사일을 추가하면 열(컬럼)이 생성됩니다.")
    else:
        st.info("사이드바에서 아군 방어 자산을 추가하면 행이 생성됩니다.")
else:
    col_config = {
        col: st.column_config.NumberColumn(
            label=col,
            min_value=0.0,
            max_value=100.0,
            step=0.5,
            format="%.1f%%",
        )
        for col in current_matrix.columns
    }

    edited_df = st.data_editor(
        current_matrix,
        column_config=col_config,
        use_container_width=True,
        num_rows="fixed",
        key="prob_editor",
    )

    # 편집된 값을 intercept_probs dict에 저장 (구조와 분리된 값 보존)
    save_prob_matrix(edited_df)

    st.markdown("#### 자산별 평균 요격 확률")
    bar_data = edited_df.mean(axis=1)
    st.bar_chart(bar_data, use_container_width=True, height=220)


# ═══════════════════════════════════════════════════════════════
# 푸터
# ═══════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "⚠️ 본 시뮬레이션은 교육·연구 목적의 가상 시나리오이며 "
    "실제 군사 정보와 무관합니다.  |  "
    "스택: Streamlit · Folium · Pandas"
)

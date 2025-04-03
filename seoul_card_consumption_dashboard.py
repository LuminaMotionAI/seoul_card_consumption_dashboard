import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

# Try to import koreanize_matplotlib, handle if not available
try:
    import koreanize_matplotlib
except ImportError:
    st.warning("koreanize_matplotlib 패키지가 설치되지 않았습니다. 한글 폰트가 정상적으로 표시되지 않을 수 있습니다.")

# 페이지 기본 설정
st.set_page_config(
    page_title="서울시민 카드소비 분석 대시보드", 
    page_icon="💳",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .insight-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 서울시민 온라인 카드소비 분석</div>', unsafe_allow_html=True)

# 샘플 데이터 생성 함수
def generate_sample_data(n_samples=10000):
    # 온라인 업종 정의
    online_categories = [
        '간편식/건강식품', '교육/학습', '도서/음반', '문화/예술',
        '미용/뷰티', '배달앱', '생활쇼핑', '스트리밍서비스',
        '여행/교통', '온라인게임', '의류/패션', '전자기기',
        '정기구독', '홈인테리어'
    ]
    
    # 행정동 코드 생성 (서울시 행정동 코드 형식)
    admin_codes = [f'11{i:03d}' for i in range(1, 51)]
    
    # 기준월 생성 (2021-01 ~ 2023-12)
    base_dates = []
    for year in range(2021, 2024):
        for month in range(1, 13):
            base_dates.append(f"{year}{month:02d}")
    
    # 연령대 및 성별
    age_groups = ['20대', '30대', '40대', '50대', '60대 이상']
    genders = ['남성', '여성']
    
    # 데이터 생성
    data = []
    for _ in range(n_samples):
        base_date = random.choice(base_dates)
        category = random.choice(online_categories)
        admin_code = random.choice(admin_codes)
        age_group = random.choice(age_groups)
        gender = random.choice(genders)
        
        # 업종별 금액 범위 차등화
        if category in ['전자기기', '여행/교통']:
            amount = random.randint(100000, 1000000)
            transactions = random.randint(1, 5)
        elif category in ['온라인게임', '스트리밍서비스', '정기구독']:
            amount = random.randint(10000, 50000)
            transactions = random.randint(1, 10)
        elif category in ['배달앱']:
            amount = random.randint(15000, 100000)
            transactions = random.randint(3, 15)
        else:
            amount = random.randint(20000, 300000)
            transactions = random.randint(1, 8)
        
        # 계절적 효과 추가
        month = int(base_date[4:6])
        year = int(base_date[:4])
        
        # 여름에는 배달앱, 겨울에는 온라인쇼핑 증가
        if category == '배달앱' and month in [6, 7, 8]:
            amount = int(amount * 1.3)
        if category == '생활쇼핑' and month in [11, 12, 1]:
            amount = int(amount * 1.4)
        
        # 코로나 효과 (2021년은 더 높은 온라인 소비)
        if year == 2021:
            amount = int(amount * 1.2)
        
        data.append({
            '기준월': base_date,
            '온라인업종': category,
            '고객행정동코드': admin_code,
            '연령대': age_group,
            '성별': gender,
            '카드이용건수': transactions,
            '카드이용금액계': amount
        })
    
    df = pd.DataFrame(data)
    
    # 후처리: 거래액이 0원인 레코드 제거
    df = df[df['카드이용금액계'] > 0]
    
    # 기준월에서 연도와 월 추출
    df['연도'] = df['기준월'].astype(str).str[:4]
    df['월'] = df['기준월'].astype(str).str[4:].str.zfill(2)
    
    return df

# 사이드바에 데이터 샘플 크기 조절
st.sidebar.header('데이터 생성 설정')
n_samples = st.sidebar.slider('샘플 데이터 수', min_value=5000, max_value=100000, value=50000, step=5000)
data_load_state = st.sidebar.text('데이터 생성 중...')
data = generate_sample_data(n_samples)
data_load_state.text(f'데이터 생성 완료: {data.shape[0]}개 레코드')

# 사이드바에 필터 추가
st.sidebar.header('데이터 필터')
year_filter = st.sidebar.multiselect(
    '연도 선택',
    options=sorted(data['연도'].unique()),
    default=sorted(data['연도'].unique())
)

# 필터 적용
filtered_data = data[data['연도'].isin(year_filter)]

# 탭 생성
탭1, 탭2, 탭3, 탭4, 탭5 = st.tabs([
    "📈 마케팅 트렌드 분석", 
    "📍 지역 기반 BI 분석", 
    "🔍 업종 군집 분석", 
    "📌 비중 급등 업종 분석", 
    "🔔 인사이트 기반 추천"
])

# 탭1: 마케팅 트렌드 분석
with 탭1:
    st.markdown("### 월별 주요 업종 소비 추이")
    
    # 연도 선택 드롭다운
    selected_year = st.selectbox(
        '분석할 연도 선택',
        options=sorted(filtered_data['연도'].unique()),
        index=len(filtered_data['연도'].unique())-1  # 가장 최근 연도 기본 선택
    )
    
    # 상위 업종 선택 슬라이더
    top_n_categories = st.slider('상위 N개 업종 표시', min_value=3, max_value=10, value=5)
    
    # 연도별, 월별, 업종별 소비 금액 합계 계산
    monthly_sum = filtered_data.groupby(['연도', '월', '온라인업종'])['카드이용금액계'].sum().reset_index()
    
    # 선택한 연도의 데이터만 필터링
    year_data = monthly_sum[monthly_sum['연도'] == selected_year]
    
    # 상위 N개 업종 찾기
    top_categories = filtered_data[filtered_data['연도'] == selected_year].groupby('온라인업종')['카드이용금액계'].sum().nlargest(top_n_categories).index.tolist()
    year_data_top = year_data[year_data['온라인업종'].isin(top_categories)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Plotly로 월별 업종 소비 추이 그래프
        fig = px.line(
            year_data_top, 
            x='월', 
            y='카드이용금액계', 
            color='온라인업종',
            markers=True,
            title=f'{selected_year}년 월별 업종 소비 추이',
            labels={'월': '월', '카드이용금액계': '카드이용금액(원)', '온라인업종': '업종'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 업종별 총 소비 금액 막대 그래프
        category_sum = filtered_data[filtered_data['연도'] == selected_year].groupby('온라인업종')['카드이용금액계'].sum().sort_values(ascending=False).reset_index()
        category_sum = category_sum.head(top_n_categories)
        
        fig = px.bar(
            category_sum,
            x='온라인업종',
            y='카드이용금액계',
            title=f'{selected_year}년 업종별 총 소비 금액',
            labels={'온라인업종': '업종', '카드이용금액계': '카드이용금액(원)'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # 연도별 업종 소비 추이 (모든 연도)
    st.markdown("### 연도별 업종 소비 추이")
    yearly_sum = filtered_data.groupby(['연도', '온라인업종'])['카드이용금액계'].sum().reset_index()
    yearly_top = yearly_sum[yearly_sum['온라인업종'].isin(top_categories)]
    
    fig = px.line(
        yearly_top,
        x='연도',
        y='카드이용금액계',
        color='온라인업종',
        markers=True,
        title='연도별 주요 업종 소비 추이',
        labels={'연도': '연도', '카드이용금액계': '카드이용금액(원)', '온라인업종': '업종'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 인사이트 박스
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown(f"#### {selected_year}년 주요 온라인 소비 트렌드 인사이트")
    
    # 가장 많이 소비된 업종
    top_category = category_sum.iloc[0]['온라인업종']
    top_amount = category_sum.iloc[0]['카드이용금액계']
    
    # 성장률이 가장 높은 업종 (첫 달과 마지막 달 비교)
    first_month = year_data.sort_values('월').iloc[0]['월']
    last_month = year_data.sort_values('월').iloc[-1]['월']
    
    growth_data = []
    for cat in top_categories:
        try:
            first_amount = year_data[(year_data['온라인업종'] == cat) & (year_data['월'] == first_month)]['카드이용금액계'].values[0]
            last_amount = year_data[(year_data['온라인업종'] == cat) & (year_data['월'] == last_month)]['카드이용금액계'].values[0]
            growth_rate = ((last_amount - first_amount) / first_amount) * 100
            growth_data.append({'업종': cat, '성장률': growth_rate})
        except:
            continue
    
    if growth_data:
        growth_df = pd.DataFrame(growth_data)
        fastest_growing = growth_df.loc[growth_df['성장률'].idxmax()]
        
        st.markdown(f"- 가장 많이 소비된 온라인 업종: **{top_category}** (총 {top_amount:,}원)")
        st.markdown(f"- 가장 높은 성장률을 보인 업종: **{fastest_growing['업종']}** ({fastest_growing['성장률']:.2f}%)")
    st.markdown('</div>', unsafe_allow_html=True)

# 탭2: 지역 기반 BI 분석
with 탭2:
    st.markdown("### 행정동별 소비 패턴 분석")
    
    # 행정동별 주요 소비 업종
    district_sum = filtered_data.groupby(['고객행정동코드', '온라인업종'])['카드이용금액계'].sum().reset_index()
    top_category_by_district = district_sum.loc[district_sum.groupby('고객행정동코드')['카드이용금액계'].idxmax()]
    
    # 행정동별 주요 업종 카운트
    district_cat_count = top_category_by_district['온라인업종'].value_counts().reset_index()
    district_cat_count.columns = ['업종', '행정동 수']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 최다 소비 업종별 행정동 수 막대 그래프
        fig = px.bar(
            district_cat_count.head(10),
            x='업종',
            y='행정동 수',
            title='행정동별 최다 소비 업종 Top 10',
            labels={'업종': '업종', '행정동 수': '행정동 수'},
            color='업종'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 파이 차트로 행정동별 주요 업종 분포
        fig = px.pie(
            district_cat_count,
            values='행정동 수',
            names='업종',
            title='행정동별 주요 소비 업종 분포'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # 행정동 선택 기능
    st.markdown("### 특정 행정동 소비 패턴 분석")
    selected_district = st.selectbox(
        '분석할 행정동 선택',
        options=sorted(filtered_data['고객행정동코드'].unique())
    )
    
    # 선택한 행정동의 업종별 소비 금액
    district_data = filtered_data[filtered_data['고객행정동코드'] == selected_district]
    district_category_sum = district_data.groupby('온라인업종')['카드이용금액계'].sum().sort_values(ascending=False).reset_index()
    
    # 선택한 행정동의 연도별 소비 추이
    district_year_sum = district_data.groupby(['연도', '온라인업종'])['카드이용금액계'].sum().reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 선택 행정동의 업종별 소비 금액
        fig = px.bar(
            district_category_sum.head(8),
            x='온라인업종',
            y='카드이용금액계',
            title=f'행정동 {selected_district}의 업종별 소비 금액',
            labels={'온라인업종': '업종', '카드이용금액계': '소비 금액(원)'},
            color='온라인업종'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 선택 행정동의 연령대별 소비 금액
        district_age_sum = district_data.groupby('연령대')['카드이용금액계'].sum().sort_values(ascending=False).reset_index()
        
        fig = px.bar(
            district_age_sum,
            x='연령대',
            y='카드이용금액계',
            title=f'행정동 {selected_district}의 연령대별 소비 금액',
            labels={'연령대': '연령대', '카드이용금액계': '소비 금액(원)'},
            color='연령대'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # 인사이트 박스
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown(f"#### 행정동 {selected_district} 소비 패턴 인사이트")
    
    # 가장 소비가 많은 업종
    top_district_category = district_category_sum.iloc[0]['온라인업종']
    top_district_amount = district_category_sum.iloc[0]['카드이용금액계']
    
    # 가장 소비가 많은 연령대
    top_district_age = district_age_sum.iloc[0]['연령대']
    top_district_age_amount = district_age_sum.iloc[0]['카드이용금액계']
    
    # 성별 소비 분포
    district_gender_sum = district_data.groupby('성별')['카드이용금액계'].sum().reset_index()
    
    st.markdown(f"- 가장 많이 소비된 업종: **{top_district_category}** (총 {top_district_amount:,}원)")
    st.markdown(f"- 가장 많이 소비한 연령대: **{top_district_age}** (총 {top_district_age_amount:,}원)")
    
    # 성별 비율 계산
    if len(district_gender_sum) == 2:
        male_ratio = district_gender_sum[district_gender_sum['성별'] == '남성']['카드이용금액계'].values[0]
        female_ratio = district_gender_sum[district_gender_sum['성별'] == '여성']['카드이용금액계'].values[0]
        total = male_ratio + female_ratio
        male_pct = (male_ratio / total) * 100
        female_pct = (female_ratio / total) * 100
        st.markdown(f"- 성별 소비 비율: 남성 **{male_pct:.1f}%**, 여성 **{female_pct:.1f}%**")
    
    st.markdown('</div>', unsafe_allow_html=True)

# 탭3: 업종 군집 분석
with 탭3:
    st.markdown("### 업종별 소비 패턴 군집 분석")
    
    # 연도 선택
    cluster_year = st.selectbox(
        '군집 분석할 연도 선택',
        options=sorted(filtered_data['연도'].unique()),
        index=len(filtered_data['연도'].unique())-1,
        key='cluster_year'
    )
    
    year_data = filtered_data[filtered_data['연도'] == cluster_year]
    
    # 업종별 월간 소비 패턴 분석을 위한 피벗 테이블 생성
    pivot_data = year_data.pivot_table(
        index='온라인업종', 
        columns='월', 
        values='카드이용금액계', 
        aggfunc='sum'
    ).fillna(0)
    
    if len(pivot_data) > 0:
        # 군집 수 선택
        n_clusters = st.slider('군집 수', min_value=2, max_value=6, value=4)
        
        # 표준화
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pivot_data)
        
        # K-means 군집화
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        # 결과 데이터프레임 생성
        cluster_df = pd.DataFrame({
            '업종': pivot_data.index,
            '군집': clusters
        })
        
        # 각 군집별 대표 업종 표시
        cluster_counts = cluster_df['군집'].value_counts().reset_index()
        cluster_counts.columns = ['군집', '업종 수']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 군집별 업종 수 막대 그래프
            fig = px.bar(
                cluster_counts,
                x='군집',
                y='업종 수',
                title=f'{cluster_year}년 K-means 군집별 업종 수',
                labels={'군집': '군집', '업종 수': '업종 수'},
                color='군집'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 군집별 업종 비율 파이 차트
            fig = px.pie(
                cluster_counts,
                values='업종 수',
                names='군집',
                title=f'{cluster_year}년 군집별 업종 분포 비율'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # 군집별 대표 업종 및 소비 패턴 분석
        st.markdown("### 군집별 대표 업종")
        
        # 각 군집 탭 생성
        cluster_tabs = st.tabs([f"군집 {i}" for i in range(n_clusters)])
        
        for i, tab in enumerate(cluster_tabs):
            with tab:
                # 현재 군집에 속하는 업종 목록
                cluster_industries = cluster_df[cluster_df['군집'] == i]
                
                # 표 형태로 업종 목록 표시
                st.dataframe(cluster_industries[['업종']])
                
                # 군집에 속한 업종들의 월별 소비 패턴 시각화
                if len(cluster_industries) > 0:
                    # 현재 군집의 업종 목록
                    industries_in_cluster = cluster_industries['업종'].tolist()
                    
                    # 해당 업종들의 월별 소비 패턴 데이터
                    pattern_data = year_data[year_data['온라인업종'].isin(industries_in_cluster)]
                    monthly_pattern = pattern_data.groupby(['월', '온라인업종'])['카드이용금액계'].sum().reset_index()
                    
                    # 월별 소비 패턴 시각화
                    fig = px.line(
                        monthly_pattern,
                        x='월',
                        y='카드이용금액계',
                        color='온라인업종',
                        markers=True,
                        title=f'군집 {i} 업종의 월별 소비 패턴',
                        labels={'월': '월', '카드이용금액계': '소비 금액(원)', '온라인업종': '업종'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 군집 특성 분석
                    cluster_sum = pattern_data.groupby('온라인업종')['카드이용금액계'].sum().sort_values(ascending=False)
                    top_industry = cluster_sum.index[0] if len(cluster_sum) > 0 else "없음"
                    
                    st.markdown(f"**군집 {i} 특성:**")
                    st.markdown(f"- 포함 업종 수: {len(industries_in_cluster)}개")
                    st.markdown(f"- 대표 업종: {top_industry}")
                    
                    # 분기별 소비 패턴 (월을 분기로 변환)
                    pattern_data['분기'] = pattern_data['월'].apply(lambda x: str((int(x)-1)//3 + 1) + '분기')
                    quarterly_pattern = pattern_data.groupby('분기')['카드이용금액계'].sum().reindex(['1분기', '2분기', '3분기', '4분기'])
                    
                    # 분기별 소비 비중 계산
                    total_amount = quarterly_pattern.sum()
                    quarterly_percentage = (quarterly_pattern / total_amount * 100).fillna(0)
                    
                    # 분기별 소비 비중 표시
                    quarters = ['1분기', '2분기', '3분기', '4분기']
                    percentages = [quarterly_percentage.get(q, 0) for q in quarters]
                    
                    # 가장 소비가 많은 분기 찾기
                    max_quarter_idx = percentages.index(max(percentages))
                    max_quarter = quarters[max_quarter_idx]
                    
                    st.markdown(f"- 주요 소비 시기: **{max_quarter}** ({percentages[max_quarter_idx]:.1f}%)")
                    
                    # 군집 특성에 따른 마케팅 인사이트
                    st.markdown("**마케팅 인사이트:**")
                    
                    # 계절성 효과 확인
                    if max_quarter == '1분기':
                        st.markdown("- 겨울~봄 시즌에 마케팅 캠페인 집중 권장")
                    elif max_quarter == '2분기':
                        st.markdown("- 봄~여름 시즌에 마케팅 캠페인 집중 권장")
                    elif max_quarter == '3분기':
                        st.markdown("- 여름~가을 시즌에 마케팅 캠페인 집중 권장") 
                    else:
                        st.markdown("- 가을~겨울 시즌에 마케팅 캠페인 집중 권장")
                    
                    # 군집 내 업종 간의 교차 판매 기회
                    if len(industries_in_cluster) > 1:
                        industry_pairs = []
                        for idx, ind1 in enumerate(industries_in_cluster[:3]):
                            for ind2 in industries_in_cluster[idx+1:4]:
                                if ind2 not in industries_in_cluster[:3]: continue
                                industry_pairs.append(f"{ind1} + {ind2}")
                        
                        if industry_pairs:
                            pairs_text = ", ".join(industry_pairs[:2])
                            st.markdown(f"- 번들 상품 기회: **{pairs_text}**")
                else:
                    st.markdown("이 군집에 속한 업종이 없습니다.")
        
        # 실루엣 점수 계산 및 최적 군집 수 제안
        from sklearn.metrics import silhouette_score
        
        st.markdown("### 군집 분석 품질 평가")
        if len(pivot_data) >= n_clusters:
            # 실루엣 점수 계산
            silhouette_avg = silhouette_score(scaled_data, clusters)
            st.metric(label="실루엣 점수", value=f"{silhouette_avg:.3f}", 
                    delta_color="normal")
            
            # 실루엣 점수 해석
            if silhouette_avg < 0.2:
                st.warning("군집 품질이 낮습니다. 군집 수를 조정해보세요.")
            elif silhouette_avg < 0.5:
                st.info("군집 품질이 보통입니다.")
            else:
                st.success("군집 품질이 좋습니다.")
        else:
            st.warning("데이터 포인트 수가 군집 수보다 작아 실루엣 점수를 계산할 수 없습니다.")
    else:
        st.warning("선택한 연도에 대한 데이터가 충분하지 않습니다.")

# 탭4: 비중 급등 업종 분석
with 탭4:
    st.markdown("### 업종 소비 비중 급등 분석")
    
    # 연도 선택
    fluctuation_year = st.selectbox(
        '분석할 연도 선택',
        options=sorted(filtered_data['연도'].unique()),
        index=len(filtered_data['연도'].unique())-1,
        key='fluctuation_year'
    )
    
    # 선택한 연도의 데이터만 필터링
    year_data_fluct = filtered_data[filtered_data['연도'] == fluctuation_year]
    
    # 월별 전체 소비 금액 계산
    monthly_total = year_data_fluct.groupby('기준월')['카드이용금액계'].sum().reset_index()
    monthly_total.columns = ['기준월', '총소비금액']
    
    # 업종별, 월별 소비 금액 계산
    category_monthly = year_data_fluct.groupby(['기준월', '온라인업종'])['카드이용금액계'].sum().reset_index()
    
    # 전체 소비 금액에 대한 데이터 합치기
    category_monthly = pd.merge(category_monthly, monthly_total, on='기준월')
    
    # 소비 비중 계산
    category_monthly['소비비중'] = (category_monthly['카드이용금액계'] / category_monthly['총소비금액']) * 100
    
    # 피벗 테이블 생성 (행: 기준월, 열: 업종, 값: 소비 비중)
    pivot_ratio = category_monthly.pivot_table(
        index='기준월',
        columns='온라인업종',
        values='소비비중'
    ).fillna(0)
    
    # 전월 대비 소비 비중 변화 계산
    ratio_change = pivot_ratio.diff()
    
    # 언피봇하고 변화량 기준 정렬
    top_fluctuation = ratio_change.unstack().reset_index()
    top_fluctuation.columns = ['업종', '기준월', '비중변화']
    top_fluctuation = top_fluctuation.dropna().sort_values(by='비중변화', ascending=False)
    
    # 상위 급증 업종 및 하위 급감 업종 선택
    num_display = st.slider('표시할 업종 수', min_value=5, max_value=20, value=10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 소비 비중 급증 업종 Top {}".format(num_display))
        top_increase = top_fluctuation.head(num_display)
        
        # 표 형태로 데이터 표시 - 수정된 부분
        st.dataframe(
            top_increase[['업종', '기준월', '비중변화']].reset_index(drop=True)
        )
        
        # 비중변화 값의 소수점 두 자리까지 표시
        st.markdown("*비중변화는 %p(퍼센트 포인트) 단위입니다*")
        
        # 급증 업종 시각화
        if len(top_increase) > 0:
            fig = px.bar(
                top_increase,
                x='업종',
                y='비중변화',
                color='비중변화',
                color_continuous_scale='Viridis',
                title=f'{fluctuation_year}년 소비 비중 급증 업종 Top {num_display}',
                labels={'업종': '업종', '비중변화': '비중 변화(%p)'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 소비 비중 급감 업종 Top {}".format(num_display))
        top_decrease = top_fluctuation.tail(num_display).sort_values(by='비중변화')
        
        # 표 형태로 데이터 표시 - 수정된 부분
        st.dataframe(
            top_decrease[['업종', '기준월', '비중변화']].reset_index(drop=True)
        )
        
        # 비중변화 값의 소수점 두 자리까지 표시
        st.markdown("*비중변화는 %p(퍼센트 포인트) 단위입니다*")
        
        # 급감 업종 시각화
        if len(top_decrease) > 0:
            fig = px.bar(
                top_decrease,
                x='업종',
                y='비중변화',
                color='비중변화',
                color_continuous_scale='Viridis_r',
                title=f'{fluctuation_year}년 소비 비중 급감 업종 Top {num_display}',
                labels={'업종': '업종', '비중변화': '비중 변화(%p)'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # 특정 업종의 월별 소비 비중 변화 시각화
    st.markdown("### 특정 업종의 소비 비중 변화 추이")
    
    # 업종 선택
    selected_category = st.selectbox(
        '분석할 업종 선택',
        options=sorted(filtered_data['온라인업종'].unique())
    )
    
    # 선택한 업종의 월별 소비 비중 데이터
    category_ratio_data = category_monthly[category_monthly['온라인업종'] == selected_category]
    
    # 시각화
    if len(category_ratio_data) > 0:
        # 월 정보 추출
        category_ratio_data['월'] = category_ratio_data['기준월'].astype(str).str[4:].str.zfill(2)
        
        fig = px.line(
            category_ratio_data,
            x='월',
            y='소비비중',
            markers=True,
            title=f'{fluctuation_year}년 {selected_category} 업종의 월별 소비 비중 변화',
            labels={'월': '월', '소비비중': '소비 비중(%)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # 비중 변화 기술 통계
        min_ratio = category_ratio_data['소비비중'].min()
        max_ratio = category_ratio_data['소비비중'].max()
        mean_ratio = category_ratio_data['소비비중'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("최소 소비 비중", f"{min_ratio:.2f}%")
        col2.metric("최대 소비 비중", f"{max_ratio:.2f}%")
        col3.metric("평균 소비 비중", f"{mean_ratio:.2f}%")
    else:
        st.warning(f"{fluctuation_year}년에 {selected_category} 업종의 소비 데이터가 없습니다.")
    
    # 인사이트 박스
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("#### 비중 변화 분석 인사이트")
    
    if len(top_increase) > 0:
        # 가장 급증한 업종
        top_increase_category = top_increase.iloc[0]['업종']
        top_increase_month = top_increase.iloc[0]['기준월']
        top_increase_change = top_increase.iloc[0]['비중변화']
        month_str = top_increase_month[4:].lstrip('0')
        
        st.markdown(f"- 가장 급증한 업종: **{top_increase_category}** ({month_str}월, +{top_increase_change:.2f}%p)")
        
        # 급증 업종의 특성 분석
        category_data = filtered_data[filtered_data['온라인업종'] == top_increase_category]
        
        # 연령대별 소비 비율
        age_ratio = category_data.groupby('연령대')['카드이용금액계'].sum()
        top_age = age_ratio.idxmax()
        
        # 성별 소비 비율
        gender_ratio = category_data.groupby('성별')['카드이용금액계'].sum()
        if len(gender_ratio) == 2:
            male_ratio = gender_ratio.get('남성', 0)
            female_ratio = gender_ratio.get('여성', 0)
            if male_ratio > female_ratio:
                gender_str = f"남성 선호 (남성 {male_ratio/(male_ratio+female_ratio)*100:.1f}%)"
            else:
                gender_str = f"여성 선호 (여성 {female_ratio/(male_ratio+female_ratio)*100:.1f}%)"
        else:
            gender_str = "데이터 부족"
        
        st.markdown(f"- **{top_increase_category}** 특성: 주 소비층 **{top_age}**, {gender_str}")
        
        # 마케팅 제안
        st.markdown("- **마케팅 제안**: ")
        st.markdown(f"  * {top_increase_category} 업종 중심 프로모션 기획 ({month_str}월)")
        
        # 급증 업종과 함께 소비되는 업종 추천
        month_data = filtered_data[filtered_data['기준월'] == top_increase_month]
        other_categories = set(month_data['온라인업종'].unique()) - {top_increase_category}
        
        recommended_bundle = []
        for cat in other_categories:
            if cat in top_increase['업종'].values[:3]:  # 상위 3개 급증 업종 중 하나인 경우
                recommended_bundle.append(cat)
        
        if recommended_bundle:
            bundle_text = ", ".join(recommended_bundle[:2])
            st.markdown(f"  * 번들 상품 추천: {top_increase_category} + **{bundle_text}**")
    
    st.markdown('</div>', unsafe_allow_html=True)

# 탭5: 인사이트 기반 추천
with 탭5:
    st.markdown("### 소비 트렌드 기반 추천 인사이트")
    
    # 메인 요약 정보 계산
    
    # 1. 상위 5개 인기 업종
    top_categories_overall = filtered_data.groupby('온라인업종')['카드이용금액계'].sum().nlargest(5)
    
    # 2. 최근 가장 성장한 업종 (첫 해와 마지막 해 비교)
    years = sorted(filtered_data['연도'].unique())
    if len(years) >= 2:
        first_year = years[0]
        last_year = years[-1]
        
        first_year_data = filtered_data[filtered_data['연도'] == first_year]
        last_year_data = filtered_data[filtered_data['연도'] == last_year]
        
        first_year_sum = first_year_data.groupby('온라인업종')['카드이용금액계'].sum()
        last_year_sum = last_year_data.groupby('온라인업종')['카드이용금액계'].sum()
        
        # 성장률 계산을 위해 데이터프레임 병합
        growth_df = pd.DataFrame({
            'first_year': first_year_sum,
            'last_year': last_year_sum
        }).fillna(0)
        
        # 성장률 계산
        growth_df['성장률'] = ((growth_df['last_year'] - growth_df['first_year']) / growth_df['first_year']) * 100
        
        # 성장률 기준 상위 5개 업종
        top_growth_categories = growth_df.sort_values('성장률', ascending=False).head(5)
    
    # 3. 연령대별 선호 업종
    age_preference = filtered_data.groupby(['연령대', '온라인업종'])['카드이용금액계'].sum().reset_index()
    top_by_age = age_preference.loc[age_preference.groupby('연령대')['카드이용금액계'].idxmax()]
    
    # 4. 성별 선호 업종
    gender_preference = filtered_data.groupby(['성별', '온라인업종'])['카드이용금액계'].sum().reset_index()
    top_by_gender = gender_preference.loc[gender_preference.groupby('성별')['카드이용금액계'].idxmax()]
    
    # 대시보드 표시
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 1. 인기 업종 기반 추천")
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**최고 인기 온라인 업종 TOP 5**")
        
        for i, (category, amount) in enumerate(top_categories_overall.items()):
            st.markdown(f"{i+1}. **{category}** - {amount:,.0f}원")
        
        st.markdown("**마케팅 추천**")
        st.markdown(f"- 상위 3개 업종 ({', '.join(top_categories_overall.index[:3])}) 중심의 메인 마케팅 전략 수립")
        st.markdown("- 인기 업종 기반 교차 판매 프로모션 활성화")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if len(years) >= 2:
            st.markdown("#### 2. 성장 업종 기반 추천")
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown(f"**{first_year}년 대비 {last_year}년 가장 성장한 업종**")
            
            for i, (category, data) in enumerate(top_growth_categories.iterrows()):
                st.markdown(f"{i+1}. **{category}** - 성장률 {data['성장률']:,.1f}%")
            
            st.markdown("**마케팅 추천**")
            st.markdown(f"- 급성장 중인 업종 ({', '.join(top_growth_categories.index[:2])})에 마케팅 예산 우선 배정")
            st.markdown("- 트렌드 변화에 맞춘 신규 서비스 기획")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 3. 고객 세그먼트 기반 추천")
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**연령대별 선호 업종**")
        
        for _, row in top_by_age.iterrows():
            st.markdown(f"- **{row['연령대']}**: {row['온라인업종']}")
        
        st.markdown("**성별 선호 업종**")
        for _, row in top_by_gender.iterrows():
            st.markdown(f"- **{row['성별']}**: {row['온라인업종']}")
        
        st.markdown("**타겟 마케팅 추천**")
        for _, row in top_by_age.iterrows():
            st.markdown(f"- {row['연령대']} 타겟: **{row['온라인업종']}** 중심 맞춤형 프로모션")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("#### 4. 시즌별 마케팅 추천")
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        
        # 계절별(분기별) 인기 업종 분석
        filtered_data['분기'] = filtered_data['월'].apply(lambda x: str((int(x)-1)//3 + 1))
        seasonal_preference = filtered_data.groupby(['분기', '온라인업종'])['카드이용금액계'].sum().reset_index()
        top_by_season = seasonal_preference.loc[seasonal_preference.groupby('분기')['카드이용금액계'].idxmax()]
        
        st.markdown("**분기별 인기 업종**")
        for _, row in top_by_season.iterrows():
            quarter_name = f"{row['분기']}분기"
            if row['분기'] == '1':
                quarter_name += " (1-3월)"
            elif row['분기'] == '2':
                quarter_name += " (4-6월)"
            elif row['분기'] == '3':
                quarter_name += " (7-9월)"
            else:
                quarter_name += " (10-12월)"
            
            st.markdown(f"- **{quarter_name}**: {row['온라인업종']}")
        
        st.markdown("**시즌별 마케팅 전략**")
        for _, row in top_by_season.iterrows():
            if row['분기'] == '1':
                st.markdown(f"- 1Q (겨울-봄): **{row['온라인업종']}** 신년/봄맞이 프로모션")
            elif row['분기'] == '2':
                st.markdown(f"- 2Q (봄-여름): **{row['온라인업종']}** 여름 준비 프로모션")
            elif row['분기'] == '3':
                st.markdown(f"- 3Q (여름-가을): **{row['온라인업종']}** 휴가/개학 시즌 프로모션")
            else:
                st.markdown(f"- 4Q (가을-겨울): **{row['온라인업종']}** 연말/홀리데이 프로모션")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 종합 추천 섹션
    st.markdown("### 전략적 마케팅 종합 추천")
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    
    # 시기적 특성과 인구통계를 결합한 타겟 마케팅 추천
    current_month = datetime.now().month
    current_quarter = (current_month - 1) // 3 + 1
    
    # 현재 분기의 인기 업종
    current_q_top = top_by_season[top_by_season['분기'] == str(current_quarter)]
    
    if len(current_q_top) > 0:
        current_q_category = current_q_top.iloc[0]['온라인업종']
        
        # 해당 업종의 주요 소비층 확인
        target_category_data = filtered_data[filtered_data['온라인업종'] == current_q_category]
        
        age_dist = target_category_data.groupby('연령대')['카드이용금액계'].sum()
        top_age = age_dist.idxmax()
        
        gender_dist = target_category_data.groupby('성별')['카드이용금액계'].sum()
        top_gender = gender_dist.idxmax() if len(gender_dist) > 0 else "알 수 없음"
        
        # 지역 분포
        district_dist = target_category_data.groupby('고객행정동코드')['카드이용금액계'].sum()
        top_districts = district_dist.nlargest(3).index.tolist()
        
        # 추천 전략 표시
        st.markdown(f"#### 현재 시즌 ({current_quarter}분기) 최적 마케팅 전략")
        st.markdown(f"- **핵심 타겟 업종**: {current_q_category}")
        st.markdown(f"- **주요 고객층**: {top_age}, {top_gender}")
        st.markdown(f"- **핵심 지역**: {', '.join(top_districts[:3])}")
        
        st.markdown("#### 추천 액션 플랜")
        st.markdown(f"1. **{current_q_category}** 업종 소비자 대상 맞춤형 프로모션 개발")
        st.markdown(f"2. **{top_age}, {top_gender}** 타겟팅한 브랜드 메시지 및 크리에이티브 최적화")
        st.markdown(f"3. **{top_districts[0]}** 지역 중심 오프라인 마케팅 활동 강화")
        
        # 번들 추천
        if len(top_categories_overall) >= 2:
            bundle_partner = None
            for cat in top_categories_overall.index:
                if cat != current_q_category:
                    bundle_partner = cat
                    break
            
            if bundle_partner:
                st.markdown(f"4. **교차 판매 전략**: {current_q_category} + {bundle_partner} 번들 상품 개발")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 데이터 기반 의사결정을 위한 캘린더 뷰
    st.markdown("### 연간 마케팅 캘린더 뷰")
    
    # 월별 인기 업종 데이터
    monthly_top = filtered_data.groupby(['월', '온라인업종'])['카드이용금액계'].sum().reset_index()
    top_by_month = monthly_top.loc[monthly_top.groupby('월')['카드이용금액계'].idxmax()]
    
    # 캘린더 뷰 데이터 생성
    calendar_data = []
    
    for month in range(1, 13):
        month_str = f"{month:02d}"
        month_top = top_by_month[top_by_month['월'] == month_str]
        
        if len(month_top) > 0:
            top_category = month_top.iloc[0]['온라인업종']
            
            # 월별 프로모션 추천
            promo_text = ""
            if month in [1, 2]:
                promo_text = "신년/설날 특별 프로모션"
            elif month in [3, 4, 5]:
                promo_text = "봄 시즌 프로모션"
            elif month in [6, 7, 8]:
                promo_text = "여름/휴가 시즌 프로모션"
            elif month in [9, 10]:
                promo_text = "가을/추석 시즌 프로모션"
            else:
                promo_text = "연말/크리스마스 프로모션"
            
            calendar_data.append({
                "월": month,
                "월_표시": f"{month}월",
                "인기업종": top_category,
                "추천프로모션": promo_text
            })
    
    # 캘린더 뷰 표시
    if calendar_data:
        calendar_df = pd.DataFrame(calendar_data)
        
        # 월별 인기 업종 히트맵
        month_cat_pivot = pd.pivot_table(
            monthly_top,
            values='카드이용금액계',
            index='온라인업종',
            columns='월'
        ).fillna(0)
        
        # 히트맵 정규화
        for col in month_cat_pivot.columns:
            month_cat_pivot[col] = month_cat_pivot[col] / month_cat_pivot[col].sum()
        
        fig = px.imshow(
            month_cat_pivot,
            title='월별 업종 소비 비중 히트맵',
            labels=dict(x="월", y="업종", color="소비 비중"),
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # 마케팅 캘린더 표시 - 수정된 부분
        st.markdown("#### 월별 마케팅 추천 캘린더")
        st.dataframe(
            calendar_df[['월_표시', '인기업종', '추천프로모션']]
        ) 
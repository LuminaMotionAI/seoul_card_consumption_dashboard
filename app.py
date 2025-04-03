import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
from matplotlib.ticker import FuncFormatter
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="서울시민 카드 소비 분석",
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
    .section {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.markdown('<div class="main-header">서울시민 카드 소비 데이터 분석</div>', unsafe_allow_html=True)

# Generate sample data function (same as in the original script)
def generate_sample_data(n_samples=10000):
    # Define constants
    categories = [
        '음식점', '카페', '패션', '마트/슈퍼', '교통', '문화/여가', '미용', 
        '의료', '교육', '가전/전자', '스포츠/레저', '주유', '숙박', '기타'
    ]
    
    districts = [
        '강남구', '서초구', '송파구', '종로구', '중구', '용산구', '마포구', 
        '영등구', '성동구', '광진구', '동대문구', '성북구', '강북구', '도봉구', 
        '노원구', '은평구', '서대문구', '강서구', '양천구', '구로구', '금천구', 
        '영등포구', '동작구', '관악구', '강동구'
    ]
    
    age_groups = ['20대', '30대', '40대', '50대', '60대 이상']
    genders = ['남성', '여성']
    
    # Set date range (3 years: 2021, 2022, 2023)
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - start_date).days
    
    # Create customer IDs
    customer_ids = [f'CUST_{i:05d}' for i in range(1, 501)]
    
    # Generate data
    data = {
        'transaction_id': [f'TX_{i:07d}' for i in range(1, n_samples+1)],
        'date': [(start_date + timedelta(days=random.randint(0, date_range))).strftime('%Y-%m-%d') for _ in range(n_samples)],
        'customer_id': [random.choice(customer_ids) for _ in range(n_samples)],
        'category': [random.choice(categories) for _ in range(n_samples)],
        'district': [random.choice(districts) for _ in range(n_samples)],
        'age_group': [random.choice(age_groups) for _ in range(n_samples)],
        'gender': [random.choice(genders) for _ in range(n_samples)]
    }
    
    # Generate amounts with realistic patterns
    amounts = []
    for cat in data['category']:
        if cat == '마트/슈퍼':
            amounts.append(random.randint(10000, 100000))
        elif cat in ['패션', '가전/전자']:
            amounts.append(random.randint(30000, 300000))
        elif cat in ['음식점', '카페']:
            amounts.append(random.randint(5000, 50000))
        elif cat == '교육':
            amounts.append(random.randint(50000, 500000))
        else:
            amounts.append(random.randint(5000, 150000))
    
    data['amount'] = amounts
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Convert date to datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Create additional features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    
    # Create day name
    day_mapping = {0: '월요일', 1: '화요일', 2: '수요일', 3: '목요일', 4: '금요일', 5: '토요일', 6: '일요일'}
    df['day_name'] = df['day_of_week'].map(day_mapping)
    
    # Create season
    season_mapping = {1: '1분기(겨울-봄)', 2: '2분기(봄-여름)', 3: '3분기(여름-가을)', 4: '4분기(가을-겨울)'}
    df['season'] = df['quarter'].map(season_mapping)
    
    return df

# Sidebar
st.sidebar.header('데이터 생성 설정')
n_samples = st.sidebar.slider('샘플 데이터 수', min_value=1000, max_value=100000, value=50000, step=1000)
data_load_state = st.sidebar.text('데이터 생성 중...')
df = generate_sample_data(n_samples)
data_load_state.text(f'데이터 생성 완료: {df.shape[0]}개 레코드')

# Sidebar navigation
st.sidebar.header('메뉴 선택')
analysis_option = st.sidebar.radio(
    '분석 카테고리 선택',
    ['데이터 개요', '소비 트렌드 분석', '업종별 소비 분석', '지역별 소비 분석', '소비자 분석', '시간별 소비 패턴']
)

# Main content based on selection
if analysis_option == '데이터 개요':
    st.markdown('<div class="sub-header">데이터 개요</div>', unsafe_allow_html=True)
    
    # Data summary
    col1, col2 = st.columns(2)
    with col1:
        st.write("### 데이터 기본 정보")
        st.write(f"- 총 거래 건수: {df.shape[0]:,}건")
        st.write(f"- 분석 기간: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")
        st.write(f"- 총 소비 금액: {df['amount'].sum():,}원")
        st.write(f"- 평균 소비 금액: {df['amount'].mean():,.0f}원")
        st.write(f"- 최소 소비 금액: {df['amount'].min():,}원")
        st.write(f"- 최대 소비 금액: {df['amount'].max():,}원")
        
    with col2:
        st.write("### 데이터셋 통계")
        st.dataframe(df.describe())
    
    # Sample data
    st.write("### 데이터 샘플")
    st.dataframe(df.head(10))
    
    # Distribution of amount
    st.write("### 소비 금액 분포")
    fig = px.histogram(df, x="amount", nbins=50, title="소비 금액 분포")
    fig.update_layout(width=800, height=500)
    st.plotly_chart(fig)

elif analysis_option == '소비 트렌드 분석':
    st.markdown('<div class="sub-header">소비 트렌드 분석</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["연도별 트렌드", "월별 트렌드", "분기별 업종 트렌드"])
    
    with tab1:
        # Annual consumption trend
        annual_trend = df.groupby('year')['amount'].agg(['sum', 'count', 'mean']).reset_index()
        annual_trend['sum_million'] = annual_trend['sum'] / 1_000_000  # 단위: 백만원
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(annual_trend, x='year', y='sum_million', 
                         title='연도별 총 소비 금액',
                         labels={'year': '연도', 'sum_million': '총 소비 금액 (백만원)'})
            st.plotly_chart(fig)
            
        with col2:
            fig = px.bar(annual_trend, x='year', y='mean', 
                         title='연도별 평균 소비 금액',
                         labels={'year': '연도', 'mean': '평균 소비 금액 (원)'})
            st.plotly_chart(fig)
        
        # Growth rate
        annual_change = annual_trend.copy()
        annual_change['growth_rate'] = annual_change['sum'].pct_change() * 100
        
        st.write("### 연도별 소비 증감률")
        fig = px.bar(annual_change[1:], x='year', y='growth_rate',
                    title='연도별 소비 증감률 (%)',
                    labels={'year': '연도', 'growth_rate': '증감률 (%)'})
        st.plotly_chart(fig)
    
    with tab2:
        # Monthly consumption trend
        monthly_trend = df.groupby(['year', 'month'])['amount'].sum().reset_index()
        monthly_trend['amount_million'] = monthly_trend['amount'] / 1_000_000  # 단위: 백만원
        
        fig = px.line(monthly_trend, x='month', y='amount_million', color='year', markers=True,
                     title='월별 소비 트렌드',
                     labels={'month': '월', 'amount_million': '총 소비 금액 (백만원)', 'year': '연도'})
        fig.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(1, 13))))
        st.plotly_chart(fig)
        
        # Heatmap of monthly consumption
        monthly_pivot = monthly_trend.pivot(index='month', columns='year', values='amount_million')
        fig = px.imshow(monthly_pivot, 
                      labels=dict(x="연도", y="월", color="소비 금액 (백만원)"),
                      x=monthly_pivot.columns, y=monthly_pivot.index,
                      aspect="auto", title="월별 소비 히트맵")
        st.plotly_chart(fig)
    
    with tab3:
        # Quarterly category trends
        quarterly_cat_trend = df.groupby(['year', 'quarter', 'category'])['amount'].sum().reset_index()
        quarterly_cat_trend['amount_million'] = quarterly_cat_trend['amount'] / 1_000_000  # 단위: 백만원
        
        # Top 5 categories by amount
        top_categories = df.groupby('category')['amount'].sum().nlargest(5).index.tolist()
        st.write(f"### 상위 5개 업종별 분기별 트렌드")
        st.write(f"상위 5개 업종: {', '.join(top_categories)}")
        
        # Filter for selected categories
        selected_categories = st.multiselect('분석할 업종 선택', options=top_categories, default=top_categories)
        
        if selected_categories:
            quarterly_cat_filtered = quarterly_cat_trend[quarterly_cat_trend['category'].isin(selected_categories)]
            
            # Plot for each year
            selected_year = st.selectbox('연도 선택', options=df['year'].unique())
            
            year_data = quarterly_cat_filtered[quarterly_cat_filtered['year'] == selected_year]
            
            fig = px.line(year_data, x='quarter', y='amount_million', color='category', markers=True,
                         title=f'{selected_year}년 분기별 업종 소비 트렌드',
                         labels={'quarter': '분기', 'amount_million': '총 소비 금액 (백만원)', 'category': '업종'})
            fig.update_layout(xaxis=dict(tickmode='array', tickvals=[1, 2, 3, 4]))
            st.plotly_chart(fig)

# Main function to run the app
if __name__ == "__main__":
    pass 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# Function to generate sample data
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
    
    return df

# Create sample data
print("생성 중인 서울시 카드 소비 샘플 데이터...")
df = generate_sample_data(50000)
print(f"샘플 데이터 생성 완료: {df.shape[0]}개의 거래 데이터")

# 기본 데이터 정보 확인
print("\n데이터 기본 정보:")
print(df.info())

print("\n데이터 첫 5개 행:")
print(df.head())

print("\n기술 통계:")
print(df.describe())

print("\n데이터 결측치:")
print(df.isnull().sum())

# 소비 트렌드 분석
print("\n\n===== 소비 트렌드 분석 =====")

# 연도별 카드 소비 트렌드
annual_trend = df.groupby('year')['amount'].agg(['sum', 'count', 'mean']).reset_index()
annual_trend['sum'] = annual_trend['sum'] / 1_000_000  # 단위: 백만원

plt.figure(figsize=(14, 8))
ax1 = plt.subplot(1, 2, 1)
sns.barplot(x='year', y='sum', data=annual_trend, ax=ax1)
ax1.set_title('연도별 총 소비 금액')
ax1.set_xlabel('연도')
ax1.set_ylabel('총 소비 금액 (백만원)')
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))

ax2 = plt.subplot(1, 2, 2)
sns.barplot(x='year', y='mean', data=annual_trend, ax=ax2)
ax2.set_title('연도별 거래당 평균 소비 금액')
ax2.set_xlabel('연도')
ax2.set_ylabel('평균 소비 금액 (원)')
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
plt.tight_layout()
plt.savefig('yearly_consumption_trend.png')
plt.close()

# 월별 소비 트렌드
monthly_trend = df.groupby(['year', 'month'])['amount'].sum().reset_index()
monthly_trend['amount'] = monthly_trend['amount'] / 1_000_000  # 단위: 백만원

plt.figure(figsize=(14, 8))
sns.lineplot(x='month', y='amount', hue='year', data=monthly_trend, marker='o')
plt.title('월별 소비 트렌드')
plt.xlabel('월')
plt.ylabel('총 소비 금액 (백만원)')
plt.xticks(range(1, 13))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='연도')
plt.tight_layout()
plt.savefig('monthly_consumption_trend.png')
plt.close()

# 분기별 업종별 소비 트렌드
quarterly_cat_trend = df.groupby(['year', 'quarter', 'category'])['amount'].sum().reset_index()
quarterly_cat_trend['amount'] = quarterly_cat_trend['amount'] / 1_000_000  # 단위: 백만원

# 소비 금액 기준 상위 5개 업종
top_categories = df.groupby('category')['amount'].sum().nlargest(5).index.tolist()
quarterly_cat_trend_top5 = quarterly_cat_trend[quarterly_cat_trend['category'].isin(top_categories)]

plt.figure(figsize=(16, 10))
for year in df['year'].unique():
    plt.subplot(1, len(df['year'].unique()), list(df['year'].unique()).index(year) + 1)
    year_data = quarterly_cat_trend_top5[quarterly_cat_trend_top5['year'] == year]
    sns.lineplot(x='quarter', y='amount', hue='category', data=year_data, marker='o')
    plt.title(f'{year}년 분기별 상위 5개 업종 소비 트렌드')
    plt.xlabel('분기')
    plt.ylabel('총 소비 금액 (백만원)')
    plt.xticks([1, 2, 3, 4])
    plt.legend(title='업종', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('quarterly_top5_categories_trend.png')
plt.close()

# 업종별 소비 비교 분석
print("\n\n===== 업종별 소비 비교 분석 =====")

# 업종별 총 소비 금액
category_sum = df.groupby('category')['amount'].sum().sort_values(ascending=False)
category_sum = category_sum / 1_000_000  # 단위: 백만원

plt.figure(figsize=(14, 8))
sns.barplot(x=category_sum.index, y=category_sum.values)
plt.title('업종별 총 소비 금액')
plt.xlabel('업종')
plt.ylabel('총 소비 금액 (백만원)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('category_total_consumption.png')
plt.close()

# 업종별 거래 건수
category_count = df['category'].value_counts()
total_transactions = len(df)
category_percentage = (category_count / total_transactions) * 100

plt.figure(figsize=(14, 8))
ax1 = plt.subplot(1, 2, 1)
sns.barplot(x=category_count.index, y=category_count.values, ax=ax1)
ax1.set_title('업종별 거래 건수')
ax1.set_xlabel('업종')
ax1.set_ylabel('거래 건수')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

ax2 = plt.subplot(1, 2, 2)
plt.pie(category_percentage, labels=category_percentage.index, autopct='%1.1f%%', startangle=90)
ax2.set_title('업종별 거래 비율')
plt.axis('equal')
plt.tight_layout()
plt.savefig('category_transaction_count.png')
plt.close()

# 업종별 평균 소비 금액
category_mean = df.groupby('category')['amount'].mean().sort_values(ascending=False)

plt.figure(figsize=(14, 8))
sns.barplot(x=category_mean.index, y=category_mean.values)
plt.title('업종별 평균 소비 금액')
plt.xlabel('업종')
plt.ylabel('평균 소비 금액 (원)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('category_average_amount.png')
plt.close()

# 지역별 소비 경향 분석
print("\n\n===== 지역별 소비 경향 분석 =====")

# 구별 총 소비 금액
district_sum = df.groupby('district')['amount'].sum().sort_values(ascending=False)
district_sum = district_sum / 1_000_000  # 단위: 백만원

plt.figure(figsize=(14, 8))
sns.barplot(x=district_sum.index, y=district_sum.values)
plt.title('서울시 구별 총 소비 금액')
plt.xlabel('구')
plt.ylabel('총 소비 금액 (백만원)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('district_total_consumption.png')
plt.close()

# 구별 인기 업종 (구별 소비 금액이 가장 많은 업종)
district_category = df.groupby(['district', 'category'])['amount'].sum().reset_index()
top_category_by_district = district_category.loc[district_category.groupby('district')['amount'].idxmax()]

plt.figure(figsize=(16, 8))
sns.barplot(x='district', y='amount', hue='category', data=top_category_by_district)
plt.title('구별 최다 소비 업종')
plt.xlabel('구')
plt.ylabel('소비 금액 (원)')
plt.xticks(rotation=45, ha='right')
plt.legend(title='업종', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('district_top_category.png')
plt.close()

# 소비자 분석
print("\n\n===== 소비자 분석 =====")

# 연령대별 소비 금액 및 거래 건수
age_analysis = df.groupby('age_group').agg(
    총소비금액=('amount', 'sum'),
    평균소비금액=('amount', 'mean'),
    거래건수=('transaction_id', 'count')
).reset_index()

age_analysis['총소비금액'] = age_analysis['총소비금액'] / 1_000_000  # 단위: 백만원

plt.figure(figsize=(16, 6))
ax1 = plt.subplot(1, 3, 1)
sns.barplot(x='age_group', y='총소비금액', data=age_analysis, ax=ax1)
ax1.set_title('연령대별 총 소비 금액')
ax1.set_xlabel('연령대')
ax1.set_ylabel('총 소비 금액 (백만원)')

ax2 = plt.subplot(1, 3, 2)
sns.barplot(x='age_group', y='평균소비금액', data=age_analysis, ax=ax2)
ax2.set_title('연령대별 평균 소비 금액')
ax2.set_xlabel('연령대')
ax2.set_ylabel('평균 소비 금액 (원)')

ax3 = plt.subplot(1, 3, 3)
sns.barplot(x='age_group', y='거래건수', data=age_analysis, ax=ax3)
ax3.set_title('연령대별 거래 건수')
ax3.set_xlabel('연령대')
ax3.set_ylabel('거래 건수')

plt.tight_layout()
plt.savefig('age_group_analysis.png')
plt.close()

# 성별 소비 금액 및 거래 건수
gender_analysis = df.groupby('gender').agg(
    총소비금액=('amount', 'sum'),
    평균소비금액=('amount', 'mean'),
    거래건수=('transaction_id', 'count')
).reset_index()

gender_analysis['총소비금액'] = gender_analysis['총소비금액'] / 1_000_000  # 단위: 백만원

plt.figure(figsize=(16, 6))
ax1 = plt.subplot(1, 3, 1)
sns.barplot(x='gender', y='총소비금액', data=gender_analysis, ax=ax1)
ax1.set_title('성별 총 소비 금액')
ax1.set_xlabel('성별')
ax1.set_ylabel('총 소비 금액 (백만원)')

ax2 = plt.subplot(1, 3, 2)
sns.barplot(x='gender', y='평균소비금액', data=gender_analysis, ax=ax2)
ax2.set_title('성별 평균 소비 금액')
ax2.set_xlabel('성별')
ax2.set_ylabel('평균 소비 금액 (원)')

ax3 = plt.subplot(1, 3, 3)
sns.barplot(x='gender', y='거래건수', data=gender_analysis, ax=ax3)
ax3.set_title('성별 거래 건수')
ax3.set_xlabel('성별')
ax3.set_ylabel('거래 건수')

plt.tight_layout()
plt.savefig('gender_analysis.png')
plt.close()

# 연령대별 선호 업종 (연령대별 소비 금액이 가장 많은 업종)
age_category = df.groupby(['age_group', 'category'])['amount'].sum().reset_index()
top_category_by_age = age_category.loc[age_category.groupby('age_group')['amount'].idxmax()]

plt.figure(figsize=(14, 8))
sns.barplot(x='age_group', y='amount', hue='category', data=top_category_by_age)
plt.title('연령대별 최다 소비 업종')
plt.xlabel('연령대')
plt.ylabel('소비 금액 (원)')
plt.legend(title='업종', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('age_group_top_category.png')
plt.close()

# 성별 선호 업종 (성별 소비 금액이 가장 많은 업종)
gender_category = df.groupby(['gender', 'category'])['amount'].sum().reset_index()
gender_category_pivot = gender_category.pivot(index='category', columns='gender', values='amount')
gender_category_pivot = gender_category_pivot.div(gender_category_pivot.sum()) * 100  # 백분율로 변환

plt.figure(figsize=(14, 10))
gender_category_pivot.plot(kind='bar')
plt.title('성별 업종 선호도 비교')
plt.xlabel('업종')
plt.ylabel('소비 비율 (%)')
plt.legend(title='성별')
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('gender_category_preference.png')
plt.close()

# 요일별/시간대별 소비 패턴
print("\n\n===== 요일별/계절별 소비 패턴 =====")

# 요일별 소비 패턴
day_mapping = {0: '월요일', 1: '화요일', 2: '수요일', 3: '목요일', 4: '금요일', 5: '토요일', 6: '일요일'}
df['day_name'] = df['day_of_week'].map(day_mapping)

day_consumption = df.groupby('day_name')['amount'].agg(['sum', 'mean', 'count']).reset_index()
day_order = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
day_consumption['day_name'] = pd.Categorical(day_consumption['day_name'], categories=day_order, ordered=True)
day_consumption = day_consumption.sort_values('day_name')
day_consumption['sum'] = day_consumption['sum'] / 1_000_000  # 단위: 백만원

plt.figure(figsize=(16, 6))
ax1 = plt.subplot(1, 3, 1)
sns.barplot(x='day_name', y='sum', data=day_consumption, ax=ax1)
ax1.set_title('요일별 총 소비 금액')
ax1.set_xlabel('요일')
ax1.set_ylabel('총 소비 금액 (백만원)')

ax2 = plt.subplot(1, 3, 2)
sns.barplot(x='day_name', y='mean', data=day_consumption, ax=ax2)
ax2.set_title('요일별 평균 소비 금액')
ax2.set_xlabel('요일')
ax2.set_ylabel('평균 소비 금액 (원)')

ax3 = plt.subplot(1, 3, 3)
sns.barplot(x='day_name', y='count', data=day_consumption, ax=ax3)
ax3.set_title('요일별 거래 건수')
ax3.set_xlabel('요일')
ax3.set_ylabel('거래 건수')

plt.tight_layout()
plt.savefig('day_of_week_consumption.png')
plt.close()

# 계절별 소비 패턴 (분기 기준)
season_mapping = {1: '1분기(겨울-봄)', 2: '2분기(봄-여름)', 3: '3분기(여름-가을)', 4: '4분기(가을-겨울)'}
df['season'] = df['quarter'].map(season_mapping)

season_consumption = df.groupby(['year', 'season'])['amount'].sum().reset_index()
season_consumption['amount'] = season_consumption['amount'] / 1_000_000  # 단위: 백만원

plt.figure(figsize=(14, 8))
sns.barplot(x='season', y='amount', hue='year', data=season_consumption)
plt.title('계절별 총 소비 금액')
plt.xlabel('계절')
plt.ylabel('총 소비 금액 (백만원)')
plt.legend(title='연도')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('seasonal_consumption.png')
plt.close()

# 계절별 인기 업종
season_category = df.groupby(['season', 'category'])['amount'].sum().reset_index()
top_category_by_season = season_category.loc[season_category.groupby('season')['amount'].idxmax()]

plt.figure(figsize=(14, 8))
sns.barplot(x='season', y='amount', hue='category', data=top_category_by_season)
plt.title('계절별 최다 소비 업종')
plt.xlabel('계절')
plt.ylabel('소비 금액 (원)')
plt.legend(title='업종', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('seasonal_top_category.png')
plt.close()

# 종합 결론
print("\n\n===== 종합 결론 =====")
print("1. 소비 트렌드 분석 결과")
# 연도별 증가/감소 분석
annual_change = annual_trend.copy()
annual_change['growth_rate'] = annual_change['sum'].pct_change() * 100
print(f"  - 연도별 소비 증감률: {annual_change['growth_rate'].to_list()[1:]}%")

# 상위 3개 업종
top3_categories = category_sum.head(3)
print(f"  - 소비 금액 기준 상위 3개 업종: {', '.join(top3_categories.index.to_list())}")

# 상위 3개 지역
top3_districts = district_sum.head(3)
print(f"  - 소비 금액 기준 상위 3개 지역: {', '.join(top3_districts.index.to_list())}")

# 연령대별 선호 업종
print("  - 연령대별 선호 업종:")
for _, row in top_category_by_age.iterrows():
    print(f"    * {row['age_group']}: {row['category']}")

# 성별 선호 업종
top_male = gender_category[gender_category['gender'] == '남성'].sort_values('amount', ascending=False)['category'].iloc[0]
top_female = gender_category[gender_category['gender'] == '여성'].sort_values('amount', ascending=False)['category'].iloc[0]
print(f"  - 성별 선호 업종:")
print(f"    * 남성: {top_male}")
print(f"    * 여성: {top_female}")

# 요일별/계절별 소비 패턴
max_day = day_consumption.loc[day_consumption['sum'].idxmax(), 'day_name']
print(f"  - 소비가 가장 많은 요일: {max_day}")

max_season = season_consumption.groupby('season')['amount'].sum().idxmax()
print(f"  - 소비가 가장 많은 계절: {max_season}")

print("\n분석이 완료되었습니다. 결과를 확인하려면 생성된 그래프 파일을 참조하세요.") 
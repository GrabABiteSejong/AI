import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, util

pd.set_option('display.unicode.east_asian_width', True)

def print_minimal_style(df, query):
    # 데이터 정리
    df_print = df.drop_duplicates(subset=['과목명'])[['학년', '학기', '이수구분', '과목명', 'score']].copy()
    df_print['score'] = df_print['score'].round(1)
    
    print(f"\n맞춤 교과목 가이드\n")
    
    # 구분선 없이 공백으로만 출력 (justify='left'로 정렬 통일)
    print(df_print.to_string(index=False, justify='left'))
    print("\n" + "="*60) # 하단 마무리선만 하나

# 1. 데이터 로드 및 전처리
df_raw = pd.read_csv('인공지능데이터사이언스학과.csv')
df = df_raw[df_raw['학년'] != 1].copy().reset_index(drop=True)

# '과목명'과 '키워드'를 합쳐 검색용 텍스트 생성 
df['search_text'] = df['과목명'] + " " + df['키워드'].fillna("")

# 2. BM25 초기화 (Sparse Retrieval)
tokenized_corpus = [doc.split(" ") for doc in df['search_text'].tolist()]
bm25 = BM25Okapi(tokenized_corpus)

# 3. 임베딩 모델 로드 (Dense Retrieval - 한국어 특화 모델)
# 해커톤 시연용으로 가장 적합한 모델입니다.
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
corpus_embeddings = model.encode(df['search_text'].tolist())

def get_hybrid_recommendation(query, alpha=0.3, top_k=5):
    """
    query: 학생의 희망 진로 또는 관심 키워드
    alpha: 0에 가까울수록 BM25(키워드) 비중 높음, 1에 가까울수록 Vector(의미) 비중 높음
    """
    # BM25 점수 산출
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # 벡터 유사도 점수 산출
    query_embedding = model.encode(query)
    vector_scores = util.cos_sim(query_embedding, corpus_embeddings).flatten().numpy()

    # 정규화
    scaler = MinMaxScaler()
    bm25_normalized = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
    vector_normalized = scaler.fit_transform(vector_scores.reshape(-1, 1)).flatten()

    # 가중합산
    # 학생의 질문이 구체적일수록 alpha를 낮추고, 추상적일수록 alpha를 높이는 전략2..
    2
    
    final_scores = (alpha * vector_normalized) + ((1 - alpha) * bm25_normalized)
    calibrated_scores = np.sqrt(final_scores)
    # 결과 데이터프레임 생성

    df['score'] = calibrated_scores*100
    results = df.sort_values(by='score', ascending=False).head(top_k)
    
    return results[['학년', '학기', '이수구분', '과목명', '키워드', 'score']]

print("=== 세종대 AI 진로-교과목 로드맵 추천 시스템 ===")
print("종료하려면 '나가기'를 입력하세요.")

while True:
    # 사용자로부터 직접 입력 받기
    student_query = input("\n질문을 입력하세요 (예: 데이터 분석가 진로 추천): ")
    
    if student_query == "나가기":
        print("프로그램을 종료합니다.")
        break
    
    if not student_query.strip():
        continue


    # 하이브리드 추천 함수 호출
    results = get_hybrid_recommendation(student_query, alpha=0.3, top_k=5)
    
    print(f"\n[ '{student_query}' 에 대한 추천 결과 ]")
    print_minimal_style(results, student_query)



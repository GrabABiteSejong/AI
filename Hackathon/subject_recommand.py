import pandas as pd
import numpy as np
import os
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, util

# --- 설정 및 전역 변수 ---
pd.set_option('display.unicode.east_asian_width', True)
MODEL_NAME = 'jhgan/ko-sroberta-multitask'
DATA_DIR = './[25해커톤]_세종대_야르'  # CSV 파일들이 저장된 폴더명

def print_minimal_style(df, query):
    """결과를 깔끔하게 출력하는 함수"""  
    # 중복 제거 및 필요한 컬럼만 선택
    df_print = df.drop_duplicates(subset=['과목명'])[['학년', '학기', '이수구분', '과목명', 'score']].copy()
    df_print['score'] = df_print['score'].round(1)
    
    print(f"\n✨ [ {query} ] 맞춤 교과목 가이드")
    print("-" * 65)
    
    # 인덱스 없이 정렬하여 출력
    print(df_print.to_string(index=False, justify='middle'))
    print("\n" + "="*65)

def load_department_data(file_path):
    """선택된 학과의 데이터를 로드하고 검색 엔진(BM25, Embedding)을 초기화합니다."""
    print(f"\n[시스템] 데이터를 분석 중입니다. 잠시만 기다려 주세요...")
    
    # 데이터 로드
    df_raw = pd.read_csv(file_path)
    
    # 1학년 제외 및 전처리 (학년 대표님 요청 사항 반영)
    df = df_raw[df_raw['학년'] != 1].copy().reset_index(drop=True)
    
    # 검색용 텍스트 생성 (과목명 + 키워드)
    df['search_text'] = df['과목명'] + " " + df['키워드'].fillna("")
    
    # 1. BM25 초기화 (키워드 기반)
    tokenized_corpus = [doc.split(" ") for doc in df['search_text'].tolist()]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 2. SBERT 임베딩 초기화 (의미 기반)
    model = SentenceTransformer(MODEL_NAME)
    corpus_embeddings = model.encode(df['search_text'].tolist(), show_progress_bar=False)
    
    return df, bm25, model, corpus_embeddings

def get_hybrid_recommendation(query, df, bm25, model, corpus_embeddings, alpha=0.3, top_k=5):
    """BM25와 Vector 유사도를 결합한 하이브리드 추천"""
    # BM25 점수
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # 벡터 유사도 점수
    query_embedding = model.encode(query)
    vector_scores = util.cos_sim(query_embedding, corpus_embeddings).flatten().numpy()

    # 점수 정규화 (0~1 사이로 변환)
    scaler = MinMaxScaler()
    bm25_norm = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
    vector_norm = scaler.fit_transform(vector_scores.reshape(-1, 1)).flatten()

    # 가중 합산 (alpha 비중에 따라 조절)
    final_scores = (alpha * vector_norm) + ((1 - alpha) * bm25_norm)
    
    # 가시성을 위해 100점 만점으로 스케일링
    df['score'] = np.sqrt(final_scores) * 100
    
    # 상위 결과 반환
    results = df.sort_values(by='score', ascending=False).head(top_k)
    return results

# --- 메인 루프 ---

def main():
    # 1. 폴더 및 파일 확인
    if not os.path.exists(DATA_DIR):
        print(f"Error: '{DATA_DIR}' 폴더를 찾을 수 없습니다. 폴더를 생성하고 CSV를 넣어주세요.")
        return

    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if not csv_files:
        print("Error: 폴더 내에 CSV 파일이 없습니다.")
        return

    # 학과 이름 맵핑 (파일명에서 확장자 제거)
    dept_map = {f.replace('.csv', ''): f for f in csv_files}

    print("=== 세종대학교 전공 로드맵 추천 시스템 ===")
    print(f"현재 로드된 학과: {', '.join(dept_map.keys())}")

    # 2. 학과 선택 (이름 입력 및 키워드 매칭)
    selected_dept = None
    selected_file = None

    while True:
        user_input = input("\n분석을 원하는 학과 이름을 입력하세요: ").strip()
        
        # 정확히 일치하는 경우
        if user_input in dept_map:
            selected_dept = user_input
            selected_file = os.path.join(DATA_DIR, dept_map[user_input])
            break
        
        # 키워드 포함 여부로 검색 (예: '인공' -> '인공지능데이터사이언스학과')
        matches = [d for d in dept_map.keys() if user_input in d]
        
        if len(matches) == 1:
            selected_dept = matches[0]
            selected_file = os.path.join(DATA_DIR, dept_map[selected_dept])
            print(f"-> [ {selected_dept} ] 가 선택되었습니다.")
            break
        elif len(matches) > 1:
            print(f"여러 개의 학과가 검색되었습니다: {', '.join(matches)}")
            print("더 정확한 이름을 입력해 주세요.")
        else:
            print("해당하는 학과를 찾을 수 없습니다. 다시 입력해 주세요.")

    # 3. 데이터 로드
    df, bm25, model, corpus_embeddings = load_department_data(selected_file)

    # 4. 추천 루프
    print(f"\n✅ {selected_dept} 추천 모드가 시작되었습니다.")
    print("종료하려면 '나가기'를 입력하세요.")

    while True:
        query = input(f"\n[{selected_dept}] 진로/관심 분야를 입력하세요: ").strip()
        
        if query == "나가기":
            print("프로그램을 종료합니다.")
            break
        if not query:
            continue

        # 하이브리드 추천 실행
        results = get_hybrid_recommendation(query, df, bm25, model, corpus_embeddings)
        
        # 결과 출력
        print_minimal_style(results, query)

if __name__ == "__main__":
    main()
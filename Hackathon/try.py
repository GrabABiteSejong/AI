from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, util
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI(title="세종대 진로 로드맵 API")

origins = [
    "http://localhost:3000",    # React/Vue 개발 서버 기본 주소
    "http://127.0.0.1:3000",   # 브라우저에 따라 127.0.0.1로 접속하는 경우 대비
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,             # ["*"] 대신 특정 주소 리스트 입력
    allow_credentials=True,            # 쿠키나 인증 정보 포함 허용 여부
    allow_methods=["*"],               # GET, POST 등 모든 HTTP 메서드 허용
    allow_headers=["*"],               # 모든 HTTP 헤더 허용
)

# --- 초기 설정 ---
MODEL_NAME = 'jhgan/ko-sroberta-multitask'
DATA_DIR = './[25해커톤]_세종대_야르'
model = SentenceTransformer(MODEL_NAME)

# 전역 변수로 데이터 관리 (메모리 효율을 위해)
class DepartmentManager:
    def __init__(self):
        self.dept_map = {}
        self.update_dept_map()

    def update_dept_map(self):
        if os.path.exists(DATA_DIR):
            files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
            self.dept_map = {f.replace('.csv', ''): f for f in files}

manager = DepartmentManager()

# --- 데이터 모델 ---
class RecommendRequest(BaseModel):
    department: str  # 프론트에서 선택된 학과 이름
    query: str       # 사용자의 자연어 질문

# --- 유틸리티 함수 (기존 로직) ---
def get_recommendations(dept_name, query):
    file_name = manager.dept_map.get(dept_name)
    if not file_name:
        matches = [d for d in manager.dept_map.keys() if dept_name in d]
        if not matches: return None
        file_name = manager.dept_map[matches[0]]

    df_raw = pd.read_csv(os.path.join(DATA_DIR, file_name))
    df = df_raw[df_raw['학년'] != 1].copy().reset_index(drop=True)
    df['search_text'] = df['과목명'] + " " + df['키워드'].fillna("")

    tokenized_corpus = [doc.split(" ") for doc in df['search_text'].tolist()]
    bm25 = BM25Okapi(tokenized_corpus)
    corpus_embeddings = model.encode(df['search_text'].tolist(), show_progress_bar=False)

    bm25_scores = bm25.get_scores(query.split(" "))
    query_embedding = model.encode(query)
    vector_scores = util.cos_sim(query_embedding, corpus_embeddings).flatten().numpy()

    scaler = MinMaxScaler()
    bm25_norm = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
    vector_norm = scaler.fit_transform(vector_scores.reshape(-1, 1)).flatten()

    final_scores = (0.4 * vector_norm) + (0.6 * bm25_norm)
    df['score'] = np.sqrt(final_scores) * 100
    
    results = df.sort_values(by='score', ascending=False).drop_duplicates(subset=['과목명']).head(8)
    
    # --- 핵심 역량 문구 생성 로직 수정 ---
    # 1. 모든 유효한 키워드를 리스트로 추출 (NaN 및 공백 제거)
    all_valid_keywords = [k.strip() for k in results['키워드'].tolist() if pd.notna(k) and str(k).strip() != '']
    
    # 2. 중복 키워드 제거
    unique_keywords = list(dict.fromkeys(all_valid_keywords))

    # 3. 랜덤하게 2개 선택 (키워드가 2개 미만일 경우 처리 포함)
    if len(unique_keywords) >= 2:
        selected_keywords = random.sample(unique_keywords, 2)
    else:
        selected_keywords = unique_keywords

    # 4. 문장 생성
    if selected_keywords:
        keyword_str = ", ".join(selected_keywords)
        competency = f"{keyword_str} 중심의 {query} 전문성 강화"
    else:
        competency = f"{query} 분야 전공 심화 역량 확보"
    # ---------------------------------------
    
    return results.to_dict(orient='records'), competency

# --- API 엔드포인트 ---

@app.get("/departments")
async def get_departments():
    """프론트엔드 드롭다운 메뉴용 학과 리스트 반환"""
    manager.update_dept_map()
    return {"departments": list(manager.dept_map.keys())}

@app.post("/recommend")
async def create_roadmap(request: RecommendRequest):
    """선택 학과와 질문을 받아 로드맵 결과 반환"""
    results, competency = get_recommendations(request.department, request.query)
    
    if results is None:
        raise HTTPException(status_code=404, detail="학과를 찾을 수 없습니다.")
    
    return {
        "target_path": request.query,
        "competency": competency,
        "roadmap": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

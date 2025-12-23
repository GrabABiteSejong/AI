from fastapi import FastAPI
from pydantic import BaseModel

# 1. 서버 객체 생성
app = FastAPI()

# 2. 데이터 규격 정의 (백엔드 팀원과 약속하는 부분)
class StudentData(BaseModel):
    name: str
    interest: str  # 예: "AI", "데이터 분석", "앱 개발"

# 3. 민석님이 짤 AI 로직 (여기에 이론적 지식을 녹이세요!)
def my_ai_recommender(interest: str):
    # 실제로는 여기서 민석님이 공부한 추천 알고리즘이 돌아갑니다.
    if "AI" in interest:
        return "인공지능 소모임 '아롬(Arom)' 프로젝트 참여를 추천합니다!"
    elif "데이터" in interest:
        return "세종대 공공데이터 분석 경진대회를 준비해보세요."
    else:
        return "SW/AI 해커톤에서 다양한 경험을 쌓는 것을 추천합니다."

# 4. API 엔드포인트 생성 (백엔드가 접속할 주소)
@app.post("/recommend")
async def recommend(data: StudentData):
    # 백엔드에서 받은 데이터 중 'interest'만 추출해서 AI 로직에 전달
    result = my_ai_recommender(data.interest)
    
    # 결과 전송
    return {
        "status": "success",
        "student_name": data.name,
        "recommendation": result
    }

# 5. 실행 확인용 (서버가 잘 켜졌는지 확인)
@app.get("/")
async def root():
    return {"message": "세종대 해커톤 AI 서버가 가동 중입니다!"}
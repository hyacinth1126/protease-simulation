# Streamlit Cloud 배포 가이드

## 1단계: GitHub 리포지토리 생성
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/protease-simulation.git
git push -u origin main
```

## 2단계: Streamlit Cloud 배포
1. https://streamlit.io/cloud 접속
2. GitHub 계정으로 로그인
3. "New app" 클릭
4. Repository: protease-simulation 선택
5. Main file path: app.py
6. Deploy 클릭

**배포 시간: 2-3분**
**URL**: https://yourusername-protease-simulation-app-xxxxx.streamlit.app

## 3단계: 커스텀 도메인 (선택사항)
- 무료: .streamlit.app 서브도메인
- 유료: 커스텀 도메인 연결 가능

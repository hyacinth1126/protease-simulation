#!/usr/bin/env python3
"""
FastAPI 버전의 Kgp Protease 키네틱 분석 API
더 빠르고 확장 가능한 웹 서비스
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import odeint
import json
import io

app = FastAPI(title="Kgp Protease Kinetics API", version="1.0.0")

# CORS 설정 (프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 모델 정의
class KineticDataInput(BaseModel):
    substrate_concentrations: List[float]
    reaction_rates: List[float]

class KineticParameters(BaseModel):
    vmax: float
    km: float
    vmax_std: float
    km_std: float
    r_squared: float

class SimulationInput(BaseModel):
    initial_substrate: float
    simulation_time: float
    vmax: float
    km: float
    time_points: int = 100

class SimulationResult(BaseModel):
    time_points: List[float]
    substrate_concentrations: List[float]
    product_concentrations: List[float]
    final_conversion: float

# 키네틱 분석 클래스 (기존 코드 재사용)
class KgpProteaseKinetics:
    @staticmethod
    def michaelis_menten(substrate_conc, vmax, km):
        return (vmax * substrate_conc) / (km + substrate_conc)
    
    @staticmethod
    def fit_kinetic_parameters(substrate_conc: np.ndarray, reaction_rates: np.ndarray) -> dict:
        try:
            # Michaelis-Menten 피팅
            popt, pcov = curve_fit(
                KgpProteaseKinetics.michaelis_menten, 
                substrate_conc, 
                reaction_rates,
                p0=[max(reaction_rates), np.median(substrate_conc)]
            )
            
            vmax, km = popt
            param_std = np.sqrt(np.diag(pcov))
            
            # R² 계산
            y_pred = KgpProteaseKinetics.michaelis_menten(substrate_conc, vmax, km)
            ss_res = np.sum((reaction_rates - y_pred) ** 2)
            ss_tot = np.sum((reaction_rates - np.mean(reaction_rates)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return {
                'vmax': float(vmax),
                'km': float(km), 
                'vmax_std': float(param_std[0]),
                'km_std': float(param_std[1]),
                'r_squared': float(r_squared)
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"파라미터 피팅 오류: {str(e)}")
    
    @staticmethod
    def time_course_ode(y, t, vmax, km):
        substrate_conc = y[0]
        reaction_rate = KgpProteaseKinetics.michaelis_menten(substrate_conc, vmax, km)
        dSdt = -reaction_rate
        dPdt = reaction_rate
        return [dSdt, dPdt]
    
    @staticmethod
    def simulate_time_course(initial_substrate: float, time_points: np.ndarray, vmax: float, km: float):
        initial_conditions = [initial_substrate, 0]
        solution = odeint(
            KgpProteaseKinetics.time_course_ode, 
            initial_conditions, 
            time_points, 
            args=(vmax, km)
        )
        return solution[:, 0], solution[:, 1]  # substrate, product

# API 엔드포인트
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>Kgp Protease Kinetics API</title></head>
        <body>
            <h1>🧬 Kgp Protease 키네틱 분석 API</h1>
            <p>정밀한 protease 키네틱 계산을 위한 RESTful API</p>
            <h2>사용 가능한 엔드포인트:</h2>
            <ul>
                <li><a href="/docs">/docs - API 문서</a></li>
                <li>POST /analyze - 키네틱 파라미터 분석</li>
                <li>POST /simulate - 시간 경과 시뮬레이션</li>
                <li>POST /upload-csv - CSV 파일 분석</li>
            </ul>
        </body>
    </html>
    """

@app.post("/analyze", response_model=KineticParameters)
async def analyze_kinetics(data: KineticDataInput):
    """키네틱 파라미터 분석"""
    try:
        substrate_conc = np.array(data.substrate_concentrations)
        reaction_rates = np.array(data.reaction_rates)
        
        # 데이터 검증
        if len(substrate_conc) != len(reaction_rates):
            raise HTTPException(status_code=400, detail="기질 농도와 반응 속도 데이터 길이가 다릅니다")
        
        if len(substrate_conc) < 3:
            raise HTTPException(status_code=400, detail="최소 3개의 데이터 포인트가 필요합니다")
        
        # 키네틱 파라미터 계산
        kinetics = KgpProteaseKinetics()
        params = kinetics.fit_kinetic_parameters(substrate_conc, reaction_rates)
        
        return KineticParameters(**params)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate", response_model=SimulationResult)
async def simulate_reaction(data: SimulationInput):
    """시간 경과 시뮬레이션"""
    try:
        time_points = np.linspace(0, data.simulation_time, data.time_points)
        
        kinetics = KgpProteaseKinetics()
        substrate_conc, product_conc = kinetics.simulate_time_course(
            data.initial_substrate, time_points, data.vmax, data.km
        )
        
        final_conversion = (data.initial_substrate - substrate_conc[-1]) / data.initial_substrate * 100
        
        return SimulationResult(
            time_points=time_points.tolist(),
            substrate_concentrations=substrate_conc.tolist(),
            product_concentrations=product_conc.tolist(),
            final_conversion=float(final_conversion)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-csv")
async def upload_csv_analysis(file: UploadFile = File(...)):
    """CSV 파일 업로드 및 분석"""
    try:
        # CSV 파일 읽기
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # 필수 컬럼 확인
        required_columns = ['substrate_concentration', 'reaction_rate']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(status_code=400, detail=f"필수 컬럼이 누락되었습니다: {required_columns}")
        
        # 데이터 전처리
        df_clean = df.dropna().drop_duplicates()
        
        # 키네틱 분석
        kinetics = KgpProteaseKinetics()
        params = kinetics.fit_kinetic_parameters(
            df_clean['substrate_concentration'].values,
            df_clean['reaction_rate'].values
        )
        
        return {
            "data_points": len(df_clean),
            "kinetic_parameters": params,
            "data_preview": df_clean.head().to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {"status": "healthy", "message": "Kgp Protease API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

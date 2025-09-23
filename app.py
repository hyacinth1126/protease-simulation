#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kgp Protease 키네틱 시뮬레이션 앱
peptide substrate의 분해 반응을 모델링하고 시뮬레이션하는 프로그램
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

class KgpProteaseKinetics:
    """Kgp Protease 키네틱 모델링 클래스"""
    
    def __init__(self):
        self.km = None  # Michaelis constant
        self.vmax = None  # Maximum velocity
        self.ki = None  # Inhibition constant (if applicable)
        self.kcat = None  # Turnover number
        self.enzyme_conc = None  # Enzyme concentration
        
    def michaelis_menten(self, substrate_conc, vmax, km):
        """미카엘리스-멘텐 방정식"""
        return (vmax * substrate_conc) / (km + substrate_conc)
    
    def competitive_inhibition(self, substrate_conc, inhibitor_conc, vmax, km, ki):
        """경쟁적 억제 모델"""
        apparent_km = km * (1 + inhibitor_conc / ki)
        return (vmax * substrate_conc) / (apparent_km + substrate_conc)
    
    def time_course_ode(self, y, t, params):
        """시간 경과에 따른 기질 농도 변화 ODE"""
        substrate_conc = y[0]
        product_conc = y[1]
        
        vmax, km = params
        
        # Michaelis-Menten 반응 속도
        reaction_rate = self.michaelis_menten(substrate_conc, vmax, km)
        
        # dS/dt = -v, dP/dt = +v
        dSdt = -reaction_rate
        dPdt = reaction_rate
        
        return [dSdt, dPdt]
    
    def simulate_time_course(self, initial_substrate, time_points, vmax, km):
        """시간 경과 시뮬레이션"""
        initial_conditions = [initial_substrate, 0]  # [substrate, product]
        params = [vmax, km]
        
        solution = odeint(self.time_course_ode, initial_conditions, time_points, args=(params,))
        
        substrate_conc = solution[:, 0]
        product_conc = solution[:, 1]
        
        return substrate_conc, product_conc
    
    def fit_kinetic_parameters(self, substrate_conc, reaction_rates):
        """실험 데이터로부터 키네틱 파라미터 추정"""
        try:
            # Michaelis-Menten 모델 피팅
            popt, pcov = curve_fit(self.michaelis_menten, substrate_conc, reaction_rates,
                                 p0=[max(reaction_rates), np.median(substrate_conc)])
            
            self.vmax, self.km = popt
            
            # 파라미터 불확실성 계산
            param_std = np.sqrt(np.diag(pcov))
            
            return {
                'vmax': self.vmax,
                'km': self.km,
                'vmax_std': param_std[0],
                'km_std': param_std[1],
                'r_squared': self.calculate_r_squared(substrate_conc, reaction_rates, popt)
            }
        except Exception as e:
            st.error(f"파라미터 피팅 중 오류 발생: {str(e)}")
            return None
    
    def calculate_r_squared(self, x_data, y_data, params):
        """R² 값 계산"""
        y_pred = self.michaelis_menten(x_data, *params)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        return 1 - (ss_res / ss_tot)

class DataProcessor:
    """실험 데이터 처리 클래스"""
    
    @staticmethod
    def load_csv_data(file_path):
        """CSV 파일에서 실험 데이터 로드"""
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            st.error(f"CSV 파일 로드 중 오류 발생: {str(e)}")
            return None
    
    @staticmethod
    def validate_data(data):
        """데이터 유효성 검사"""
        required_columns = ['substrate_concentration', 'reaction_rate']
        
        if not all(col in data.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in data.columns]
            st.error(f"필수 컬럼이 누락되었습니다: {missing_cols}")
            return False
        
        # 음수 값 체크
        if (data['substrate_concentration'] < 0).any() or (data['reaction_rate'] < 0).any():
            st.warning("음수 값이 포함되어 있습니다. 데이터를 확인해주세요.")
        
        return True
    
    @staticmethod
    def preprocess_data(data):
        """데이터 전처리"""
        # 결측값 제거
        data_clean = data.dropna()
        
        # 중복값 제거
        data_clean = data_clean.drop_duplicates()
        
        # 농도순으로 정렬
        data_clean = data_clean.sort_values('substrate_concentration')
        
        return data_clean

class Visualizer:
    """데이터 시각화 클래스"""
    
    @staticmethod
    def plot_michaelis_menten(substrate_conc, reaction_rates, kinetics_model, fit_params=None):
        """미카엘리스-멘텐 플롯 생성"""
        fig = go.Figure()
        
        # 실험 데이터 점
        fig.add_trace(go.Scatter(
            x=substrate_conc,
            y=reaction_rates,
            mode='markers',
            name='실험 데이터',
            marker=dict(size=8, color='blue')
        ))
        
        # 피팅 곡선
        if fit_params:
            x_fit = np.linspace(0, max(substrate_conc) * 1.2, 100)
            y_fit = kinetics_model.michaelis_menten(x_fit, fit_params['vmax'], fit_params['km'])
            
            fig.add_trace(go.Scatter(
                x=x_fit,
                y=y_fit,
                mode='lines',
                name=f'M-M 피팅 (Vmax={fit_params["vmax"]:.3f}, Km={fit_params["km"]:.3f})',
                line=dict(color='red', width=2)
            ))
        
        fig.update_layout(
            title='Kgp Protease 키네틱 분석 - 미카엘리스-멘텐 플롯',
            xaxis_title='기질 농도 (μM)',
            yaxis_title='반응 속도 (μM/min)',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_lineweaver_burk(substrate_conc, reaction_rates):
        """Lineweaver-Burk 플롯 (1/v vs 1/[S])"""
        inv_substrate = 1 / substrate_conc
        inv_rate = 1 / reaction_rates
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=inv_substrate,
            y=inv_rate,
            mode='markers+lines',
            name='Lineweaver-Burk',
            marker=dict(size=8, color='green')
        ))
        
        fig.update_layout(
            title='Lineweaver-Burk 플롯',
            xaxis_title='1/[S] (1/μM)',
            yaxis_title='1/v (min/μM)',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_time_course(time_points, substrate_conc, product_conc):
        """시간 경과 시뮬레이션 플롯"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=substrate_conc,
            mode='lines',
            name='기질 농도',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=product_conc,
            mode='lines',
            name='생성물 농도',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='Kgp Protease 반응 시간 경과',
            xaxis_title='시간 (분)',
            yaxis_title='농도 (μM)',
            template='plotly_white'
        )
        
        return fig

def create_sample_data():
    """샘플 데이터 생성"""
    # 실제 Kgp protease의 대략적인 키네틱 파라미터를 사용
    substrate_concentrations = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])
    
    # 시뮬레이션된 반응 속도 (노이즈 추가)
    vmax_true = 0.8
    km_true = 25.0
    
    kinetics = KgpProteaseKinetics()
    true_rates = kinetics.michaelis_menten(substrate_concentrations, vmax_true, km_true)
    
    # 실험 오차 추가
    noise = np.random.normal(0, 0.02, len(true_rates))
    reaction_rates = true_rates + noise
    reaction_rates = np.maximum(reaction_rates, 0)  # 음수 방지
    
    sample_data = pd.DataFrame({
        'substrate_concentration': substrate_concentrations,
        'reaction_rate': reaction_rates,
        'time': np.arange(len(substrate_concentrations)),
        'experiment_id': ['exp_1'] * len(substrate_concentrations)
    })
    
    return sample_data

def main():
    """메인 Streamlit 앱"""
    st.set_page_config(
        page_title="Kgp Protease 키네틱 시뮬레이션",
        page_icon="🧬",
        layout="wide"
    )
    
    st.title("🧬 Kgp Protease 키네틱 시뮬레이션")
    st.markdown("---")
    
    # 사이드바 설정
    st.sidebar.title("설정")
    
    # 데이터 입력 방법 선택
    data_source = st.sidebar.selectbox(
        "데이터 소스 선택",
        ["샘플 데이터 사용", "CSV 파일 업로드"]
    )
    
    kinetics_model = KgpProteaseKinetics()
    visualizer = Visualizer()
    data_processor = DataProcessor()
    
    # 데이터 로드
    if data_source == "CSV 파일 업로드":
        uploaded_file = st.sidebar.file_uploader(
            "CSV 파일 선택",
            type=['csv'],
            help="컬럼: substrate_concentration, reaction_rate"
        )
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            
            if data_processor.validate_data(data):
                data = data_processor.preprocess_data(data)
                st.success("데이터가 성공적으로 로드되었습니다!")
            else:
                st.stop()
        else:
            st.info("CSV 파일을 업로드해주세요.")
            st.stop()
    else:
        data = create_sample_data()
        st.info("샘플 데이터를 사용합니다.")
    
    # 데이터 미리보기
    st.subheader("📊 데이터 미리보기")
    st.dataframe(data.head(10))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("데이터 포인트 수", len(data))
    
    with col2:
        st.metric("기질 농도 범위", f"{data['substrate_concentration'].min():.1f} - {data['substrate_concentration'].max():.1f} μM")
    
    # 키네틱 분석
    st.subheader("🔬 키네틱 파라미터 분석")
    
    substrate_conc = data['substrate_concentration'].values
    reaction_rates = data['reaction_rate'].values
    
    # 파라미터 피팅
    fit_params = kinetics_model.fit_kinetic_parameters(substrate_conc, reaction_rates)
    
    if fit_params:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Vmax (μM/min)",
                f"{fit_params['vmax']:.4f}",
                delta=f"±{fit_params['vmax_std']:.4f}"
            )
        
        with col2:
            st.metric(
                "Km (μM)",
                f"{fit_params['km']:.2f}",
                delta=f"±{fit_params['km_std']:.2f}"
            )
        
        with col3:
            st.metric(
                "R² 값",
                f"{fit_params['r_squared']:.4f}"
            )
        
        # 미카엘리스-멘텐 플롯
        st.subheader("📈 미카엘리스-멘텐 플롯")
        mm_plot = visualizer.plot_michaelis_menten(substrate_conc, reaction_rates, kinetics_model, fit_params)
        st.plotly_chart(mm_plot, use_container_width=True)
        
        # Lineweaver-Burk 플롯
        st.subheader("📉 Lineweaver-Burk 플롯")
        lb_plot = visualizer.plot_lineweaver_burk(substrate_conc, reaction_rates)
        st.plotly_chart(lb_plot, use_container_width=True)
    
    # 시간 경과 시뮬레이션
    st.subheader("⏱️ 시간 경과 시뮬레이션")
    
    col1, col2 = st.columns(2)
    
    with col1:
        initial_substrate = st.number_input(
            "초기 기질 농도 (μM)",
            min_value=1.0,
            max_value=1000.0,
            value=100.0,
            step=10.0
        )
    
    with col2:
        simulation_time = st.number_input(
            "시뮬레이션 시간 (분)",
            min_value=1,
            max_value=1440,
            value=60,
            step=5
        )
    
    if fit_params and st.button("시뮬레이션 실행"):
        time_points = np.linspace(0, simulation_time, 100)
        substrate_time, product_time = kinetics_model.simulate_time_course(
            initial_substrate, time_points, fit_params['vmax'], fit_params['km']
        )
        
        time_plot = visualizer.plot_time_course(time_points, substrate_time, product_time)
        st.plotly_chart(time_plot, use_container_width=True)
        
        # 결과 요약
        final_conversion = (initial_substrate - substrate_time[-1]) / initial_substrate * 100
        st.success(f"시뮬레이션 완료! 최종 전환율: {final_conversion:.1f}%")
    
    # 데이터 다운로드
    st.subheader("💾 결과 다운로드")
    
    if fit_params:
        results_df = pd.DataFrame({
            'Parameter': ['Vmax (μM/min)', 'Km (μM)', 'R²'],
            'Value': [fit_params['vmax'], fit_params['km'], fit_params['r_squared']],
            'Std_Error': [fit_params['vmax_std'], fit_params['km_std'], 'N/A']
        })
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="키네틱 파라미터 다운로드 (CSV)",
            data=csv,
            file_name="kgp_kinetic_parameters.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()

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
        
        # FRET 관련 파라미터
        self.donor_quantum_yield = 0.8  # Donor 양자 수율
        self.acceptor_extinction = 50000  # Acceptor 흡광계수 (M⁻¹cm⁻¹)
        self.forster_radius = 5.5  # Förster 반경 (nm)
        self.background_fluorescence = 0.05  # 배경 형광
        
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
    
    def fret_efficiency(self, distance):
        """FRET 효율 계산 (Förster 방정식)"""
        if distance <= 0:
            return 1.0
        return 1 / (1 + (distance / self.forster_radius) ** 6)
    
    def calculate_fluorescence_intensity(self, substrate_conc, initial_substrate, max_intensity=1000):
        """기질 농도에 따른 형광 강도 계산"""
        # 절단된 기질 비율
        cleaved_fraction = (initial_substrate - substrate_conc) / initial_substrate
        
        # FRET가 없을 때의 최대 형광 강도
        max_fluor = max_intensity
        
        # FRET substrate의 경우: 절단되면 FRET가 해제되어 형광 증가
        # 평균 거리 모델: 절단되지 않은 기질은 FRET 상태, 절단된 기질은 형광 방출
        intact_fraction = substrate_conc / initial_substrate
        
        # FRET 효율 (intact substrate에 대해서만 적용)
        fret_efficiency = self.fret_efficiency(2.5)  # 2.5nm 가정 (일반적인 FRET substrate)
        
        # 형광 강도 = 배경 + (절단된 부분의 형광) + (남은 부분의 억제된 형광)
        fluorescence_intensity = (
            self.background_fluorescence * max_fluor +  # 배경 형광
            cleaved_fraction * max_fluor +  # 절단된 부분의 완전한 형광
            intact_fraction * max_fluor * (1 - fret_efficiency)  # 남은 부분의 억제된 형광
        )
        
        return fluorescence_intensity
    
    def simulate_fret_time_course(self, initial_substrate, time_points, vmax, km, max_intensity=1000):
        """FRET 기반 시간 경과 형광 시뮬레이션"""
        substrate_conc, product_conc = self.simulate_time_course(initial_substrate, time_points, vmax, km)
        
        fluorescence_intensities = [
            self.calculate_fluorescence_intensity(s_conc, initial_substrate, max_intensity)
            for s_conc in substrate_conc
        ]
        
        return substrate_conc, product_conc, np.array(fluorescence_intensities)
    
    def fit_fret_parameters(self, time_points, fluorescence_data, initial_substrate, 
                           initial_guess_vmax=1.0, initial_guess_km=50.0):
        """FRET 형광 데이터로부터 키네틱 파라미터 추정"""
        def fluorescence_model(time, vmax, km, max_intensity):
            substrate_conc, _, _ = self.simulate_fret_time_course(
                initial_substrate, time, vmax, km, max_intensity
            )
            return self.calculate_fluorescence_intensity(substrate_conc, initial_substrate, max_intensity)
        
        try:
            from scipy.optimize import curve_fit
            
            # 초기 추정값
            p0 = [initial_guess_vmax, initial_guess_km, max(fluorescence_data)]
            
            # 파라미터 피팅
            popt, pcov = curve_fit(
                fluorescence_model, 
                time_points, 
                fluorescence_data,
                p0=p0,
                maxfev=5000
            )
            
            vmax_fit, km_fit, max_intensity_fit = popt
            param_std = np.sqrt(np.diag(pcov))
            
            # R² 계산
            y_pred = fluorescence_model(time_points, *popt)
            ss_res = np.sum((fluorescence_data - y_pred) ** 2)
            ss_tot = np.sum((fluorescence_data - np.mean(fluorescence_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return {
                'vmax': vmax_fit,
                'km': km_fit,
                'max_intensity': max_intensity_fit,
                'vmax_std': param_std[0],
                'km_std': param_std[1],
                'max_intensity_std': param_std[2],
                'r_squared': r_squared
            }
        except Exception as e:
            st.error(f"FRET 파라미터 피팅 중 오류 발생: {str(e)}")
            return None

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
    def validate_fret_data(data):
        """FRET 데이터 유효성 검사"""
        required_columns = ['time', 'fluorescence_intensity']
        
        if not all(col in data.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in data.columns]
            st.error(f"FRET 데이터 필수 컬럼이 누락되었습니다: {missing_cols}")
            return False
        
        # 시간 데이터 검사
        if (data['time'] < 0).any():
            st.warning("음수 시간 값이 포함되어 있습니다.")
        
        # 형광 데이터 검사
        if (data['fluorescence_intensity'] < 0).any():
            st.warning("음수 형광 강도 값이 포함되어 있습니다.")
        
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
    
    @staticmethod
    def plot_fret_fluorescence(time_points, fluorescence_intensity, experimental_data=None):
        """FRET 형광 강도 시간 경과 플롯"""
        fig = go.Figure()
        
        # 시뮬레이션 데이터
        fig.add_trace(go.Scatter(
            x=time_points,
            y=fluorescence_intensity,
            mode='lines',
            name='FRET 시뮬레이션',
            line=dict(color='purple', width=3)
        ))
        
        # 실험 데이터 (있는 경우)
        if experimental_data is not None:
            fig.add_trace(go.Scatter(
                x=experimental_data['time'],
                y=experimental_data['fluorescence_intensity'],
                mode='markers',
                name='실험 데이터',
                marker=dict(size=8, color='orange', symbol='circle')
            ))
        
        fig.update_layout(
            title='🔬 FRET 기반 Kgp Protease 활성 분석',
            xaxis_title='시간 (분)',
            yaxis_title='형광 강도 (RFU)',
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_fret_comparison(time_points, sim_fluorescence, exp_data, fit_params=None):
        """FRET 시뮬레이션과 실험 데이터 비교 플롯"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('FRET 형광 강도 비교', '잔차 분석'),
            vertical_spacing=0.12
        )
        
        # 상단: 시뮬레이션 vs 실험 데이터
        fig.add_trace(go.Scatter(
            x=time_points,
            y=sim_fluorescence,
            mode='lines',
            name='시뮬레이션',
            line=dict(color='purple', width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=exp_data['time'],
            y=exp_data['fluorescence_intensity'],
            mode='markers',
            name='실험 데이터',
            marker=dict(size=6, color='orange')
        ), row=1, col=1)
        
        # 하단: 잔차 (실험값 - 시뮬레이션값)
        if len(exp_data) > 0:
            # 실험 시간에 해당하는 시뮬레이션 값 보간
            sim_interp = np.interp(exp_data['time'], time_points, sim_fluorescence)
            residuals = exp_data['fluorescence_intensity'] - sim_interp
            
            fig.add_trace(go.Scatter(
                x=exp_data['time'],
                y=residuals,
                mode='markers',
                name='잔차',
                marker=dict(size=6, color='red'),
                showlegend=False
            ), row=2, col=1)
            
            # 0 라인 추가
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # 레이아웃 업데이트
        fig.update_xaxes(title_text="시간 (분)", row=2, col=1)
        fig.update_yaxes(title_text="형광 강도 (RFU)", row=1, col=1)
        fig.update_yaxes(title_text="잔차 (RFU)", row=2, col=1)
        
        title_text = 'FRET 데이터 피팅 결과'
        if fit_params:
            title_text += f" (R² = {fit_params['r_squared']:.4f})"
        
        fig.update_layout(
            title=title_text,
            template='plotly_white',
            height=600
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

def create_sample_fret_data():
    """FRET 실험용 샘플 데이터 생성"""
    # 시간 포인트 (0-60분)
    time_points = np.linspace(0, 60, 25)  # 25개 포인트
    
    # 실제 키네틱 파라미터
    vmax_true = 0.8
    km_true = 25.0
    initial_substrate = 100.0
    max_intensity = 1000.0
    
    kinetics = KgpProteaseKinetics()
    
    # 이론적 형광 강도 계산
    substrate_conc, product_conc, fluorescence_theoretical = kinetics.simulate_fret_time_course(
        initial_substrate, time_points, vmax_true, km_true, max_intensity
    )
    
    # 실험적 노이즈 추가 (형광 측정의 특성상 multiplicative noise 포함)
    noise_additive = np.random.normal(0, 5, len(fluorescence_theoretical))  # 절대 노이즈
    noise_multiplicative = np.random.normal(1, 0.02, len(fluorescence_theoretical))  # 상대 노이즈
    
    fluorescence_experimental = fluorescence_theoretical * noise_multiplicative + noise_additive
    fluorescence_experimental = np.maximum(fluorescence_experimental, 0)  # 음수 방지
    
    fret_sample_data = pd.DataFrame({
        'time': time_points,
        'fluorescence_intensity': fluorescence_experimental,
        'substrate_concentration': substrate_conc,
        'product_concentration': product_conc,
        'fluorescence_theoretical': fluorescence_theoretical,
        'experiment_id': ['fret_exp_1'] * len(time_points)
    })
    
    return fret_sample_data

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
    
    # 분석 모드 선택
    analysis_mode = st.sidebar.selectbox(
        "분석 모드 선택",
        ["일반 키네틱 분석", "FRET 형광 분석"]
    )
    
    # 데이터 입력 방법 선택
    if analysis_mode == "일반 키네틱 분석":
        data_source = st.sidebar.selectbox(
            "데이터 소스 선택",
            ["샘플 데이터 사용", "CSV 파일 업로드"]
        )
    else:  # FRET 분석
        data_source = st.sidebar.selectbox(
            "데이터 소스 선택",
            ["FRET 샘플 데이터 사용", "FRET CSV 파일 업로드"]
        )
    
    kinetics_model = KgpProteaseKinetics()
    visualizer = Visualizer()
    data_processor = DataProcessor()
    
    # 데이터 로드
    if analysis_mode == "일반 키네틱 분석":
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
            
    else:  # FRET 분석
        if data_source == "FRET CSV 파일 업로드":
            uploaded_file = st.sidebar.file_uploader(
                "FRET CSV 파일 선택",
                type=['csv'],
                help="컬럼: time, fluorescence_intensity"
            )
            
            if uploaded_file is not None:
                fret_data = pd.read_csv(uploaded_file)
                
                if data_processor.validate_fret_data(fret_data):
                    fret_data = data_processor.preprocess_data(fret_data)
                    st.success("FRET 데이터가 성공적으로 로드되었습니다!")
                else:
                    st.stop()
            else:
                st.info("FRET CSV 파일을 업로드해주세요.")
                st.stop()
        else:
            fret_data = create_sample_fret_data()
            st.info("FRET 샘플 데이터를 사용합니다.")
    
    # 데이터 미리보기 및 분석 (모드에 따라 분기)
    if analysis_mode == "일반 키네틱 분석":
        st.subheader("📊 데이터 미리보기")
        st.dataframe(data.head(10))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("데이터 포인트 수", len(data))
        
        with col2:
            st.metric("기질 농도 범위", f"{data['substrate_concentration'].min():.1f} - {data['substrate_concentration'].max():.1f} μM")
            
    else:  # FRET 분석
        st.subheader("🔬 FRET 데이터 미리보기")
        st.dataframe(fret_data.head(10))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("데이터 포인트 수", len(fret_data))
        
        with col2:
            st.metric("시간 범위", f"{fret_data['time'].min():.1f} - {fret_data['time'].max():.1f} 분")
        
        with col3:
            st.metric("형광 강도 범위", f"{fret_data['fluorescence_intensity'].min():.0f} - {fret_data['fluorescence_intensity'].max():.0f} RFU")
    
    if analysis_mode == "일반 키네틱 분석":
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
        
    else:  # FRET 분석
        # FRET 키네틱 분석
        st.subheader("🔬 FRET 키네틱 파라미터 분석")
        
        # 사용자 입력: 초기 기질 농도
        col1, col2 = st.columns(2)
        with col1:
            initial_substrate_fret = st.number_input(
                "초기 기질 농도 (μM)",
                min_value=1.0,
                max_value=1000.0,
                value=100.0,
                key="fret_substrate"
            )
        with col2:
            max_intensity_fret = st.number_input(
                "최대 형광 강도 (RFU)",
                min_value=100,
                max_value=10000,
                value=1000,
                key="fret_max_intensity"
            )
        
        # FRET 데이터로부터 키네틱 파라미터 피팅
        time_points_fret = fret_data['time'].values
        fluorescence_fret = fret_data['fluorescence_intensity'].values
        
        if st.button("FRET 데이터 분석 실행", key="fret_analysis"):
            with st.spinner("FRET 키네틱 파라미터를 분석 중입니다..."):
                fret_fit_params = kinetics_model.fit_fret_parameters(
                    time_points_fret, 
                    fluorescence_fret, 
                    initial_substrate_fret
                )
                
                if fret_fit_params:
                    st.success("FRET 파라미터 피팅 완료!")
                    
                    # 피팅 결과 표시
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Vmax (μM/min)",
                            f"{fret_fit_params['vmax']:.4f}",
                            delta=f"±{fret_fit_params['vmax_std']:.4f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Km (μM)",
                            f"{fret_fit_params['km']:.2f}",
                            delta=f"±{fret_fit_params['km_std']:.2f}"
                        )
                    
                    with col3:
                        st.metric(
                            "R² 값",
                            f"{fret_fit_params['r_squared']:.4f}"
                        )
                    
                    with col4:
                        st.metric(
                            "Max 형광 강도",
                            f"{fret_fit_params['max_intensity']:.0f} RFU"
                        )
                    
                    # FRET 시뮬레이션 생성
                    time_sim = np.linspace(0, fret_data['time'].max(), 100)
                    substrate_sim, product_sim, fluorescence_sim = kinetics_model.simulate_fret_time_course(
                        initial_substrate_fret, time_sim, 
                        fret_fit_params['vmax'], fret_fit_params['km'], 
                        fret_fit_params['max_intensity']
                    )
                    
                    # FRET 형광 플롯
                    st.subheader("📈 FRET 형광 강도 분석")
                    fret_plot = visualizer.plot_fret_comparison(
                        time_sim, fluorescence_sim, fret_data, fret_fit_params
                    )
                    st.plotly_chart(fret_plot, use_container_width=True)
                    
                    # 추가 분석 정보
                    st.subheader("📋 FRET 분석 결과 해석")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**키네틱 파라미터**")
                        st.write(f"• **Vmax**: {fret_fit_params['vmax']:.4f} μM/min")
                        st.write(f"• **Km**: {fret_fit_params['km']:.2f} μM")
                        st.write(f"• **kcat/Km**: {(fret_fit_params['vmax']/initial_substrate_fret)/fret_fit_params['km']:.6f} min⁻¹μM⁻¹")
                    
                    with col2:
                        st.write("**FRET 분석 품질**")
                        st.write(f"• **R² 값**: {fret_fit_params['r_squared']:.4f}")
                        quality = "우수" if fret_fit_params['r_squared'] > 0.95 else "양호" if fret_fit_params['r_squared'] > 0.9 else "보통"
                        st.write(f"• **피팅 품질**: {quality}")
                        st.write(f"• **데이터 포인트**: {len(fret_data)}개")
                    
                    # 결과 다운로드
                    st.subheader("💾 FRET 분석 결과 다운로드")
                    
                    fret_results_df = pd.DataFrame({
                        'Parameter': ['Vmax (μM/min)', 'Km (μM)', 'Max Intensity (RFU)', 'R²'],
                        'Value': [fret_fit_params['vmax'], fret_fit_params['km'], 
                                 fret_fit_params['max_intensity'], fret_fit_params['r_squared']],
                        'Std_Error': [fret_fit_params['vmax_std'], fret_fit_params['km_std'], 
                                     fret_fit_params['max_intensity_std'], 'N/A']
                    })
                    
                    csv_fret = fret_results_df.to_csv(index=False)
                    st.download_button(
                        label="FRET 키네틱 파라미터 다운로드 (CSV)",
                        data=csv_fret,
                        file_name="kgp_fret_kinetic_parameters.csv",
                        mime="text/csv",
                        key="download_fret"
                    )
                    
                    fit_params = fret_fit_params  # 시뮬레이션 섹션을 위해
                else:
                    st.error("FRET 파라미터 피팅에 실패했습니다.")
                    fit_params = None
        else:
            # 기본 FRET 형광 플롯 (피팅 전)
            st.subheader("📈 FRET 형광 데이터")
            basic_fret_plot = visualizer.plot_fret_fluorescence(
                fret_data['time'], fret_data['fluorescence_intensity']
            )
            st.plotly_chart(basic_fret_plot, use_container_width=True)
            fit_params = None
    
    # 시간 경과 시뮬레이션 (공통)
    if fit_params:
        st.subheader("⏱️ 시간 경과 시뮬레이션")
        
        col1, col2 = st.columns(2)
        
        with col1:
            initial_substrate = st.number_input(
                "초기 기질 농도 (μM)",
                min_value=1.0,
                max_value=1000.0,
                value=100.0,
                step=10.0,
                key="sim_substrate"
            )
        
        with col2:
            simulation_time = st.number_input(
                "시뮬레이션 시간 (분)",
                min_value=1,
                max_value=1440,
                value=60,
                step=5,
                key="sim_time"
            )
        
        if st.button("시뮬레이션 실행", key="run_simulation"):
            time_points = np.linspace(0, simulation_time, 100)
            
            if analysis_mode == "일반 키네틱 분석":
                substrate_time, product_time = kinetics_model.simulate_time_course(
                    initial_substrate, time_points, fit_params['vmax'], fit_params['km']
                )
                
                time_plot = visualizer.plot_time_course(time_points, substrate_time, product_time)
                st.plotly_chart(time_plot, use_container_width=True)
                
                # 결과 요약
                final_conversion = (initial_substrate - substrate_time[-1]) / initial_substrate * 100
                st.success(f"시뮬레이션 완료! 최종 전환율: {final_conversion:.1f}%")
                
            else:  # FRET 분석
                substrate_time, product_time, fluorescence_time = kinetics_model.simulate_fret_time_course(
                    initial_substrate, time_points, fit_params['vmax'], fit_params['km']
                )
                
                fret_sim_plot = visualizer.plot_fret_fluorescence(time_points, fluorescence_time)
                st.plotly_chart(fret_sim_plot, use_container_width=True)
                
                # 결과 요약
                final_conversion = (initial_substrate - substrate_time[-1]) / initial_substrate * 100
                final_fluorescence = fluorescence_time[-1]
                st.success(f"FRET 시뮬레이션 완료! 최종 전환율: {final_conversion:.1f}%, 최종 형광: {final_fluorescence:.0f} RFU")
        
        # 데이터 다운로드
        st.subheader("💾 결과 다운로드")
        
        if analysis_mode == "일반 키네틱 분석":
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
                mime="text/csv",
                key="download_kinetic"
            )

if __name__ == "__main__":
    main()

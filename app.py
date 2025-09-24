#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protease Kinetic and FRET Dequenching Simulation App
A program for modeling and simulating peptide substrate cleavage reactions
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

# Font configuration for plots
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

class SurfacePeptideKinetics:
    """Surface-bound peptide cleavage kinetic modeling class (MMP9, etc.)"""
    
    def __init__(self):
        # Surface kinetic parameters
        self.kcat = None  # Catalytic rate constant (min⁻¹)
        self.km = None  # Michaelis-Menten constant (nM)
        self.keff = None  # Effective rate constant (min⁻¹)
        
        # Electrode surface parameters
        self.peptide_coverage = 1.33e-10  # Total peptide coverage (mol/cm²)
        self.max_cleavage_fraction = 0.3  # ε - maximum fraction of cleavable peptides
        self.electrode_area = 1.0  # Electrode area (cm²)
        
    def calculate_initial_substrate(self):
        """Calculate initial substrate concentration: [S]₀ = ε[S]ₜ"""
        return self.max_cleavage_fraction * self.peptide_coverage
    
    def surface_reaction_rate(self, time, enzyme_conc, keff):
        """
        Surface reaction rate equation (Equation 1)
        d[P]/dt = keff * [S]₀ * exp(-keff * t)
        """
        s0 = self.calculate_initial_substrate()
        return keff * s0 * np.exp(-keff * time)
    
    def product_concentration(self, time, enzyme_conc, keff):
        """
        Product concentration (Equation 2)
        [P]/[S]ₜ = ε * (1 - exp(-keff * t))
        """
        return self.max_cleavage_fraction * (1 - np.exp(-keff * time))
    
    def effective_rate_constant(self, enzyme_conc, kcat, km):
        """
        Effective rate constant (Equation 3)
        keff = (kcat * [E]) / (Km + [E])
        """
        return (kcat * enzyme_conc) / (km + enzyme_conc)
    
    def signal_suppression(self, time, enzyme_conc, kcat, km):
        """
        신호 억제 계산 (SWV 피크 전류 변화)
        Signal suppression = [P]/[S]ₜ
        """
        keff = self.effective_rate_constant(enzyme_conc, kcat, km)
        return self.product_concentration(time, enzyme_conc, keff)
    
    def fit_surface_kinetics(self, enzyme_concentrations, keff_values):
        """
        표면 Kinetic Parameters 피팅
        1/keff vs 1/[E] 선형화를 통한 kcat, Km 계산
        """
        try:
            # 선형화: 1/keff = (Km/kcat) * (1/[E]) + (1/kcat)
            x_data = 1 / np.array(enzyme_concentrations)  # 1/[E]
            y_data = 1 / np.array(keff_values)  # 1/keff
            
            # 선형 피팅
            coeffs = np.polyfit(x_data, y_data, 1)
            slope = coeffs[0]  # Km/kcat
            intercept = coeffs[1]  # 1/kcat
            
            # 파라미터 계산
            kcat_fitted = 1 / intercept
            km_fitted = slope * kcat_fitted
            
            # R² 계산
            y_pred = slope * x_data + intercept
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return {
                'kcat': kcat_fitted,
                'km': km_fitted,
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared,
                'kcat_km_ratio': kcat_fitted / km_fitted * 1e-9 * 60  # M⁻¹s⁻¹ 단위로 변환
            }
        except Exception as e:
            st.error(f"표면 Kinetic Parameters 피팅 중 오류: {str(e)}")
            return None
    
    def simulate_time_course_surface(self, time_points, enzyme_conc, kcat, km):
        """표면 반응 시간 경과 시뮬레이션"""
        keff = self.effective_rate_constant(enzyme_conc, kcat, km)
        
        # 신호 억제 (생성물 농도 비율)
        signal_suppression = np.array([
            self.signal_suppression(t, enzyme_conc, kcat, km) for t in time_points
        ])
        
        # 남은 기질 농도 비율
        substrate_fraction = 1 - signal_suppression / self.max_cleavage_fraction
        
        return substrate_fraction, signal_suppression, keff

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
        """실험 데이터로부터 Kinetic Parameters 추정"""
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
            st.error(f"Error during parameter fitting: {str(e)}")
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
        """FRET 형광 데이터로부터 Kinetic Parameters 추정"""
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
            st.error(f"FRET Error during parameter fitting: {str(e)}")
            return None

class DataProcessor:
    """Experimental data processing class"""
    
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
            st.error(f"Required columns missing: {missing_cols}")
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
            st.error(f"FRET 데이터 Required columns missing: {missing_cols}")
            return False
        
        # 표준편차 컬럼 검사 (선택적)
        has_std = 'fluorescence_intensity_std' in data.columns
        if has_std:
            st.info("Standard deviation data detected. Error bars will be displayed.")
            # 표준편차 값 검사
            if (data['fluorescence_intensity_std'] < 0).any():
                st.warning("Negative standard deviation values detected.")
        
        # 시간 데이터 검사
        if (data['time'] < 0).any():
            st.warning("Negative time values detected.")
        
        # 형광 데이터 검사
        if (data['fluorescence_intensity'] < 0).any():
            st.warning("Negative fluorescence intensity values detected.")
        
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
            title='Protease Kinetic Analysis - Michaelis-Menten Plot',
            xaxis_title='Substrate Concentration (μM)',
            yaxis_title='Reaction Rate (μM/min)',
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
            title='Lineweaver-Burk Plot',
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
            title='Protease Reaction Time Course',
            xaxis_title='Time (min)',
            yaxis_title='Concentration (μM)',
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
            # 표준편차 데이터가 있는지 확인
            has_std = 'fluorescence_intensity_std' in experimental_data.columns
            
            if has_std:
                # 에러바가 있는 실험 데이터
                fig.add_trace(go.Scatter(
                    x=experimental_data['time'],
                    y=experimental_data['fluorescence_intensity'],
                    error_y=dict(
                        type='data',
                        array=experimental_data['fluorescence_intensity_std'],
                        visible=True,
                        color='rgba(255, 165, 0, 0.8)',
                        thickness=2,
                        width=3
                    ),
                    mode='markers',
                    name='실험 데이터 (±SD)',
                    marker=dict(size=8, color='orange', symbol='circle')
                ))
            else:
                # 기본 실험 데이터 (에러바 없음)
                fig.add_trace(go.Scatter(
                    x=experimental_data['time'],
                    y=experimental_data['fluorescence_intensity'],
                    mode='markers',
                    name='실험 데이터',
                    marker=dict(size=8, color='orange', symbol='circle')
                ))
        
        fig.update_layout(
            title='🔬 FRET-based Protease Activity Analysis',
            xaxis_title='Time (min)',
            yaxis_title='Fluorescence Intensity (RFU)',
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_surface_kinetics_lineweaver(enzyme_conc, keff_values, fit_params=None):
        """표면 키네틱 Lineweaver-Burk 타입 플롯 (1/keff vs 1/[E])"""
        fig = go.Figure()
        
        # 실험 데이터 점
        inv_enzyme = 1 / np.array(enzyme_conc)
        inv_keff = 1 / np.array(keff_values)
        
        fig.add_trace(go.Scatter(
            x=inv_enzyme,
            y=inv_keff,
            mode='markers',
            name='실험 데이터',
            marker=dict(size=10, color='blue', symbol='circle')
        ))
        
        # 피팅 직선
        if fit_params:
            x_fit = np.linspace(min(inv_enzyme) * 0.8, max(inv_enzyme) * 1.2, 100)
            y_fit = fit_params['slope'] * x_fit + fit_params['intercept']
            
            fig.add_trace(go.Scatter(
                x=x_fit,
                y=y_fit,
                mode='lines',
                name=f'선형 피팅 (R² = {fit_params["r_squared"]:.4f})',
                line=dict(color='red', width=2)
            ))
        
        fig.update_layout(
            title='🔬 표면 키네틱 분석 - 1/keff vs 1/[E] 플롯',
            xaxis_title='1/[E] (nM⁻¹)',
            yaxis_title='1/keff (min)',
            template='plotly_white',
            annotations=[
                dict(
                    x=0.02, y=0.98,
                    xref='paper', yref='paper',
                    text=f'kcat = {fit_params["kcat"]:.4f} min⁻¹<br>Km = {fit_params["km"]:.2f} nM<br>kcat/Km = {fit_params["kcat_km_ratio"]:.2e} M⁻¹s⁻¹' if fit_params else '',
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )
            ] if fit_params else []
        )
        
        return fig
    
    @staticmethod
    def plot_signal_suppression(time_points, signal_suppression_data, experimental_data=None):
        """신호 억제 시간 경과 플롯"""
        fig = go.Figure()
        
        # 시뮬레이션 데이터
        for i, (enzyme_conc, suppression) in enumerate(signal_suppression_data.items()):
            fig.add_trace(go.Scatter(
                x=time_points,
                y=suppression,
                mode='lines',
                name=f'[E] = {enzyme_conc} nM (시뮬레이션)',
                line=dict(width=2)
            ))
        
        # 실험 데이터 (있는 경우)
        if experimental_data is not None:
            for enzyme_conc, data in experimental_data.items():
                fig.add_trace(go.Scatter(
                    x=data['time'],
                    y=data['signal_suppression'],
                    mode='markers',
                    name=f'[E] = {enzyme_conc} nM (실험)',
                    marker=dict(size=8, symbol='circle-open')
                ))
        
        fig.update_layout(
            title='🔬 MMP9 Surface Peptide Cleavage - Signal Suppression Time Course',
            xaxis_title='Time (min)',
            yaxis_title='Signal Suppression ([P]/[S]ₜ)',
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_surface_reaction_rate(time_points, enzyme_concentrations, surface_kinetics, kcat, km):
        """표면 반응 속도 시간 경과 플롯"""
        fig = go.Figure()
        
        for enzyme_conc in enzyme_concentrations:
            keff = surface_kinetics.effective_rate_constant(enzyme_conc, kcat, km)
            reaction_rates = [surface_kinetics.surface_reaction_rate(t, enzyme_conc, keff) for t in time_points]
            
            fig.add_trace(go.Scatter(
                x=time_points,
                y=reaction_rates,
                mode='lines',
                name=f'[E] = {enzyme_conc} nM',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='🔬 Surface Reaction Rate Time Course',
            xaxis_title='Time (min)',
            yaxis_title='Reaction Rate (d[P]/dt)',
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
        
        # 실험 데이터에 표준편차가 있는지 확인
        has_std = 'fluorescence_intensity_std' in exp_data.columns
        
        if has_std:
            # 에러바가 있는 실험 데이터
            fig.add_trace(go.Scatter(
                x=exp_data['time'],
                y=exp_data['fluorescence_intensity'],
                error_y=dict(
                    type='data',
                    array=exp_data['fluorescence_intensity_std'],
                    visible=True,
                    color='rgba(255, 165, 0, 0.8)',
                    thickness=2,
                    width=3
                ),
                mode='markers',
                name='실험 데이터 (±SD)',
                marker=dict(size=6, color='orange')
            ), row=1, col=1)
        else:
            # 기본 실험 데이터 (에러바 없음)
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
    # 실제 Kgp protease의 대략적인 Kinetic Parameters를 사용
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
    
    # 실제 Kinetic Parameters
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
    
    # 표준편차 계산 (실험적 변동성을 시뮬레이션)
    # 형광 강도에 비례한 표준편차 (일반적으로 형광 측정에서 관찰됨)
    base_std = 5.0  # 기본 표준편차
    proportional_std = fluorescence_experimental * 0.03  # 비례 표준편차 (3%)
    fluorescence_std = base_std + proportional_std
    
    fret_sample_data = pd.DataFrame({
        'time': time_points,
        'fluorescence_intensity': fluorescence_experimental,
        'fluorescence_intensity_std': fluorescence_std,
        'substrate_concentration': substrate_conc,
        'product_concentration': product_conc,
        'fluorescence_theoretical': fluorescence_theoretical,
        'experiment_id': ['fret_exp_1'] * len(time_points)
    })
    
    return fret_sample_data

def create_surface_kinetics_sample_data():
    """표면 키네틱 분석용 샘플 데이터 생성 (MMP9 기준)"""
    # 논문에서 제공된 실제 값들
    kcat_true = 0.0809  # min⁻¹
    km_true = 55.75  # nM
    
    # 다양한 효소 농도
    enzyme_concentrations = np.array([5, 10, 20, 40, 80, 160, 320])  # nM
    
    surface_kinetics = SurfacePeptideKinetics()
    
    # 각 효소 농도에서 keff 계산
    keff_values = []
    for enzyme_conc in enzyme_concentrations:
        keff = surface_kinetics.effective_rate_constant(enzyme_conc, kcat_true, km_true)
        # 실험 오차 추가
        noise = np.random.normal(1, 0.05)  # 5% 상대 오차
        keff_noisy = keff * noise
        keff_values.append(max(keff_noisy, 0.001))  # 최소값 제한
    
    surface_sample_data = pd.DataFrame({
        'enzyme_concentration': enzyme_concentrations,
        'keff': keff_values,
        'experiment_id': ['surface_exp_1'] * len(enzyme_concentrations)
    })
    
    return surface_sample_data

def create_surface_time_course_data():
    """표면 키네틱 시간 경과 샘플 데이터 생성"""
    # 시간 포인트 (0-120분)
    time_points = np.linspace(0, 120, 25)
    
    # 논문 기준 파라미터
    kcat_true = 0.0809  # min⁻¹
    km_true = 55.75  # nM
    enzyme_concentrations = [20, 50, 100]  # nM
    
    surface_kinetics = SurfacePeptideKinetics()
    
    all_data = []
    
    for enzyme_conc in enzyme_concentrations:
        substrate_fraction, signal_suppression, keff = surface_kinetics.simulate_time_course_surface(
            time_points, enzyme_conc, kcat_true, km_true
        )
        
        # 실험 노이즈 추가
        noise_suppression = np.random.normal(0, 0.01, len(signal_suppression))
        signal_suppression_noisy = signal_suppression + noise_suppression
        signal_suppression_noisy = np.clip(signal_suppression_noisy, 0, 0.3)  # 물리적 제한
        
        for i, t in enumerate(time_points):
            all_data.append({
                'time': t,
                'enzyme_concentration': enzyme_conc,
                'signal_suppression': signal_suppression_noisy[i],
                'substrate_fraction': substrate_fraction[i],
                'keff': keff,
                'experiment_id': f'surface_time_exp_{int(enzyme_conc)}'
            })
    
    return pd.DataFrame(all_data)

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Protease Kinetic and FRET Dequenching Simulation",
        page_icon="🧬",
        layout="wide"
    )
    
    st.title("🧬 Protease Kinetic and FRET Dequenching Simulation")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.title("Settings")
    
    # Analysis mode selection
    analysis_mode = st.sidebar.selectbox(
        "Select Analysis Mode",
        ["General Kinetic Analysis", "FRET Fluorescence Analysis", "Surface Peptide Cleavage Analysis (MMP9)"]
    )
    
    # Data input method selection
    if analysis_mode == "General Kinetic Analysis":
        data_source = st.sidebar.selectbox(
            "Select Data Source",
            ["Use Sample Data", "Upload CSV File"]
        )
    elif analysis_mode == "FRET Fluorescence Analysis":
        data_source = st.sidebar.selectbox(
            "Select Data Source",
            ["Use FRET Sample Data", "Upload FRET CSV File"]
        )
    else:  # Surface Peptide Cleavage Analysis
        data_source = st.sidebar.selectbox(
            "Select Data Source",
            ["Surface Kinetic Sample Data", "Surface Time Course Data", "Upload Surface Kinetic CSV"]
        )
    
    kinetics_model = KgpProteaseKinetics()
    surface_kinetics = SurfacePeptideKinetics()
    visualizer = Visualizer()
    data_processor = DataProcessor()
    
    # Initialize fit_params to avoid UnboundLocalError
    fit_params = None
    
    # Data loading
    if analysis_mode == "General Kinetic Analysis":
        if data_source == "Upload CSV File":
            uploaded_file = st.sidebar.file_uploader(
                "Select CSV File",
                type=['csv'],
                help="Columns: substrate_concentration, reaction_rate"
            )
            
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                
                if data_processor.validate_data(data):
                    data = data_processor.preprocess_data(data)
                    st.success("Data loaded successfully!")
                else:
                    st.stop()
            else:
                st.info("Please upload a CSV file.")
                st.stop()
        else:
            data = create_sample_data()
            st.info("Using sample data.")
            
    elif analysis_mode == "FRET Fluorescence Analysis":  # FRET 분석
        if data_source == "Upload FRET CSV File":
            uploaded_file = st.sidebar.file_uploader(
                "FRET CSV 파일 선택",
                type=['csv'],
                help="Required columns: time, fluorescence_intensity\nOptional columns: fluorescence_intensity_std (standard deviation, displays error bars)"
            )
            
            if uploaded_file is not None:
                fret_data = pd.read_csv(uploaded_file)
                
                if data_processor.validate_fret_data(fret_data):
                    fret_data = data_processor.preprocess_data(fret_data)
                    st.success("FRET Data loaded successfully!")
                else:
                    st.stop()
            else:
                st.info("FRET Please upload a CSV file.")
                st.stop()
        else:
            fret_data = create_sample_fret_data()
            st.info("FRET Using sample data.")
    
    elif analysis_mode == "Surface Peptide Cleavage Analysis (MMP9)":  # 표면 펩타이드 절단 분석
        if data_source == "Upload Surface Kinetic CSV":
            uploaded_file = st.sidebar.file_uploader(
                "표면 키네틱 CSV 파일 선택",
                type=['csv'],
                help="Required columns: enzyme_concentration, keff\nOptional columns: experiment_id"
            )
            
            if uploaded_file is not None:
                surface_data = pd.read_csv(uploaded_file)
                
                required_columns = ['enzyme_concentration', 'keff']
                if not all(col in surface_data.columns for col in required_columns):
                    missing_cols = [col for col in required_columns if col not in surface_data.columns]
                    st.error(f"Required columns missing: {missing_cols}")
                    st.stop()
                else:
                    st.success("표면 키네틱 Data loaded successfully!")
            else:
                st.info("표면 키네틱 Please upload a CSV file.")
                st.stop()
        elif data_source == "Surface Time Course Data":
            surface_data = create_surface_time_course_data()
            st.info("표면 키네틱 시간 경과 Using sample data.")
        else:
            surface_data = create_surface_kinetics_sample_data()
            st.info("표면 키네틱 Using sample data.")
    
    # 데이터 미리보기 및 분석 (모드에 따라 분기)
    if analysis_mode == "General Kinetic Analysis":
        st.subheader("📊 Data Preview")
        st.dataframe(data.head(10))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Number of Data Points", len(data))
        
        with col2:
            st.metric("Substrate Concentration Range", f"{data['substrate_concentration'].min():.1f} - {data['substrate_concentration'].max():.1f} μM")
            
    elif analysis_mode == "FRET Fluorescence Analysis":  # FRET 분석
        st.subheader("🔬 FRET Data Preview")
        st.dataframe(fret_data.head(10))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Data Points", len(fret_data))
        
        with col2:
            st.metric("Time Range", f"{fret_data['time'].min():.1f} - {fret_data['time'].max():.1f} 분")
        
        with col3:
            st.metric("Fluorescence Intensity Range", f"{fret_data['fluorescence_intensity'].min():.0f} - {fret_data['fluorescence_intensity'].max():.0f} RFU")
    
    elif analysis_mode == "Surface Peptide Cleavage Analysis (MMP9)":  # 표면 펩타이드 절단 분석
        st.subheader("🔬 Surface Kinetic Data Preview")
        st.dataframe(surface_data.head(10))
        
        if data_source == "표면 시간 경과 데이터":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Number of Data Points", len(surface_data))
            
            with col2:
                st.metric("Time Range", f"{surface_data['time'].min():.1f} - {surface_data['time'].max():.1f} 분")
            
            with col3:
                unique_enzymes = surface_data['enzyme_concentration'].nunique()
                st.metric("Enzyme Concentration Conditions", f"{unique_enzymes}개")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Number of Data Points", len(surface_data))
            
            with col2:
                st.metric("Enzyme Concentration Range", f"{surface_data['enzyme_concentration'].min():.1f} - {surface_data['enzyme_concentration'].max():.1f} nM")
    
    if analysis_mode == "General Kinetic Analysis":
        # 키네틱 분석
        st.subheader("🔬 Kinetic Parameter Analysis")
        
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
            st.subheader("📈 Michaelis-Menten Plot")
            mm_plot = visualizer.plot_michaelis_menten(substrate_conc, reaction_rates, kinetics_model, fit_params)
            st.plotly_chart(mm_plot, use_container_width=True)
            
            # Lineweaver-Burk 플롯
            st.subheader("📉 Lineweaver-Burk Plot")
            lb_plot = visualizer.plot_lineweaver_burk(substrate_conc, reaction_rates)
            st.plotly_chart(lb_plot, use_container_width=True)
        
    elif analysis_mode == "FRET Fluorescence Analysis":  # FRET 분석
        # FRET 키네틱 분석
        st.subheader("🔬 FRET Kinetic Parameter Analysis")
        
        # 사용자 입력: 초기 기질 농도
        col1, col2 = st.columns(2)
        with col1:
            initial_substrate_fret = st.number_input(
                "Initial Substrate Concentration (μM)",
                min_value=1.0,
                max_value=1000.0,
                value=100.0,
                key="fret_substrate"
            )
        with col2:
            max_intensity_fret = st.number_input(
                "Maximum Fluorescence Intensity (RFU)",
                min_value=100,
                max_value=10000,
                value=1000,
                key="fret_max_intensity"
            )
        
        # FRET 데이터로부터 Kinetic Parameters 피팅
        time_points_fret = fret_data['time'].values
        fluorescence_fret = fret_data['fluorescence_intensity'].values
        
        if st.button("Run FRET Data Analysis", key="fret_analysis"):
            with st.spinner("FRET Kinetic Parameters를 분석 중입니다..."):
                fret_fit_params = kinetics_model.fit_fret_parameters(
                    time_points_fret, 
                    fluorescence_fret, 
                    initial_substrate_fret
                )
                
                if fret_fit_params:
                    st.success("FRET parameter fitting completed!")
                    
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
                    st.subheader("📈 FRET Fluorescence Intensity Analysis")
                    fret_plot = visualizer.plot_fret_comparison(
                        time_sim, fluorescence_sim, fret_data, fret_fit_params
                    )
                    st.plotly_chart(fret_plot, use_container_width=True)
                    
                    # 추가 분석 정보
                    st.subheader("📋 FRET Analysis Results Interpretation")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Kinetic Parameters**")
                        st.write(f"• **Vmax**: {fret_fit_params['vmax']:.4f} μM/min")
                        st.write(f"• **Km**: {fret_fit_params['km']:.2f} μM")
                        st.write(f"• **kcat/Km**: {(fret_fit_params['vmax']/initial_substrate_fret)/fret_fit_params['km']:.6f} min⁻¹μM⁻¹")
                    
                    with col2:
                        st.write("**FRET Analysis Quality**")
                        st.write(f"• **R² 값**: {fret_fit_params['r_squared']:.4f}")
                        quality = "Excellent" if fret_fit_params['r_squared'] > 0.95 else "Good" if fret_fit_params['r_squared'] > 0.9 else "Average"
                        st.write(f"• **Fitting Quality**: {quality}")
                        st.write(f"• **데이터 포인트**: {len(fret_data)}개")
                    
                    # 결과 다운로드
                    st.subheader("💾 Download FRET Analysis Results")
                    
                    fret_results_df = pd.DataFrame({
                        'Parameter': ['Vmax (μM/min)', 'Km (μM)', 'Max Intensity (RFU)', 'R²'],
                        'Value': [fret_fit_params['vmax'], fret_fit_params['km'], 
                                 fret_fit_params['max_intensity'], fret_fit_params['r_squared']],
                        'Std_Error': [fret_fit_params['vmax_std'], fret_fit_params['km_std'], 
                                     fret_fit_params['max_intensity_std'], 'N/A']
                    })
                    
                    csv_fret = fret_results_df.to_csv(index=False)
                    st.download_button(
                        label="FRET Kinetic Parameters 다운로드 (CSV)",
                        data=csv_fret,
                        file_name="kgp_fret_kinetic_parameters.csv",
                        mime="text/csv",
                        key="download_fret"
                    )
                    
                    fit_params = fret_fit_params  # 시뮬레이션 섹션을 위해
                else:
                    st.error("FRET parameter fitting failed.")
                    fit_params = None
        else:
            # 기본 FRET 형광 플롯 (피팅 전)
            st.subheader("📈 FRET Fluorescence Data")
            basic_fret_plot = visualizer.plot_fret_fluorescence(
                fret_data['time'], fret_data['fluorescence_intensity']
            )
            st.plotly_chart(basic_fret_plot, use_container_width=True)
            fit_params = None
    
    elif analysis_mode == "Surface Peptide Cleavage Analysis (MMP9)":
        if data_source == "표면 시간 경과 데이터":
            # 시간 경과 데이터 분석
            st.subheader("🔬 Surface Peptide Cleavage Time Course Analysis")
            
            # 효소 농도별 데이터 그룹화
            enzyme_groups = surface_data.groupby('enzyme_concentration')
            
            # 신호 억제 시간 경과 플롯
            signal_suppression_data = {}
            for enzyme_conc, group in enzyme_groups:
                signal_suppression_data[enzyme_conc] = group['signal_suppression'].values
            
            time_points = surface_data[surface_data['enzyme_concentration'] == list(enzyme_groups.groups.keys())[0]]['time'].values
            
            time_course_plot = visualizer.plot_signal_suppression(
                time_points, signal_suppression_data
            )
            st.plotly_chart(time_course_plot, use_container_width=True)
            
            # 각 농도별 통계
            st.subheader("📊 Analysis Results by Enzyme Concentration")
            
            results_data = []
            for enzyme_conc, group in enzyme_groups:
                final_suppression = group['signal_suppression'].iloc[-1]
                max_suppression = group['signal_suppression'].max()
                keff_value = group['keff'].iloc[0]
                
                results_data.append({
                    'Enzyme Concentration (nM)': enzyme_conc,
                    'Final Signal Suppression': f"{final_suppression:.3f}",
                    'Maximum Signal Suppression': f"{max_suppression:.3f}",
                    'keff (min⁻¹)': f"{keff_value:.4f}"
                })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df)
            
        else:
            # Kinetic Parameters 분석
            st.subheader("🔬 Surface Kinetic Parameter Analysis")
            
            enzyme_conc = surface_data['enzyme_concentration'].values
            keff_values = surface_data['keff'].values
            
            # 파라미터 피팅
            fit_params = surface_kinetics.fit_surface_kinetics(enzyme_conc, keff_values)
            
            if fit_params:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "kcat (min⁻¹)",
                        f"{fit_params['kcat']:.4f}"
                    )
                
                with col2:
                    st.metric(
                        "Km (nM)",
                        f"{fit_params['km']:.2f}"
                    )
                
                with col3:
                    st.metric(
                        "kcat/Km (M⁻¹s⁻¹)",
                        f"{fit_params['kcat_km_ratio']:.2e}"
                    )
                
                with col4:
                    st.metric(
                        "R² 값",
                        f"{fit_params['r_squared']:.4f}"
                    )
                
                # Lineweaver-Burk 타입 플롯 (1/keff vs 1/[E])
                st.subheader("📈 Surface Kinetic Analysis - 1/keff vs 1/[E] Plot")
                lineweaver_plot = visualizer.plot_surface_kinetics_lineweaver(
                    enzyme_conc, keff_values, fit_params
                )
                st.plotly_chart(lineweaver_plot, use_container_width=True)
                
                # 분석 결과 해석
                st.subheader("📋 Analysis Results Interpretation")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Kinetic Parameters**")
                    st.write(f"• **kcat**: {fit_params['kcat']:.4f} min⁻¹")
                    st.write(f"• **Km**: {fit_params['km']:.2f} nM")
                    st.write(f"• **kcat/Km**: {fit_params['kcat_km_ratio']:.2e} M⁻¹s⁻¹")
                
                with col2:
                    st.write("**Surface Reaction Properties**")
                    st.write(f"• **R² 값**: {fit_params['r_squared']:.4f}")
                    quality = "Excellent" if fit_params['r_squared'] > 0.95 else "Good" if fit_params['r_squared'] > 0.9 else "Average"
                    st.write(f"• **Fitting Quality**: {quality}")
                    st.write(f"• **펩타이드 피복률**: {surface_kinetics.peptide_coverage:.2e} mol/cm²")
                
                # 논문 비교
                st.info(f"""
                **📚 문헌값 비교 (GPLGMWSRC 펩타이드)**
                - 보고된 kcat: 0.0809 min⁻¹
                - 보고된 Km: 55.75 nM  
                - 보고된 kcat/Km: 2.4 × 10⁴ M⁻¹s⁻¹
                
                **현재 분석 결과**
                - 계산된 kcat: {fit_params['kcat']:.4f} min⁻¹
                - 계산된 Km: {fit_params['km']:.2f} nM
                - 계산된 kcat/Km: {fit_params['kcat_km_ratio']:.2e} M⁻¹s⁻¹
                """)
            else:
                st.error("표면 Kinetic Parameters 피팅에 실패했습니다.")
                fit_params = None
    
    # 시간 경과 시뮬레이션 (공통)
    if fit_params:
        st.subheader("⏱️ Time Course Simulation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            initial_substrate = st.number_input(
                "Initial Substrate Concentration (μM)",
                min_value=1.0,
                max_value=1000.0,
                value=100.0,
                step=10.0,
                key="sim_substrate"
            )
        
        with col2:
            simulation_time = st.number_input(
                "Simulation Time (min)",
                min_value=1,
                max_value=1440,
                value=60,
                step=5,
                key="sim_time"
            )
        
        if st.button("Run Simulation", key="run_simulation"):
            time_points = np.linspace(0, simulation_time, 100)
            
            if analysis_mode == "General Kinetic Analysis":
                substrate_time, product_time = kinetics_model.simulate_time_course(
                    initial_substrate, time_points, fit_params['vmax'], fit_params['km']
                )
                
                time_plot = visualizer.plot_time_course(time_points, substrate_time, product_time)
                st.plotly_chart(time_plot, use_container_width=True)
                
                # 결과 요약
                final_conversion = (initial_substrate - substrate_time[-1]) / initial_substrate * 100
                st.success(f"Simulation completed! Final conversion: {final_conversion:.1f}%")
                
            elif analysis_mode == "FRET Fluorescence Analysis":  # FRET 분석
                substrate_time, product_time, fluorescence_time = kinetics_model.simulate_fret_time_course(
                    initial_substrate, time_points, fit_params['vmax'], fit_params['km']
                )
                
                fret_sim_plot = visualizer.plot_fret_fluorescence(time_points, fluorescence_time)
                st.plotly_chart(fret_sim_plot, use_container_width=True)
                
                # 결과 요약
                final_conversion = (initial_substrate - substrate_time[-1]) / initial_substrate * 100
                final_fluorescence = fluorescence_time[-1]
                st.success(f"FRET Simulation completed! Final conversion: {final_conversion:.1f}%, Final fluorescence: {final_fluorescence:.0f} RFU")
            
            elif analysis_mode == "Surface Peptide Cleavage Analysis (MMP9)":
                # 표면 키네틱 시뮬레이션
                col1, col2 = st.columns(2)
                
                with col1:
                    enzyme_conc_sim = st.number_input(
                        "Enzyme Concentration (nM)",
                        min_value=1.0,
                        max_value=1000.0,
                        value=50.0,
                        step=5.0,
                        key="surface_enzyme"
                    )
                
                with col2:
                    simulation_time = st.number_input(
                        "Simulation Time (min)",
                        min_value=1,
                        max_value=1440,
                        value=120,
                        step=5,
                        key="surface_time"
                    )
                
                time_points = np.linspace(0, simulation_time, 100)
                substrate_fraction, signal_suppression, keff = surface_kinetics.simulate_time_course_surface(
                    time_points, enzyme_conc_sim, fit_params['kcat'], fit_params['km']
                )
                
                # 신호 억제 플롯
                signal_data = {enzyme_conc_sim: signal_suppression}
                surface_sim_plot = visualizer.plot_signal_suppression(time_points, signal_data)
                st.plotly_chart(surface_sim_plot, use_container_width=True)
                
                # 추가 분석 플롯
                st.subheader("📈 Surface Reaction Rate Analysis")
                rate_plot = visualizer.plot_surface_reaction_rate(
                    time_points, [enzyme_conc_sim], surface_kinetics, 
                    fit_params['kcat'], fit_params['km']
                )
                st.plotly_chart(rate_plot, use_container_width=True)
                
                # 결과 요약
                final_suppression = signal_suppression[-1]
                max_suppression = np.max(signal_suppression)
                st.success(f"""
                표면 키네틱 Simulation completed!
                - Final signal suppression: {final_suppression:.3f}
                - Maximum signal suppression: {max_suppression:.3f}
                - keff: {keff:.4f} min⁻¹
                """)
                
                # 실험 설계 도움말
                st.info(f"""
                **💡 실험 설계 가이드라인**
                - 권장 측정 시간: {simulation_time/2:.0f}-{simulation_time:.0f}분
                - 예상 신호 변화: {final_suppression*100:.1f}%
                - Reaction completion: {(final_suppression/surface_kinetics.max_cleavage_fraction)*100:.1f}%
                """)
        
        # 데이터 다운로드
        st.subheader("💾 Download Results")
        
        if analysis_mode == "General Kinetic Analysis":
            results_df = pd.DataFrame({
                'Parameter': ['Vmax (μM/min)', 'Km (μM)', 'R²'],
                'Value': [fit_params['vmax'], fit_params['km'], fit_params['r_squared']],
                'Std_Error': [fit_params['vmax_std'], fit_params['km_std'], 'N/A']
            })
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Kinetic Parameters 다운로드 (CSV)",
                data=csv,
                file_name="kgp_kinetic_parameters.csv",
                mime="text/csv",
                key="download_kinetic"
            )
        
        elif analysis_mode == "Surface Peptide Cleavage Analysis (MMP9)":
            surface_results_df = pd.DataFrame({
                'Parameter': ['kcat (min⁻¹)', 'Km (nM)', 'kcat/Km (M⁻¹s⁻¹)', 'R²', 
                             'Peptide Coverage (mol/cm²)', 'Max Cleavage Fraction'],
                'Value': [fit_params['kcat'], fit_params['km'], fit_params['kcat_km_ratio'], 
                         fit_params['r_squared'], surface_kinetics.peptide_coverage, 
                         surface_kinetics.max_cleavage_fraction]
            })
            
            csv_surface = surface_results_df.to_csv(index=False)
            st.download_button(
                label="표면 Kinetic Parameters 다운로드 (CSV)",
                data=csv_surface,
                file_name="mmp9_surface_kinetic_parameters.csv",
                mime="text/csv",
                key="download_surface"
            )

if __name__ == "__main__":
    main()

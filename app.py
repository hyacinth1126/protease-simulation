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
            st.error(f"FRET 파라미터 피팅 중 오류 발생: {str(e)}")
            return None

class HydrogelPeptideKinetics:
    """Hydrogel-immobilized peptide kinetics with FRET quenching"""
    
    def __init__(self):
        self.k_eff = None  # k_eff (M^-1 s^-1), related to kcat/KM
        self.k0 = None     # Background reaction rate (s^-1)
        
    def fraction_model(self, t, E, k_eff, k0):
        """
        Fraction of cleaved peptide
        X(t) = 1 - exp(-(k_eff*E + k0)*t)
        """
        return 1.0 - np.exp(-(k_eff * E + k0) * t)
    
    def normalize_data(self, df, saturation_threshold=0.85):
        """
        Normalize fluorescence data to fraction X(t) = (F(t) - F0)/(Fmax - F0)
        Group by enzyme concentration
        Determine saturation line and exclude saturation region data
        """
        def normalize_group(g):
            # Support multiple fluor column names
            fluor_col = 'FL_intensity' if 'FL_intensity' in g.columns else 'fluor'
            
            g_sorted = g.sort_values('time_s')
            
            # F0: Use the value at t=0 as baseline
            F0_mask = g_sorted['time_s'] == 0
            if F0_mask.any():
                F0_val = float(g_sorted.loc[F0_mask, fluor_col].iloc[0])
            else:
                F0_val = float(g_sorted[fluor_col].iloc[0])
            
            # Fmax: Use saturation value - average of last few time points
            # Take the average of the last 3-5 points as saturation line
            n_points = min(5, len(g_sorted))
            if n_points >= 3:
                last_points = g_sorted[fluor_col].tail(n_points).values
                Fmax = float(np.mean(last_points))
                
                # Calculate standard deviation to find saturation region
                Fmax_std = float(np.std(last_points))
                Fmax_upper = Fmax + 2 * Fmax_std  # ~95% confidence
                Fmax_lower = Fmax - 2 * Fmax_std
            else:
                Fmax = float(g_sorted[fluor_col].max())
                Fmax_std = 0
                Fmax_upper = Fmax
                Fmax_lower = Fmax
            
            # Ensure Fmax > F0 (avoid division by zero)
            if Fmax <= F0_val:
                # If no change, set Fmax slightly higher
                Fmax = F0_val + 100
                Fmax_upper = Fmax + 50
                Fmax_lower = Fmax - 50
            
            g = g_sorted.copy()
            
            # Calculate X(t)
            g['X'] = (g[fluor_col] - F0_val) / (Fmax - F0_val)
            
            # Identify saturation region
            # Consider data within +/- saturation_threshold*100% of Fmax as saturated
            saturation_lower = Fmax_lower / (Fmax - F0_val + 1e-12)  # Normalized saturation lower bound
            g['is_saturated'] = g['X'] >= saturation_threshold
            
            # Alternative: use absolute fluorescence values
            # g['is_saturated'] = (g[fluor_col] >= Fmax_lower) & (g[fluor_col] <= Fmax_upper)
            
            # Clip values between 0 and 1
            g['X'] = g['X'].clip(0, 1)
            
            # Store normalization parameters
            g['F0'] = F0_val
            g['Fmax'] = Fmax
            g['Fmax_std'] = Fmax_std
            
            return g
        
        # Support both E_nM and enzyme_ugml column names
        enzyme_col = 'enzyme_ugml' if 'enzyme_ugml' in df.columns else 'E_nM'
        df_normalized = df.groupby([enzyme_col], group_keys=False).apply(normalize_group)
        return df_normalized
    
    def estimate_initial_params(self, df, enzyme_mw=56.6):
        """Estimate initial parameters from individual curves
        Use only exponential growth phase, exclude saturation region
        """
        kobs_list, E_list = [], []
        
        # Get enzyme column name
        enzyme_col = 'enzyme_ugml' if 'enzyme_ugml' in df.columns else 'E_nM'
        ugml_col = enzyme_col == 'enzyme_ugml'
        
        for enm, g in df.groupby(enzyme_col):
            # Exclude saturation region (is_saturated == True)
            # and early phase (X < 0.1)
            g2 = g[~g.get('is_saturated', False) & (g['X'] > 0.1) & (g['X'] < 0.85)].copy()
            
            # If is_saturated column doesn't exist, use simple threshold
            if 'is_saturated' not in g.columns:
                g2 = g[(g['X'] > 0.1) & (g['X'] < 0.85)].copy()
            
            if len(g2) > 3:
                y2 = np.log(1 - g2['X'].values + 1e-12)
                slope, _ = np.polyfit(g2['time_s'].values, y2, 1)
                kobs_list.append(-slope)
                # Convert to M: ug/ml -> M
                if ugml_col:
                    # Convert ug/ml to M using user-specified molecular weight
                    MW = enzyme_mw * 1000  # kDa to g/mol
                    E_M = (enm / MW) * 1e-6  # ug/ml to M
                else:
                    E_M = enm * 1e-9  # nM to M
                E_list.append(E_M)
        
        if len(kobs_list) >= 2:
            # Linear regression to get initial estimates
            A = np.vstack([E_list, np.ones(len(E_list))]).T
            keff0, k00 = np.linalg.lstsq(A, kobs_list, rcond=None)[0]
        else:
            keff0, k00 = 1e5, 1e-6
        
        return max(keff0, 1e2), max(k00, 0.0)
    
    def fit_global_model(self, df, enzyme_mw=56.6, use_exponential_only=True):
        """Global fit to all concentration data
        Optionally exclude saturation region for better fitting
        """
        # Get enzyme column name and convert to M
        enzyme_col = 'enzyme_ugml' if 'enzyme_ugml' in df.columns else 'E_nM'
        ugml_col = enzyme_col == 'enzyme_ugml'
        
        if ugml_col:
            # Convert ug/ml to M using user-specified molecular weight
            MW = enzyme_mw * 1000  # kDa to g/mol
            df['E_M'] = (df['enzyme_ugml'] / MW) * 1e-6  # ug/ml to M
        else:
            df['E_M'] = df['E_nM'] * 1e-9  # nM to M
        
        # Optionally filter out saturation region
        if use_exponential_only and 'is_saturated' in df.columns:
            df_fit = df[~df['is_saturated'] & (df['X'] > 0.1) & (df['X'] < 0.85)].copy()
        else:
            df_fit = df[(df['X'] > 0.05) & (df['X'] < 0.90)].copy()
        
        if len(df_fit) < 5:
            # Fall back to all data if filtered data is too small
            df_fit = df.copy()
        
        t = df_fit['time_s'].values
        E = df_fit['E_M'].values
        y = df_fit['X'].values
        
        # Estimate initial parameters
        keff0, k00 = self.estimate_initial_params(df, enzyme_mw)
        p0 = [keff0, k00]
        
        # Global fitting
        try:
            popt, pcov = curve_fit(
                lambda _t, k_eff, k0: self.fraction_model(_t, E, k_eff, k0),
                t, y, p0=p0, bounds=([0, 0], [np.inf, np.inf])
            )
            
            self.k_eff, self.k0 = popt
            perr = np.sqrt(np.diag(pcov))
            
            return {
                'k_eff': self.k_eff,
                'k0': self.k0,
                'k_eff_std': perr[0],
                'k0_std': perr[1],
                'keff_low': self.k_eff - perr[0],
                'keff_high': self.k_eff + perr[0],
                'k0_low': self.k0 - perr[1],
                'k0_high': self.k0 + perr[1]
            }
        except Exception as e:
            st.error(f"Global fitting error: {str(e)}")
            return None
    
    def get_kobs_values(self, df, enzyme_mw=56.6):
        """Calculate k_obs for each enzyme concentration
        Use only exponential growth phase, exclude saturation region
        """
        # Get enzyme column name from df
        enzyme_col = None
        if 'enzyme_ugml' in df.columns:
            enzyme_col = 'enzyme_ugml'
        elif 'E_nM' in df.columns:
            enzyme_col = 'E_nM'
        else:
            raise ValueError("enzyme_ugml or E_nM column not found in dataframe")
        
        ugml_col = enzyme_col == 'enzyme_ugml'
        
        kobs_df = []
        for enm, g in df.groupby(enzyme_col):
            # Exclude saturation region and use exponential phase only
            g2 = g[~g.get('is_saturated', False) & (g['X'] > 0.1) & (g['X'] < 0.85)]
            
            # If is_saturated column doesn't exist, use simple threshold
            if 'is_saturated' not in g.columns:
                g2 = g[(g['X'] > 0.1) & (g['X'] < 0.85)]
            
            if len(g2) > 3:
                y2 = np.log(1 - g2['X'].values + 1e-12)
                slope, intercept = np.polyfit(g2['time_s'].values, y2, 1)
                kobs = -slope
                
                # Convert to molar units
                if ugml_col:
                    MW = enzyme_mw * 1000  # kDa to g/mol
                    E_M = (enm / MW) * 1e-6
                else:
                    E_M = enm * 1e-9
                
                kobs_df.append({
                    enzyme_col: enm,
                    'E_M': E_M,
                    'kobs_s-1': kobs
                })
        
        result_df = pd.DataFrame(kobs_df)
        if len(result_df) > 0 and enzyme_col in result_df.columns:
            return result_df.sort_values(enzyme_col)
        return result_df

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
        
        # 표준편차 컬럼 검사 (선택적)
        has_std = 'fluorescence_intensity_std' in data.columns
        if has_std:
            st.info("표준편차 데이터가 감지되었습니다. 에러바가 표시됩니다.")
            # 표준편차 값 검사
            if (data['fluorescence_intensity_std'] < 0).any():
                st.warning("음수 표준편차 값이 감지되었습니다.")
        
        # 시간 데이터 검사
        if (data['time'] < 0).any():
            st.warning("음수 시간 값이 감지되었습니다.")
        
        # 형광 데이터 검사
        if (data['fluorescence_intensity'] < 0).any():
            st.warning("음수 형광 강도 값이 감지되었습니다.")
        
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
            title='프로테아제 키네틱 분석 - 미카엘리스-멘텐 플롯',
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
            title='프로테아제 반응 시간 경과',
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
            title='🔬 FRET 기반 프로테아제 활성 분석',
            xaxis_title='시간 (분)',
            yaxis_title='형광 강도 (RFU)',
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
            title='🔬 표면 반응 속도 시간 경과',
            xaxis_title='시간 (분)',
            yaxis_title='반응 속도 (d[P]/dt)',
            template='plotly_white',
            showlegend=True
        )
        
        return fig

    @staticmethod
    def plot_hydrogel_fret_raw(df):
        """Plot raw fluorescence data for each enzyme concentration"""
        fig = go.Figure()
        
        # Get enzyme column name
        enzyme_col = 'enzyme_ugml' if 'enzyme_ugml' in df.columns else 'E_nM'
        unit = 'μg/mL' if enzyme_col == 'enzyme_ugml' else 'nM'
        fluor_col = 'FL_intensity' if 'FL_intensity' in df.columns else 'fluor'
        has_sd = 'SD' in df.columns
        
        for enm, g in df.groupby(enzyme_col):
            if has_sd:
                # Plot with error bars
                fig.add_trace(go.Scatter(
                    x=g['time_s'],
                    y=g[fluor_col],
                    error_y=dict(
                        type='data',
                        array=g['SD'],
                        visible=True
                    ),
                    mode='markers+lines',
                    name=f'[E] = {enm} {unit}',
                    marker=dict(size=6)
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=g['time_s'],
                    y=g[fluor_col],
                    mode='markers+lines',
                    name=f'[E] = {enm} {unit}',
                    marker=dict(size=6)
                ))
        
        fig.update_layout(
            title='🔬 Hydrogel FRET - Raw Fluorescence Data',
            xaxis_title='Time (s)',
            yaxis_title='Fluorescence Intensity (RFU)',
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def plot_hydrogel_fret_fit(df, fit_params=None, enzyme_mw=56.6):
        """Hydrogel peptide FRET fitting visualization"""
        fig = go.Figure()
        
        # Get enzyme column name and unit
        enzyme_col = 'enzyme_ugml' if 'enzyme_ugml' in df.columns else 'E_nM'
        unit = 'μg/mL' if enzyme_col == 'enzyme_ugml' else 'nM'
        ugml_col = enzyme_col == 'enzyme_ugml'
        
        # Get E_M column if available
        has_E_M = 'E_M' in df.columns
        
        # Get unique enzyme concentrations for color indexing
        unique_enzymes = sorted(df[enzyme_col].unique())
        
        # Plotly default colors
        plotly_colors = px.colors.qualitative.Set1
        
        # Plot experimental data for each enzyme concentration
        for idx, (enm, g) in enumerate(df.groupby(enzyme_col)):
            # Get color index
            color_idx = unique_enzymes.index(enm) if enm in unique_enzymes else idx
            trace_color = plotly_colors[color_idx % len(plotly_colors)]
            
            # Plot experimental data
            fig.add_trace(go.Scatter(
                x=g['time_s'],
                y=g['X'],
                mode='markers',
                name=f'[E] = {enm} {unit} (실험)',
                marker=dict(size=6, color=trace_color)
            ))
            
            # Plot saturation line if available
            if 'is_saturated' in g.columns and g['is_saturated'].any():
                # Get saturation threshold
                saturation_X = g[g['is_saturated']]['X'].mean() if g['is_saturated'].any() else g['X'].max()
                t_max = g['time_s'].max()
                t_min = g['time_s'].min()
                
                fig.add_trace(go.Scatter(
                    x=[t_min, t_max],
                    y=[saturation_X, saturation_X],
                    mode='lines',
                    name=f'Saturation: {enm} {unit}',
                    line=dict(width=1.5, dash='dot', color=trace_color),
                    showlegend=False,
                    hovertemplate=f'Saturation: X={saturation_X:.3f}<extra></extra>'
                ))
            
            # Plot fitted curve
            if fit_params:
                tfit = np.linspace(0, g['time_s'].max(), 200)
                
                # Convert to molar concentration
                if ugml_col:
                    MW = enzyme_mw * 1000  # kDa to g/mol
                    E_M = (enm / MW) * 1e-6
                else:
                    E_M = enm * 1e-9
                
                X_fit = 1 - np.exp(-(fit_params['k_eff'] * E_M + fit_params['k0']) * tfit)
                fig.add_trace(go.Scatter(
                    x=tfit,
                    y=X_fit,
                    mode='lines',
                    name=f'피팅: {enm} {unit}',
                    line=dict(width=2, dash='dash')
                ))
        
        title_text = '🔬 Hydrogel Peptide FRET Kinetics'
        if fit_params:
            title_text += f" (k_eff = {fit_params['k_eff']:.2e} M⁻¹s⁻¹)"
        
        fig.update_layout(
            title=title_text,
            xaxis_title='Time (s)',
            yaxis_title='Fraction Cleaved X(t)',
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def plot_kobs_linearity(kobs_df, fit_params=None, enzyme_mw=56.6):
        """Plot k_obs vs [E] for linearity check"""
        fig = go.Figure()
        
        if kobs_df.empty:
            return fig
        
        # Get the concentration column (enzyme_ugml or E_nM)
        conc_col = None
        if 'enzyme_ugml' in kobs_df.columns:
            conc_col = 'enzyme_ugml'
        elif 'E_nM' in kobs_df.columns:
            conc_col = 'E_nM'
        
        if conc_col is None:
            return fig
        
        unit = 'μg/mL' if conc_col == 'enzyme_ugml' else 'nM'
        
        # Plot k_obs values
        fig.add_trace(go.Scatter(
            x=kobs_df[conc_col],
            y=kobs_df['kobs_s-1'],
            mode='markers+lines',
            name='k_obs',
            marker=dict(size=10, color='blue')
        ))
        
        # Plot linear fit if parameters are available
        if fit_params and len(kobs_df) > 0:
            E_fit = np.linspace(kobs_df[conc_col].min(), kobs_df[conc_col].max(), 100)
            
            # Convert to molar for calculation
            if conc_col == 'enzyme_ugml':
                MW = enzyme_mw * 1000  # kDa to g/mol
                E_M_fit = (E_fit / MW) * 1e-6
            else:
                E_M_fit = E_fit * 1e-9
            
            kobs_fit = fit_params['k_eff'] * E_M_fit + fit_params['k0']
            fig.add_trace(go.Scatter(
                x=E_fit,
                y=kobs_fit,
                mode='lines',
                name=f'선형 피팅 (k_eff = {fit_params["k_eff"]:.2e})',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title='Linearity Check: k_obs vs [E]',
            xaxis_title=f'[E] ({unit})',
            yaxis_title='k_obs (s⁻¹)',
            template='plotly_white'
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

def create_hydrogel_fret_timeseries_data():
    """Hydrogel peptide FRET timeseries 샘플 데이터 생성"""
    # 다양한 효소 농도
    E_nM_list = [10, 25, 50, 100, 200]  # nM
    time_range = 3600  # 3600초 (60분)
    
    # True kinetic parameters
    k_eff_true = 5e4  # M^-1 s^-1
    k0_true = 1e-6   # s^-1
    
    all_data = []
    
    for E_nM in E_nM_list:
        E_M = E_nM * 1e-9
        
        # Time points for this concentration
        time_points = np.linspace(0, time_range, 60)
        
        # True fraction: X(t) = 1 - exp(-(k_eff*E + k0)*t)
        X_true = 1 - np.exp(-(k_eff_true * E_M + k0_true) * time_points)
        
        # Convert to fluorescence (FRET quenching -> dequenching)
        F0 = 50.0  # Initial fluorescence
        Fmax = 1000.0  # Maximum fluorescence
        F_true = F0 + X_true * (Fmax - F0)
        
        # Add experimental noise
        noise = np.random.normal(0, F_true * 0.02 + 2, len(F_true))  # 2% relative + additive
        F_exp = F_true + noise
        F_exp = np.maximum(F_exp, F0)  # No negative values
        
        for i, t in enumerate(time_points):
            all_data.append({
                'time_s': t,
                'E_nM': E_nM,
                'fluor': F_exp[i],
                'X_true': X_true[i]
            })
    
    return pd.DataFrame(all_data)

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
        page_title="프로테아제 키네틱 및 FRET 소광 해제 시뮬레이션",
        page_icon="🧬",
        layout="wide"
    )
    
    st.title("🧬 프로테아제 키네틱 및 FRET 소광 해제 시뮬레이션")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.title("설정")
    
    # Analysis mode selection
    analysis_mode = st.sidebar.selectbox(
        "분석 모드 선택",
        ["Hydrogel FRET Timeseries 분석", "일반 키네틱 분석", "FRET 형광 분석", "표면 펩타이드 절단 분석 (MMP9)"]
    )
    
    # Data input method selection
    if analysis_mode == "일반 키네틱 분석":
        data_source = st.sidebar.selectbox(
            "데이터 소스 선택",
            ["샘플 데이터 사용", "CSV 파일 업로드"]
        )
    elif analysis_mode == "FRET 형광 분석":
        data_source = st.sidebar.selectbox(
            "데이터 소스 선택",
            ["FRET 샘플 데이터 사용", "FRET CSV 파일 업로드"]
        )
    elif analysis_mode == "Hydrogel FRET Timeseries 분석":
        data_source = st.sidebar.selectbox(
            "데이터 소스 선택",
            ["Hydrogel FRET 샘플 데이터 사용", "CSV 파일 업로드 (time_s, enzyme_ugml, fluor)"]
        )
    else:  # Surface Peptide Cleavage Analysis
        data_source = st.sidebar.selectbox(
            "데이터 소스 선택",
            ["표면 키네틱 샘플 데이터", "표면 시간 경과 데이터", "표면 키네틱 CSV 업로드"]
        )
    
    kinetics_model = KgpProteaseKinetics()
    surface_kinetics = SurfacePeptideKinetics()
    visualizer = Visualizer()
    data_processor = DataProcessor()
    
    # Initialize fit_params to avoid UnboundLocalError
    fit_params = None
    
    # Data loading
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
            
    elif analysis_mode == "FRET 형광 분석":  # FRET 분석
        if data_source == "FRET CSV 파일 업로드":
            uploaded_file = st.sidebar.file_uploader(
                "FRET CSV 파일 선택",
                type=['csv'],
                help="필수 컬럼: time, fluorescence_intensity\n선택 컬럼: fluorescence_intensity_std (표준편차, 에러바 표시)"
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
    
    elif analysis_mode == "Hydrogel FRET Timeseries 분석":
        hydrogel_kinetics = HydrogelPeptideKinetics()
        
        # User input for molecular parameters
        st.sidebar.subheader("분자 파라미터 설정")
        enzyme_mw = st.sidebar.number_input(
            "효소 분자량 (kDa)",
            min_value=1.0,
            max_value=1000.0,
            value=56.6,
            step=0.1,
            help="μg/mL에서 M로 변환할 때 사용합니다. (KGP: 56.6 kDa)"
        )
        
        peptide_conc = st.sidebar.number_input(
            "펩타이드 농도 (mM)",
            min_value=0.1,
            max_value=100.0,
            value=2.0,
            step=0.1,
            help="고정된 펩타이드 농도"
        )
        
        # Store in session state for use in fitting
        st.session_state['enzyme_mw'] = enzyme_mw
        st.session_state['peptide_conc'] = peptide_conc
        
        if data_source == "CSV 파일 업로드 (time_s, enzyme_ugml, fluor)":
            uploaded_file = st.sidebar.file_uploader(
                "CSV 파일 선택",
                type=['csv'],
                help="필수 컬럼: time_s, enzyme_ugml (또는 E_nM), FL_intensity (또는 fluor), SD (선택)"
            )
            
            if uploaded_file is not None:
                hydrogel_data = pd.read_csv(uploaded_file)
                
                # Validate required columns (enzyme_ugml or E_nM)
                enzyme_col = None
                if 'enzyme_ugml' in hydrogel_data.columns:
                    enzyme_col = 'enzyme_ugml'
                elif 'E_nM' in hydrogel_data.columns:
                    enzyme_col = 'E_nM'
                else:
                    st.error("필수 컬럼이 누락되었습니다: enzyme_ugml (또는 E_nM)를 포함해야 합니다.")
                    st.stop()
                
                # Validate fluor column (FL_intensity or fluor)
                fluor_col = None
                if 'FL_intensity' in hydrogel_data.columns:
                    fluor_col = 'FL_intensity'
                elif 'fluor' in hydrogel_data.columns:
                    fluor_col = 'fluor'
                else:
                    st.error("필수 컬럼이 누락되었습니다: FL_intensity (또는 fluor)를 포함해야 합니다.")
                    st.stop()
                
                if 'time_s' not in hydrogel_data.columns:
                    st.error("필수 컬럼이 누락되었습니다: time_s")
                    st.stop()
                    
                st.success("Hydrogel FRET 데이터가 성공적으로 로드되었습니다!")
            else:
                st.info("CSV 파일을 업로드해주세요.")
                st.stop()
        else:
            # Use sample CSV file
            try:
                hydrogel_data = pd.read_csv("sample_hydrogel_fret_timeseries.csv")
                st.info("Hydrogel FRET 샘플 데이터를 사용합니다. (펩타이드 농도: 2 mM 고정)")
            except FileNotFoundError:
                # Fallback to generated data if file doesn't exist
                hydrogel_data = create_hydrogel_fret_timeseries_data()
                st.info("Hydrogel FRET 생성 샘플 데이터를 사용합니다.")
        
        # Normalize data
        hydrogel_data = hydrogel_kinetics.normalize_data(hydrogel_data)
    
    elif analysis_mode == "표면 펩타이드 절단 분석 (MMP9)":  # 표면 펩타이드 절단 분석
        if data_source == "표면 키네틱 CSV 업로드":
            uploaded_file = st.sidebar.file_uploader(
                "표면 키네틱 CSV 파일 선택",
                type=['csv'],
                help="필수 컬럼: enzyme_concentration, keff\n선택 컬럼: experiment_id"
            )
            
            if uploaded_file is not None:
                surface_data = pd.read_csv(uploaded_file)
                
                required_columns = ['enzyme_concentration', 'keff']
                if not all(col in surface_data.columns for col in required_columns):
                    missing_cols = [col for col in required_columns if col not in surface_data.columns]
                    st.error(f"필수 컬럼이 누락되었습니다: {missing_cols}")
                    st.stop()
                else:
                    st.success("표면 키네틱 데이터가 성공적으로 로드되었습니다!")
            else:
                st.info("표면 키네틱 CSV 파일을 업로드해주세요.")
                st.stop()
        elif data_source == "표면 시간 경과 데이터":
            surface_data = create_surface_time_course_data()
            st.info("표면 키네틱 시간 경과 샘플 데이터를 사용합니다.")
        else:
            surface_data = create_surface_kinetics_sample_data()
            st.info("표면 키네틱 샘플 데이터를 사용합니다.")
    
    # 데이터 미리보기 및 분석 (모드에 따라 분기)
    if analysis_mode == "일반 키네틱 분석":
        st.subheader("📊 데이터 미리보기")
        st.dataframe(data.head(10))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("데이터 포인트 수", len(data))
        
        with col2:
            st.metric("기질 농도 범위", f"{data['substrate_concentration'].min():.1f} - {data['substrate_concentration'].max():.1f} μM")
            
    elif analysis_mode == "FRET 형광 분석":  # FRET 분석
        st.subheader("🔬 FRET 데이터 미리보기")
        st.dataframe(fret_data.head(10))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("데이터 포인트 수", len(fret_data))
        
        with col2:
            st.metric("시간 범위", f"{fret_data['time'].min():.1f} - {fret_data['time'].max():.1f} 분")
        
        with col3:
            st.metric("형광 강도 범위", f"{fret_data['fluorescence_intensity'].min():.0f} - {fret_data['fluorescence_intensity'].max():.0f} RFU")
    
    elif analysis_mode == "Hydrogel FRET Timeseries 분석":
        st.subheader("🔬 Hydrogel FRET 데이터 미리보기")
        st.dataframe(hydrogel_data.head(10))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("데이터 포인트 수", len(hydrogel_data))
        
        with col2:
            # Get enzyme column name
            enzyme_col = 'enzyme_ugml' if 'enzyme_ugml' in hydrogel_data.columns else 'E_nM'
            unique_enzymes = hydrogel_data[enzyme_col].nunique()
            st.metric("효소 농도 조건", f"{unique_enzymes}개")
        
        with col3:
            st.metric("시간 범위", f"{hydrogel_data['time_s'].min():.0f} - {hydrogel_data['time_s'].max():.0f} 초")
    
    elif analysis_mode == "표면 펩타이드 절단 분석 (MMP9)":  # 표면 펩타이드 절단 분석
        st.subheader("🔬 표면 키네틱 데이터 미리보기")
        st.dataframe(surface_data.head(10))
        
        if data_source == "표면 시간 경과 데이터":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("데이터 포인트 수", len(surface_data))
            
            with col2:
                st.metric("시간 범위", f"{surface_data['time'].min():.1f} - {surface_data['time'].max():.1f} 분")
            
            with col3:
                unique_enzymes = surface_data['enzyme_concentration'].nunique()
                st.metric("효소 농도 조건", f"{unique_enzymes}개")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("데이터 포인트 수", len(surface_data))
            
            with col2:
                st.metric("효소 농도 범위", f"{surface_data['enzyme_concentration'].min():.1f} - {surface_data['enzyme_concentration'].max():.1f} nM")
    
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
            st.subheader("📈 Michaelis-Menten Plot")
            mm_plot = visualizer.plot_michaelis_menten(substrate_conc, reaction_rates, kinetics_model, fit_params)
            st.plotly_chart(mm_plot, use_container_width=True)
            
            # Lineweaver-Burk 플롯
            st.subheader("📉 Lineweaver-Burk Plot")
            lb_plot = visualizer.plot_lineweaver_burk(substrate_conc, reaction_rates)
            st.plotly_chart(lb_plot, use_container_width=True)
        
    elif analysis_mode == "FRET 형광 분석":  # FRET 분석
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
        
        # FRET 데이터로부터 Kinetic Parameters 피팅
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
                    st.success("FRET 파라미터 피팅이 완료되었습니다!")
                    
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
                            "최대 형광 강도",
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
    
    elif analysis_mode == "Hydrogel FRET Timeseries 분석":
        # Raw fluorescence data plot
        st.subheader("📈 Raw Fluorescence Data")
        raw_plot = visualizer.plot_hydrogel_fret_raw(hydrogel_data)
        st.plotly_chart(raw_plot, use_container_width=True)
        
        # Hydrogel FRET 글로벌 키네틱 분석
        st.subheader("🔬 Hydrogel FRET 키네틱 파라미터 분석")
        
        # Normalized data info
        st.info("""
        **데이터 정규화 정보:**
        - 각 효소 농도별로 X(t) = (F(t) - F0)/(Fmax - F0)로 정규화됩니다.
        - F0: 초기 형광 강도 (FRET quenching 상태)
        - Fmax: 최대 형광 강도 (완전 절단 상태)
        """)
        
        if st.button("글로벌 피팅 실행", key="hydrogel_fit"):
            with st.spinner("글로벌 키네틱 파라미터를 분석 중입니다..."):
                # Get enzyme MW from session state (default: 56.6 kDa for KGP)
                enzyme_mw = st.session_state.get('enzyme_mw', 56.6)
                
                # Global fit to all data
                global_fit_params = hydrogel_kinetics.fit_global_model(hydrogel_data, enzyme_mw)
                
                if global_fit_params:
                    st.success("글로벌 피팅이 완료되었습니다!")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "k_eff (M⁻¹s⁻¹)",
                            f"{global_fit_params['k_eff']:.2e}",
                            delta=f"±{global_fit_params['k_eff_std']:.2e}"
                        )
                        st.caption("k_eff ≈ kcat/KM (효율성 상수)")
                    
                    with col2:
                        st.metric(
                            "k0 (s⁻¹)",
                            f"{global_fit_params['k0']:.2e}",
                            delta=f"±{global_fit_params['k0_std']:.2e}"
                        )
                        st.caption("배경 반응 속도")
                    
                    # Display 95% confidence intervals
                    st.info(f"""
                    **95% 신뢰구간:**
                    - k_eff: [{global_fit_params['keff_low']:.2e}, {global_fit_params['keff_high']:.2e}] M⁻¹s⁻¹
                    - k0: [{global_fit_params['k0_low']:.2e}, {global_fit_params['k0_high']:.2e}] s⁻¹
                    """)
                    
                    # Show normalized data plot
                    st.subheader("📊 정규화된 데이터 (Fraction Cleaved X(t))")
                    normalized_plot = visualizer.plot_hydrogel_fret_fit(hydrogel_data, enzyme_mw=enzyme_mw)
                    st.plotly_chart(normalized_plot, use_container_width=True)
                    
                    # Plot fit results
                    st.subheader("📈 글로벌 피팅 결과")
                    fit_plot = visualizer.plot_hydrogel_fret_fit(hydrogel_data, global_fit_params, enzyme_mw)
                    st.plotly_chart(fit_plot, use_container_width=True)
                    
                    # Get k_obs values for each concentration
                    kobs_df = hydrogel_kinetics.get_kobs_values(hydrogel_data, enzyme_mw)
                    
                    # Plot linearity check
                    st.subheader("📈 선형성 확인: k_obs vs [E]")
                    linearity_plot = visualizer.plot_kobs_linearity(kobs_df, global_fit_params, enzyme_mw)
                    st.plotly_chart(linearity_plot, use_container_width=True)
                    
                    # Display k_obs table
                    st.subheader("📊 농도별 k_obs 값")
                    st.dataframe(kobs_df, use_container_width=True)
                    
                    # Calculate R² for linear fit
                    if len(kobs_df) >= 2 and 'E_M' in kobs_df.columns:
                        from scipy.stats import linregress
                        result = linregress(kobs_df['E_M'], kobs_df['kobs_s-1'])
                        r_squared = result.rvalue ** 2
                        
                        st.metric(
                            "선형 피팅 R²",
                            f"{r_squared:.4f}",
                            help="k_obs와 효소 농도의 선형 관계 품질"
                        )
                    
                    # Download results
                    st.subheader("💾 결과 다운로드")
                    results_df = pd.DataFrame({
                        'Parameter': ['k_eff (M^-1 s^-1)', 'k0 (s^-1)', 'k_eff_std', 'k0_std'],
                        'Value': [global_fit_params['k_eff'], global_fit_params['k0'],
                                 global_fit_params['k_eff_std'], global_fit_params['k0_std']]
                    })
                    
                    csv_hydrogel = results_df.to_csv(index=False)
                    st.download_button(
                        label="Hydrogel FRET 파라미터 다운로드 (CSV)",
                        data=csv_hydrogel,
                        file_name="hydrogel_fret_parameters.csv",
                        mime="text/csv",
                        key="download_hydrogel"
                    )
                    
                    fit_params = global_fit_params
                else:
                    st.error("글로벌 피팅에 실패했습니다.")
                    fit_params = None
        
        fit_params = None  # Initialize for other sections
    
    elif analysis_mode == "표면 펩타이드 절단 분석 (MMP9)":
        if data_source == "표면 시간 경과 데이터":
            # 시간 경과 데이터 분석
            st.subheader("🔬 표면 펩타이드 절단 시간 경과 분석")
            
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
            st.subheader("📊 효소 농도별 분석 결과")
            
            results_data = []
            for enzyme_conc, group in enzyme_groups:
                final_suppression = group['signal_suppression'].iloc[-1]
                max_suppression = group['signal_suppression'].max()
                keff_value = group['keff'].iloc[0]
                
                results_data.append({
                    '효소 농도 (nM)': enzyme_conc,
                    '최종 신호 억제': f"{final_suppression:.3f}",
                    '최대 신호 억제': f"{max_suppression:.3f}",
                    'keff (min⁻¹)': f"{keff_value:.4f}"
                })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df)
            
        else:
            # 키네틱 파라미터 분석
            st.subheader("🔬 표면 키네틱 파라미터 분석")
            
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
                st.subheader("📈 표면 키네틱 분석 - 1/keff vs 1/[E] 플롯")
                lineweaver_plot = visualizer.plot_surface_kinetics_lineweaver(
                    enzyme_conc, keff_values, fit_params
                )
                st.plotly_chart(lineweaver_plot, use_container_width=True)
                
                # 분석 결과 해석
                st.subheader("📋 분석 결과 해석")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**키네틱 파라미터**")
                    st.write(f"• **kcat**: {fit_params['kcat']:.4f} min⁻¹")
                    st.write(f"• **Km**: {fit_params['km']:.2f} nM")
                    st.write(f"• **kcat/Km**: {fit_params['kcat_km_ratio']:.2e} M⁻¹s⁻¹")
                
                with col2:
                    st.write("**표면 반응 특성**")
                    st.write(f"• **R² 값**: {fit_params['r_squared']:.4f}")
                    quality = "우수" if fit_params['r_squared'] > 0.95 else "양호" if fit_params['r_squared'] > 0.9 else "보통"
                    st.write(f"• **피팅 품질**: {quality}")
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
                st.error("표면 키네틱 파라미터 피팅에 실패했습니다.")
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
                st.success(f"시뮬레이션이 완료되었습니다! 최종 전환율: {final_conversion:.1f}%")
                
            elif analysis_mode == "FRET 형광 분석":  # FRET 분석
                substrate_time, product_time, fluorescence_time = kinetics_model.simulate_fret_time_course(
                    initial_substrate, time_points, fit_params['vmax'], fit_params['km']
                )
                
                fret_sim_plot = visualizer.plot_fret_fluorescence(time_points, fluorescence_time)
                st.plotly_chart(fret_sim_plot, use_container_width=True)
                
                # 결과 요약
                final_conversion = (initial_substrate - substrate_time[-1]) / initial_substrate * 100
                final_fluorescence = fluorescence_time[-1]
                st.success(f"FRET 시뮬레이션이 완료되었습니다! 최종 전환율: {final_conversion:.1f}%, 최종 형광: {final_fluorescence:.0f} RFU")
            
            elif analysis_mode == "표면 펩타이드 절단 분석 (MMP9)":
                # 표면 키네틱 시뮬레이션
                col1, col2 = st.columns(2)
                
                with col1:
                    enzyme_conc_sim = st.number_input(
                        "효소 농도 (nM)",
                        min_value=1.0,
                        max_value=1000.0,
                        value=50.0,
                        step=5.0,
                        key="surface_enzyme"
                    )
                
                with col2:
                    simulation_time = st.number_input(
                        "시뮬레이션 시간 (분)",
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
                st.subheader("📈 표면 반응 속도 분석")
                rate_plot = visualizer.plot_surface_reaction_rate(
                    time_points, [enzyme_conc_sim], surface_kinetics, 
                    fit_params['kcat'], fit_params['km']
                )
                st.plotly_chart(rate_plot, use_container_width=True)
                
                # 결과 요약
                final_suppression = signal_suppression[-1]
                max_suppression = np.max(signal_suppression)
                st.success(f"""
                표면 키네틱 시뮬레이션이 완료되었습니다!
                - 최종 신호 억제: {final_suppression:.3f}
                - 최대 신호 억제: {max_suppression:.3f}
                - keff: {keff:.4f} min⁻¹
                """)
                
                # 실험 설계 도움말
                st.info(f"""
                **💡 실험 설계 가이드라인**
                - 권장 측정 시간: {simulation_time/2:.0f}-{simulation_time:.0f}분
                - 예상 신호 변화: {final_suppression*100:.1f}%
                - 반응 완료도: {(final_suppression/surface_kinetics.max_cleavage_fraction)*100:.1f}%
                """)
        
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
        
        elif analysis_mode == "표면 펩타이드 절단 분석 (MMP9)":
            surface_results_df = pd.DataFrame({
                'Parameter': ['kcat (min⁻¹)', 'Km (nM)', 'kcat/Km (M⁻¹s⁻¹)', 'R²', 
                             'Peptide Coverage (mol/cm²)', 'Max Cleavage Fraction'],
                'Value': [fit_params['kcat'], fit_params['km'], fit_params['kcat_km_ratio'], 
                         fit_params['r_squared'], surface_kinetics.peptide_coverage, 
                         surface_kinetics.max_cleavage_fraction]
            })
            
            csv_surface = surface_results_df.to_csv(index=False)
            st.download_button(
                label="표면 키네틱 파라미터 다운로드 (CSV)",
                data=csv_surface,
                file_name="mmp9_surface_kinetic_parameters.csv",
                mime="text/csv",
                key="download_surface"
            )

if __name__ == "__main__":
    main()

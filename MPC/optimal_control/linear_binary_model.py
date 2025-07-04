# This file is part of rtMPCuGC.
# 
# Copyright (c) 2025, Daniel Bull
# Developed at HKA - Karlsruhe University of Applied Sciences.
# All rights reserved.
# 
# The BSD 3-Clause License
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import pyomo.environ as pyo
import numpy as np
import pandas as pd
from datetime import datetime

class Linear_Binary_Model():
    
    def __init__(self):
        self.warmstart_available = False

    def setProfiles(self,profileForecastHeat,profileForecastCool,profileForecastDry,profileForecastWeather,profileForecastPrice,profileForecastFrost):
        self.q_dem_HS_I = profileForecastHeat
        self.q_dem_CS_I = profileForecastCool
        self.q_dem_RLTS_I = profileForecastDry
        self.temp_amb_I = profileForecastWeather
        self.c_ELECTRICITY_buy_I = profileForecastPrice
        self.temp_frost_I = profileForecastFrost

    def setParams(self,timeSteps,stepSizeInSec,controlPeriod,NMcCormick):
        ## Time
        self.I = timeSteps
        self.StepSizeInSec2 = stepSizeInSec
        ## Control steps
        self.ControlPeriod3 = controlPeriod
        ## HP
        self.H = list(range(0,5))
        ## McCormick
        self.N_MC = NMcCormick
        ## General
        self.c_w = 4.18
        self.c_b = 3.56
        self.c_a = 1.01
        self.c_c = 0.879
        self.t_conection_delta = 50
        self.t_default = 20
        self.t_hour_in_sec = 3600
        self.T_upper_MC = 60
        self.T_lower_MC = -60
        self.W_upper_MC = self.T_upper_MC
        self.W_lower_MC = self.T_lower_MC
        ## Slack constants
        self.s_T_HP = 100
        self.s_T_HS = 100
        self.s_T_IS_C = 100
        self.s_T_IS_W = 100
        self.s_T_HXH = 100
        self.s_T_HGC = 100
        self.s_T_HXC = 100
        self.s_T_HGS = 100
        self.s_T_CS = 100
        self.s_T_RLTS = 100
        self.s_T_HXA = 100
        self.s_T_GS_C = 100
        self.s_T_GS_W = 100
        ## HP
        self.a_HP_HT_0 = 221.04
        self.a_HP_HT_1 = -1.168
        self.a_HP_HT_2 =  5.392
        self.a_HP_LT_0 = 213.61
        self.a_HP_LT_1 = -1.992
        self.a_HP_LT_2 = 5.286
        self.a_HP_EL_0 = 7.505
        self.a_HP_EL_1 = 0.874
        self.a_HP_EL_2 = 0.107
        self.d_HP_power_H = [0,0.5,1,1.5,2]
        self.mdot_digit_HP_H = [0,1,1,1,1]
        self.mdot_HP_w_H = [0,12.22,12.22,24.44,24.44]
        self.mdot_HP_b_H = [0,19.72,19.72,39.44,39.44]
        self.q_HP_HT_max = 400
        self.q_HP_HT_min = 0
        self.q_HP_LT_max = 400
        self.q_HP_LT_min = 0
        self.q_HP_EL_max = 100
        self.q_HP_EL_min = 0
        self.T_HP_HT_max = 80 # 80
        self.T_HP_HT_min = 0 
        self.T_HP_LT_max = 50 
        self.T_HP_LT_min = -50 
        self.m_HP_HT_w = 200 # was 200
        self.m_HP_LT_b = 200
        self.T_HP_delta_max = 100 # was 50 
        self.T_HP_delta_min = -100 # was 50 
        self.e_HP_EL_pumps = [0, 3.5 * 1 + 1.3 * 2, 3.5 * 1 + 1.3 * 2, 3.5 * 2 + 1.3 * 4, 3.5 * 2 + 1.3 * 4] # see datasheets 
        self.alpha_HP_time = 0.004
        self.c_switch_HP_3 = 0.5
        ## HS
        self.alpha_HS_time = 0.01 
        self.T_HS_max = 40 
        self.T_HS_min = 33 
        self.m_HS_w = 6000 
        self.T_HS_delta_max = 100 
        self.T_HS_delta_min = -100 
        ## HXA
        self.V_HXA_min = 0
        self.V_HXA_max = 1
        self.T_HXAR_min_N = [0] * (self.N_MC[-1]+1)
        self.T_HXAR_max_N = [0] * (self.N_MC[-1]+1)
        for i in self.N_MC:
            self.T_HXAR_min_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * i + self.T_lower_MC
        for i in self.N_MC:
            self.T_HXAR_max_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * (i+1) + self.T_lower_MC

        self.mdot_HXA_a = 44.94 * 2 # see datasheet recooler
        self.mdot_HXA_b = 11.67
        self.e_HXA_EL_pump = 15.5 
        self.e_HXA_EL_device = 5.75 * 2 # see datasheet recooler
        self.T_HXA_max = 60 
        self.T_HXA_min = -30 
        self.m_HXA_b = 310 * 2 # see datasheet recooler 303 l
        self.T_HXA_delta_max = 100 
        self.T_HXA_delta_min = -100 
        self.alpha_factor_HXA = 0.89
        self.alpha_HXA_time = 0.15
        self.c_switch_HXA = 0.1
        ## V_HP_HXH_HS
        self.V_HP_HXH_min = 0
        self.V_HP_HXH_max = 1
        self.T_HP_HXH_min_N = [0] * (self.N_MC[-1]+1)
        self.T_HP_HXH_max_N = [0] * (self.N_MC[-1]+1)
        for i in self.N_MC:
            self.T_HP_HXH_min_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * i + self.T_lower_MC
        for i in self.N_MC:
            self.T_HP_HXH_max_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * (i+1) + self.T_lower_MC

        self.V_HP_HS_min = 0
        self.V_HP_HS_max = 1
        self.T_HP_HS_min_N = [0] * (self.N_MC[-1]+1)
        self.T_HP_HS_max_N = [0] * (self.N_MC[-1]+1)
        for i in self.N_MC:
            self.T_HP_HS_min_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * i + self.T_lower_MC
        for i in self.N_MC:
            self.T_HP_HS_max_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * (i+1) + self.T_lower_MC

        self.c_switch_HXH_HS = 0.01
        ## HXH
        self.T_HXH_max = 60
        self.T_HXH_min = 0 
        self.T_HXH_delta_max = 100 
        self.T_HXH_delta_min = -100 
        self.a_HXH_w_b = 44.12 # see datasheet m2
        self.alpha_HXH_w_b = 4 # see datasheet W/m2 K

        self.alpha_HXH_time = 0.004
        self.m_HXH_b = 100
        self.m_HXH_w = 100
        ## V_HXA_HXH_HGC
        self.V_HXA_HXH_min = 0
        self.V_HXA_HXH_max = 1
        self.T_HXA_HXH_min_N = [0] * (self.N_MC[-1]+1)
        self.T_HXA_HXH_max_N = [0] * (self.N_MC[-1]+1)
        for i in self.N_MC:
            self.T_HXA_HXH_min_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * i + self.T_lower_MC
        for i in self.N_MC:
            self.T_HXA_HXH_max_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * (i+1) + self.T_lower_MC

        self.V_HXA_HGC_min = 0
        self.V_HXA_HGC_max = 1
        self.T_HXA_HGC_min_N = [0] * (self.N_MC[-1]+1)
        self.T_HXA_HGC_max_N = [0] * (self.N_MC[-1]+1)
        for i in self.N_MC:
            self.T_HXA_HGC_min_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * i + self.T_lower_MC
        for i in self.N_MC:
            self.T_HXA_HGC_max_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * (i+1) + self.T_lower_MC

        self.c_switch_HXH_HGC = 0.01
        ## HGC
        self.alpha_HGC_time = 0.004
        self.T_HGC_max = 60 
        self.T_HGC_min = -20 
        self.m_HGC_b = 100
        self.T_HGC_delta_max = 100 
        self.T_HGC_delta_min = -100 
        ## HXC
        self.T_HXC_max = 60 
        self.T_HXC_min = -20 
        self.T_HXC_delta_max = 100 
        self.T_HXC_delta_min = -100 
        self.a_HXC_w_b = 44.12 # see datasheet m2
        self.alpha_HXC_w_b = 4 # see datasheet W/m2 K

        self.alpha_HXC_time = 0.004
        self.m_HXC_w = 100
        self.m_HXC_b = 100
        ## HGS
        self.alpha_HGS_time = 0.004
        self.T_HGS_max = 40 
        self.T_HGS_min = 0 
        self.m_HGS_w = 100
        self.T_HGS_delta_max = 100 
        self.T_HGS_delta_min = -100 
        ## IS
        self.r_IS = list(range(0,5))
        self.cr_IS = self.r_IS
        self.wr_IS = self.r_IS[0::2]
        self.T_IS_max_c = 30 
        self.T_IS_min_c = 19 
        self.T_IS_max_w = 30 
        self.T_IS_min_w = 19 
        self.T_IS_delta_max = 100 
        self.T_IS_delta_min = -100 
        self.mdot_IS_w = 36.11
        self.n_IS_blocks = 165
        self.e_IS_EL = (3 * 1.3 + 1.2) # see datasheet
        self.m_IS_c = 9052.728 / (len(self.cr_IS))  # per block
        self.m_IS_w = 18.16 / (len(self.wr_IS))  # per block
        self.height_IS = 0.09 # fix, since we don't want to change the layer depth
        self.width_IS = 0.75 # fix, since we change the length for less columns, not the width (ratio of water tube to concrete stays the same)
        self.length_IS = 11.23
        self.diameter_IS = 0.0262
        self.a_north_south_IS = self.width_IS * self.length_IS
        self.a_east_west_IS = self.height_IS * self.length_IS
        self.a_pipe_IS = self.diameter_IS * 3.14159 * self.length_IS
        self.alpha_IS_w_c = 0.7
        self.lambda_IS_c_c = 0.0025
        self.lambda_IS_c_a = 0.000413
        self.c_switch_HS_HGS = 0.01
        ## GS
        self.r_GS = list(range(0,7)) # fix, since we don't want to change the layer depth
        self.c_GS = list(range(0,1)) 
        self.cr_GS = self.r_GS
        self.cc_GS = self.c_GS
        self.wr_GS = self.r_GS[1::2]
        self.wc_GS = self.c_GS
        self.T_GS_max_c = 19.5
        self.T_GS_min_c = 3.5
        self.T_GS_max_w = 19.5
        self.T_GS_min_w = 3.5
        self.m_GS_c = 22481.304 / (len(self.r_GS) * len(self.c_GS))  # per block
        self.m_GS_w = 30.569 / (len(self.wr_GS) * len(self.wc_GS))  # per block
        self.height_GS = 0.1657 # fix, since we don't want to change the layer depth
        self.width_GS = 0.45 # fix, since we change the length for less columns, not the width (ratio of water tube to concrete stays the same)
        self.length_GS = 18/len(self.c_GS)
        self.diameter_GS = 0.0262
        self.a_north_south_GS = self.width_GS * self.length_GS
        self.a_east_west_GS = self.height_GS * self.length_GS
        self.a_pipe_GS = self.diameter_GS * 3.14159 * self.length_GS
        self.alpha_GS_w_c = 0.7
        self.lambda_GS_c_c = 0.0025
        self.lambda_GS_c_s = 0.00165
        self.lambda_GS_c_a = 0.000413
        self.n_GS_blocks = 306 
        self.mdot_GS_w = 50.00
        self.e_GS_EL = (5 * 1.3) # see datasheet
        ## VP_HS_IS
        self.V_HS_IS_min = 0
        self.V_HS_IS_max = 1
        self.T_HS_IS_min_N = [0] * (self.N_MC[-1]+1)
        self.T_HS_IS_max_N = [0] * (self.N_MC[-1]+1)
        for i in self.N_MC:
            self.T_HS_IS_min_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * i + self.T_lower_MC
        for i in self.N_MC:
            self.T_HS_IS_max_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * (i+1) + self.T_lower_MC
        ## V_HP_HGC_HGCHXC
        self.V_HP_HGC_min = 0
        self.V_HP_HGC_max = 1
        self.T_HP_HGC_min_N = [0] * (self.N_MC[-1]+1)
        self.T_HP_HGC_max_N = [0] * (self.N_MC[-1]+1)
        for i in self.N_MC:
            self.T_HP_HGC_min_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * i + self.T_lower_MC
        for i in self.N_MC:
            self.T_HP_HGC_max_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * (i+1) + self.T_lower_MC

        self.V_HP_HGCHXC_min = 0
        self.V_HP_HGCHXC_max = 1
        self.T_HP_HGCHXC_min_N = [0] * (self.N_MC[-1]+1)
        self.T_HP_HGCHXC_max_N = [0] * (self.N_MC[-1]+1)
        for i in self.N_MC:
            self.T_HP_HGCHXC_min_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * i + self.T_lower_MC
        for i in self.N_MC:
            self.T_HP_HGCHXC_max_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * (i+1) + self.T_lower_MC

        self.c_switch_HGC_HGCHXC = 0.01
        ## VP_IS_HGS
        self.V_IS_HGS_min = 0
        self.V_IS_HGS_max = 1
        self.T_IS_HGS_min_N = [0] * (self.N_MC[-1]+1)
        self.T_IS_HGS_max_N = [0] * (self.N_MC[-1]+1)
        for i in self.N_MC:
            self.T_IS_HGS_min_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * i + self.T_lower_MC
        for i in self.N_MC:
            self.T_IS_HGS_max_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * (i+1) + self.T_lower_MC
        ## V_GS_HGS_CS
        self.V_GS_HGS_min = 0
        self.V_GS_HGS_max = 1
        self.T_GS_HGS_min_N = [0] * (self.N_MC[-1]+1)
        self.T_GS_HGS_max_N = [0] * (self.N_MC[-1]+1)
        for i in self.N_MC:
            self.T_GS_HGS_min_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * i + self.T_lower_MC
        for i in self.N_MC:
            self.T_GS_HGS_max_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * (i+1) + self.T_lower_MC

        self.V_GS_CS_min = 0
        self.V_GS_CS_max = 1
        self.T_GS_CS_min_N = [0] * (self.N_MC[-1]+1)
        self.T_GS_CS_max_N = [0] * (self.N_MC[-1]+1)
        for i in self.N_MC:
            self.T_GS_CS_min_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * i + self.T_lower_MC
        for i in self.N_MC:
            self.T_GS_CS_max_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * (i+1) + self.T_lower_MC

        self.c_switch_HGS_CS = 0.01
        ## CS
        self.alpha_CS_time = 0.01
        self.T_CS_max = 18 
        self.T_CS_min = 10 
        self.m_CS_w = 6000 
        self.T_CS_delta_max = 100 
        self.T_CS_delta_min = -100 
        ## RLTS
        self.alpha_RLTS_time = 0.003
        self.T_RLTS_max = 18 
        self.T_RLTS_min = 6 
        self.m_RLTS_w = 2000 
        self.T_RLTS_delta_max = 100 
        self.T_RLTS_delta_min = -100 
        ## VP_HXC_HGS_CS_RLTS
        self.V_HXC_HGS_min = 0
        self.V_HXC_HGS_max = 1
        self.T_HXC_HGS_min_N = [0] * (self.N_MC[-1]+1)
        self.T_HXC_HGS_max_N = [0] * (self.N_MC[-1]+1)
        for i in self.N_MC:
            self.T_HXC_HGS_min_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * i + self.T_lower_MC
        for i in self.N_MC:
            self.T_HXC_HGS_max_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * (i+1) + self.T_lower_MC

        self.V_HXC_CS_min = 0
        self.V_HXC_CS_max = 1
        self.T_HXC_CS_min_N = [0] * (self.N_MC[-1]+1)
        self.T_HXC_CS_max_N = [0] * (self.N_MC[-1]+1)
        for i in self.N_MC:
            self.T_HXC_CS_min_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * i + self.T_lower_MC
        for i in self.N_MC:
            self.T_HXC_CS_max_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * (i+1) + self.T_lower_MC

        self.V_HXC_RLTS_min = 0
        self.V_HXC_RLTS_max = 1
        self.T_HXC_RLTS_min_N = [0] * (self.N_MC[-1]+1)
        self.T_HXC_RLTS_max_N = [0] * (self.N_MC[-1]+1)
        for i in self.N_MC:
            self.T_HXC_RLTS_min_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * i + self.T_lower_MC
        for i in self.N_MC:
            self.T_HXC_RLTS_max_N[i] = (self.T_upper_MC-self.T_lower_MC)/(self.N_MC[-1]+1) * (i+1) + self.T_lower_MC

        self.mdot_VP_tot = 16.00
        self.c_switch_VP_lin = 0.01
        self.e_VP_EL = 3.7

    def setVariables(self,model,binary=1):
        self.m = model
        ## General variables
        self.m.C_TOT_I_ = pyo.Var(domain=pyo.NonNegativeReals)
        self.m.C_OP_I = pyo.Var(self.I[0:-1], domain=pyo.NonNegativeReals)
        ## Slack variables
        self.m.S_TOT_I_ = pyo.Var(domain=pyo.NonNegativeReals)
        self.m.S_OP_I = pyo.Var(self.I[1:], domain=pyo.NonNegativeReals)
        ## Toggle variables (Switching variables)
        self.m.T_TOT_I_ = pyo.Var(domain=pyo.NonNegativeReals)
        self.m.T_OP_I = pyo.Var(self.I[0:-1], domain=pyo.NonNegativeReals)
        ## HP
        self.m.E_HP_EL_in_I = pyo.Var(self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.E_HP_EL_I = pyo.Var(self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.Q_HP_HT_I = pyo.Var(self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.Q_HP_LT_I = pyo.Var(self.I[0:-1], domain=pyo.NonNegativeReals)
        if binary == 0:
            self.m.B_HP_H_I = pyo.Var(self.H, self.I[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_HP_H_I = pyo.Var(self.H, self.I[0:-1], domain=pyo.Binary)
        self.m.Z_HP_I_3 = pyo.Var(self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.Z_HP_Q_HT_H_I = pyo.Var(self.H, self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.Z_HP_Q_LT_H_I = pyo.Var(self.H, self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.Z_HP_E_EL_H_I = pyo.Var(self.H, self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_HP_HT_I = pyo.Var(self.I, domain=pyo.Reals)
        self.m.T_HP_HT_in_I = pyo.Var(self.I[0:-1], domain=pyo.Reals)
        self.m.Z_HP_HT_in_H_I = pyo.Var(self.H, self.I[0:-1], domain=pyo.Reals)
        self.m.T_HP_HT_out_I = pyo.Var(self.I[0:-1], domain=pyo.Reals)
        self.m.T_HP_LT_I = pyo.Var(self.I, domain=pyo.Reals)
        self.m.T_HP_LT_in_I = pyo.Var(self.I[0:-1], domain=pyo.Reals)
        self.m.Z_HP_LT_in_H_I = pyo.Var(self.H, self.I[0:-1], domain=pyo.Reals)
        self.m.T_HP_LT_out_I = pyo.Var(self.I[0:-1], domain=pyo.Reals)
        self.m.S_T_HP_I = pyo.Var(self.I[1:], domain=pyo.NonNegativeReals)
        ## HS  
        self.m.T_HS_I = pyo.Var(self.I, domain=pyo.Reals)
        self.m.S_T_HS_I = pyo.Var(self.I, domain=pyo.NonNegativeReals)
        ## HXA
        if binary == 0:
            self.m.B_T_HXAR_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_T_HXAR_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.Binary)
        self.m.P_HXA_I = pyo.Var(self.I[0:-1], domain = pyo.NonNegativeReals)
        self.m.P_HXA_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_HXAR_I = pyo.Var(self.I, domain = pyo.Reals)
        self.m.Z_HXAR_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.T_HXAR_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.W_HXA_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)
        self.m.T_HXAR_in_I = pyo.Var(self.I[0:-1], domain=pyo.Reals)
        self.m.E_HXA_EL_I = pyo.Var(self.I[0:-1], domain=pyo.NonNegativeReals)

        self.m.T_HXA_I = pyo.Var(self.I, domain=pyo.Reals)
        self.m.S_T_HXA_I = pyo.Var(self.I, domain=pyo.NonNegativeReals)
        self.m.Z_HXA_I = pyo.Var(self.I[0:-1], domain=pyo.NonNegativeReals)
        ## V_HP_HXH_HS
        if binary == 0:
            self.m.B_T_HP_HXH_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_T_HP_HXH_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.Binary)
        self.m.V_HP_HXH_I = pyo.Var(self.I[0:-1], domain = pyo.NonNegativeReals)
        self.m.V_HP_HXH_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_HP_HXH_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)
        self.m.Z_HP_HXH_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.T_HP_HXH_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.W_HP_HXH_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)

        if binary == 0:
            self.m.B_T_HP_HS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_T_HP_HS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.Binary)
        self.m.V_HP_HS_I = pyo.Var(self.I[0:-1], domain = pyo.NonNegativeReals)
        self.m.V_HP_HS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_HP_HS_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)
        self.m.Z_HP_HS_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.T_HP_HS_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.W_HP_HS_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)

        self.m.Z_HXH_HS_I = pyo.Var(self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.Z_HP_HXH_H_I = pyo.Var(self.H,self.I[0:-1],domain=pyo.Reals)
        self.m.Z_HP_HS_H_I = pyo.Var(self.H,self.I[0:-1],domain=pyo.Reals)

        self.m.Z_HXH_HXH_w_I = pyo.Var(self.H,self.I[0:-1], domain = pyo.Reals)
        ## HXH
        self.m.T_HXH_I = pyo.Var(self.I[0:-1], domain=pyo.Reals) 
        self.m.T_HXH_w_out_I = pyo.Var(self.I[0:-1], domain=pyo.Reals)
        self.m.T_HXH_b_out_I = pyo.Var(self.I[0:-1], domain=pyo.Reals)
        self.m.Qdot_HXH_w_b_I = pyo.Var(self.I[0:-1], domain=pyo.Reals)
        self.m.S_T_HXH_I = pyo.Var(self.I[1:], domain=pyo.NonNegativeReals)

        self.m.T_HXH_b_I = pyo.Var(self.I, domain=pyo.Reals)
        self.m.S_T_HXH_b_I = pyo.Var(self.I[1:], domain=pyo.NonNegativeReals) 

        self.m.T_HXH_w_I = pyo.Var(self.I, domain=pyo.Reals)
        self.m.S_T_HXH_w_I = pyo.Var(self.I[1:], domain=pyo.NonNegativeReals)       
        ## V_HXA_HXH_HGC
        if binary ==0:
            self.m.B_T_HXA_HXH_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_T_HXA_HXH_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.Binary)
        self.m.V_HXA_HXH_I = pyo.Var(self.I[0:-1], domain = pyo.NonNegativeReals)
        self.m.V_HXA_HXH_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_HXA_HXH_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)
        self.m.Z_HXA_HXH_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.T_HXA_HXH_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.W_HXA_HXH_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)

        if binary ==0:
            self.m.B_T_HXA_HGC_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_T_HXA_HGC_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.Binary)
        self.m.V_HXA_HGC_I = pyo.Var(self.I[0:-1], domain = pyo.NonNegativeReals)
        self.m.V_HXA_HGC_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_HXA_HGC_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)
        self.m.Z_HXA_HGC_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.T_HXA_HGC_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.W_HXA_HGC_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)

        self.m.Z_HXH_HXH_b_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)
        self.m.Z_HXH_HGC_I = pyo.Var(self.I[0:-1], domain=pyo.NonNegativeReals) 
        ## HGC
        self.m.T_HGC_I = pyo.Var(self.I, domain=pyo.Reals)
        self.m.S_T_HGC_I = pyo.Var(self.I[1:], domain=pyo.NonNegativeReals)
        ## HXC
        self.m.T_HXC_I = pyo.Var(self.I[0:-1], domain=pyo.Reals)
        self.m.T_HXC_b_out_I = pyo.Var(self.I[0:-1], domain=pyo.Reals)
        self.m.T_HXC_w_out_I = pyo.Var(self.I[0:-1], domain=pyo.Reals)
        self.m.Qdot_HXC_w_b_I = pyo.Var(self.I[0:-1], domain=pyo.Reals)
        self.m.S_T_HXC_I = pyo.Var(self.I[1:], domain=pyo.NonNegativeReals)

        self.m.T_HXC_w_I = pyo.Var(self.I, domain=pyo.Reals)
        self.m.S_T_HXC_w_I = pyo.Var(self.I[1:], domain=pyo.NonNegativeReals)

        self.m.T_HXC_b_I = pyo.Var(self.I, domain=pyo.Reals)
        self.m.S_T_HXC_b_I = pyo.Var(self.I[1:], domain=pyo.NonNegativeReals)
        ## HGS
        self.m.T_HGS_I = pyo.Var(self.I, domain=pyo.Reals)
        self.m.S_T_HGS_I = pyo.Var(self.I[1:], domain=pyo.NonNegativeReals)
        ## IS
        self.m.T_IS_C_I_CR = pyo.Var(self.I,self.cr_IS, domain=pyo.Reals)
        self.m.T_IS_W_I_WR = pyo.Var(self.I,self.wr_IS, domain=pyo.Reals)
        self.m.S_T_IS_C_I_CR = pyo.Var(self.I[1:],self.cr_IS, domain=pyo.NonNegativeReals)
        self.m.S_T_IS_W_I_WR = pyo.Var(self.I[1:],self.wr_IS, domain=pyo.NonNegativeReals)
        self.m.Q_IS_C_NORTH_I_CR = pyo.Var(self.I[:-1],self.cr_IS, domain=pyo.Reals)
        self.m.Q_IS_C_SOUTH_I_CR = pyo.Var(self.I[:-1],self.cr_IS, domain=pyo.Reals)
        self.m.Q_IS_C_W_I_WR = pyo.Var(self.I[:-1],self.wr_IS, domain=pyo.Reals)
        self.m.Q_IS_W_I_WR = pyo.Var(self.I[:-1],self.wr_IS, domain=pyo.Reals)
        self.m.Q_IS_W_I_IN = pyo.Var(self.I[:-1], domain=pyo.Reals)
        self.m.Q_IS_W_C_I_WR = pyo.Var(self.I[:-1],self.wr_IS, domain=pyo.Reals)
        self.m.T_IS_I = pyo.Var(self.I, domain=pyo.Reals)
        self.m.E_IS_EL_I = pyo.Var(self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.Z_IS_pump_I = pyo.Var(self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.Z_HS_HGS_I = pyo.Var(self.I[0:-1], domain=pyo.NonNegativeReals)
        ## GS
        self.m.T_GS_C_I_CR_CC = pyo.Var(self.I,self.cc_GS,self.cr_GS, domain=pyo.Reals)
        self.m.T_GS_W_I_WR_WC = pyo.Var(self.I,self.wc_GS,self.wr_GS, domain=pyo.Reals)
        self.m.S_T_GS_C_I_CR_CC = pyo.Var(self.I,self.cc_GS,self.cr_GS, domain=pyo.NonNegativeReals)
        self.m.S_T_GS_W_I_WR_WC = pyo.Var(self.I,self.wc_GS,self.wr_GS, domain=pyo.NonNegativeReals)
        self.m.Q_GS_C_NORTH_I_CR_CC = pyo.Var(self.I,self.cc_GS,self.cr_GS, domain=pyo.Reals)
        self.m.Q_GS_C_EAST_I_CR_CC = pyo.Var(self.I,self.cc_GS,self.cr_GS, domain=pyo.Reals)
        self.m.Q_GS_C_SOUTH_I_CR_CC = pyo.Var(self.I,self.cc_GS,self.cr_GS, domain=pyo.Reals)
        self.m.Q_GS_C_WEST_I_CR_CC = pyo.Var(self.I,self.cc_GS,self.cr_GS, domain=pyo.Reals)
        self.m.Q_GS_C_W_I_WR_WC = pyo.Var(self.I,self.wc_GS,self.wr_GS, domain=pyo.Reals)
        self.m.Q_GS_W_EAST_I_WR_WC = pyo.Var(self.I,self.wc_GS,self.wr_GS, domain=pyo.Reals)
        self.m.Q_GS_W_WEST_I_WR_WC = pyo.Var(self.I,self.wc_GS,self.wr_GS, domain=pyo.Reals)
        self.m.Q_GS_W_C_I_WR_WC = pyo.Var(self.I,self.wc_GS,self.wr_GS, domain=pyo.Reals)
        self.m.T_GS_I = pyo.Var(self.I, domain=pyo.Reals)
        self.m.E_GS_EL_I = pyo.Var(self.I[0:-1], domain=pyo.NonNegativeReals)
        ## VP_HS_IS
        if binary ==0:
            self.m.B_T_HS_IS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_T_HS_IS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.Binary)
        self.m.V_HS_IS_I = pyo.Var(self.I[0:-1], domain = pyo.NonNegativeReals)
        self.m.V_HS_IS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_HS_IS_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)
        self.m.Z_HS_IS_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.T_HS_IS_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.W_HS_IS_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)

        if binary ==0:
            self.m.B_T_HS_IS_N_I_2 = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_T_HS_IS_N_I_2 = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.Binary)
        self.m.V_HS_IS_I_2 = pyo.Var(self.I[0:-1], domain = pyo.NonNegativeReals)
        self.m.V_HS_IS_N_I_2 = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_HS_IS_I_2 = pyo.Var(self.I[0:-1], domain = pyo.Reals)
        self.m.Z_HS_IS_N_I_2 = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.T_HS_IS_N_I_2 = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.W_HS_IS_I_2 = pyo.Var(self.I[0:-1], domain = pyo.Reals)
        ## V_HP_HGC_HGCHXC
        if binary==0:
            self.m.B_T_HP_HGC_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_T_HP_HGC_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.Binary)
        self.m.V_HP_HGC_I = pyo.Var(self.I[0:-1], domain = pyo.NonNegativeReals)
        self.m.V_HP_HGC_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_HP_HGC_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)
        self.m.Z_HP_HGC_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.T_HP_HGC_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.W_HP_HGC_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)

        if binary ==0:
            self.m.B_T_HP_HGCHXC_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_T_HP_HGCHXC_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.Binary)
        self.m.V_HP_HGCHXC_I = pyo.Var(self.I[0:-1], domain = pyo.NonNegativeReals)
        self.m.V_HP_HGCHXC_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_HP_HGCHXC_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)
        self.m.Z_HP_HGCHXC_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.T_HP_HGCHXC_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.W_HP_HGCHXC_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)

        self.m.Z_HGC_HGCHXC_I = pyo.Var(self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.Z_HP_HGC_H_I = pyo.Var(self.H,self.I[0:-1],domain=pyo.Reals)
        self.m.Z_HP_HGC_2_H_I = pyo.Var(self.H,self.I[0:-1],domain=pyo.Reals)
        self.m.Z_HP_HGCHXC_H_I = pyo.Var(self.H,self.I[0:-1],domain=pyo.Reals)
        self.m.Z_HP_HGCHXC_2_H_I = pyo.Var(self.H,self.I[0:-1],domain=pyo.Reals)
        
        if binary==0:
            self.m.B_T_HP_HGCHXC_N_2_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_T_HP_HGCHXC_N_2_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.Binary)
        self.m.V_HP_HGCHXC_2_I = pyo.Var(self.I[0:-1], domain = pyo.NonNegativeReals)
        self.m.V_HP_HGCHXC_N_2_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_HP_HGCHXC_2_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)
        self.m.Z_HP_HGCHXC_N_2_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.T_HP_HGCHXC_N_2_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.W_HP_HGCHXC_2_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)

        self.m.Z_HXC_HXC_b_I = pyo.Var(self.H,self.I[0:-1], domain = pyo.Reals)
        ## VP_IS_HGS
        if binary ==0:
            self.m.B_T_IS_HGS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_T_IS_HGS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.Binary)
        self.m.V_IS_HGS_I = pyo.Var(self.I[0:-1], domain = pyo.NonNegativeReals)
        self.m.V_IS_HGS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_IS_HGS_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)
        self.m.Z_IS_HGS_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.T_IS_HGS_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.W_IS_HGS_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)

        if binary ==0:
            self.m.B_T_IS_HGS_N_I_2 = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_T_IS_HGS_N_I_2 = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.Binary)
        self.m.V_IS_HGS_I_2 = pyo.Var(self.I[0:-1], domain = pyo.NonNegativeReals)
        self.m.V_IS_HGS_N_I_2 = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_IS_HGS_I_2 = pyo.Var(self.I[0:-1], domain = pyo.Reals)
        self.m.Z_IS_HGS_N_I_2 = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.T_IS_HGS_N_I_2 = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.W_IS_HGS_I_2 = pyo.Var(self.I[0:-1], domain = pyo.Reals)        
        ## V_GS_HGS_CS
        if binary ==0:
            self.m.B_T_GS_HGS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_T_GS_HGS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.Binary)
        self.m.V_GS_HGS_I = pyo.Var(self.I[0:-1], domain = pyo.NonNegativeReals)
        self.m.V_GS_HGS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_GS_HGS_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)
        self.m.Z_GS_HGS_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.T_GS_HGS_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.W_GS_HGS_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)

        if binary ==0:
            self.m.B_T_GS_CS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_T_GS_CS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.Binary)
        self.m.V_GS_CS_I = pyo.Var(self.I[0:-1], domain = pyo.NonNegativeReals)
        self.m.V_GS_CS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_GS_CS_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)
        self.m.Z_GS_CS_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.T_GS_CS_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.W_GS_CS_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)

        if binary ==0:
            self.m.B_T_GS_HGS_N_I_2 = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_T_GS_HGS_N_I_2 = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.Binary)
        self.m.V_GS_HGS_I_2 = pyo.Var(self.I[0:-1], domain = pyo.NonNegativeReals)
        self.m.V_GS_HGS_N_I_2 = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_GS_HGS_I_2 = pyo.Var(self.I[0:-1], domain = pyo.Reals)
        self.m.Z_GS_HGS_N_I_2 = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.T_GS_HGS_N_I_2 = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.W_GS_HGS_I_2 = pyo.Var(self.I[0:-1], domain = pyo.Reals)

        if binary ==0:
            self.m.B_T_GS_CS_N_I_2 = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_T_GS_CS_N_I_2 = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.Binary)
        self.m.V_GS_CS_I_2 = pyo.Var(self.I[0:-1], domain = pyo.NonNegativeReals)
        self.m.V_GS_CS_N_I_2 = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_GS_CS_I_2 = pyo.Var(self.I[0:-1], domain = pyo.Reals)
        self.m.Z_GS_CS_N_I_2 = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.T_GS_CS_N_I_2 = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.W_GS_CS_I_2 = pyo.Var(self.I[0:-1], domain = pyo.Reals)

        self.m.Z_HGS_CS_I = pyo.Var(self.I[0:-1], domain=pyo.NonNegativeReals)
        ## CS
        self.m.T_CS_I = pyo.Var(self.I, domain=pyo.Reals)
        self.m.S_T_CS_I = pyo.Var(self.I[1:], domain=pyo.NonNegativeReals)
        ## RLTS
        self.m.T_RLTS_I = pyo.Var(self.I, domain=pyo.Reals)
        self.m.S_T_RLTS_I = pyo.Var(self.I[1:], domain=pyo.NonNegativeReals)
        ## VP_HXC_HGS_CS_RLTS
        if binary ==0:
            self.m.B_T_HXC_HGS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_T_HXC_HGS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.Binary)
        self.m.V_HXC_HGS_I = pyo.Var(self.I[0:-1], domain = pyo.NonNegativeReals)
        self.m.V_HXC_HGS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_HXC_HGS_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)
        self.m.Z_HXC_HGS_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.T_HXC_HGS_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.W_HXC_HGS_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)

        if binary ==0:
            self.m.B_T_HXC_CS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_T_HXC_CS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.Binary)
        self.m.V_HXC_CS_I = pyo.Var(self.I[0:-1], domain = pyo.NonNegativeReals)
        self.m.V_HXC_CS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_HXC_CS_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)
        self.m.Z_HXC_CS_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.T_HXC_CS_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.W_HXC_CS_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)

        if binary ==0:
            self.m.B_T_HXC_RLTS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_T_HXC_RLTS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.Binary)
        self.m.V_HXC_RLTS_I = pyo.Var(self.I[0:-1], domain = pyo.NonNegativeReals)
        self.m.V_HXC_RLTS_N_I = pyo.Var(self.N_MC,self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_HXC_RLTS_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)
        self.m.Z_HXC_RLTS_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.T_HXC_RLTS_N_I = pyo.Var(self.N_MC, self.I[0:-1], domain = pyo.Reals)
        self.m.W_HXC_RLTS_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)

        self.m.Z_HXC_HXC_w_I = pyo.Var(self.I[0:-1], domain = pyo.Reals)

        self.m.Z_VP_I = pyo.Var(self.I[0:-1], domain=pyo.NonNegativeReals)
        self.m.E_VP_EL_I = pyo.Var(self.I[0:-1], domain=pyo.NonNegativeReals)

        return self.m

    def setStartValues(self,model,T_HP_HT_start,T_HP_LT_start,T_HS_start,T_HXA_start,T_HXH_start,T_HGC_start,T_HXC_start,T_HGS_start,T_IS_w_1_start,T_IS_w_2_start,T_IS_w_3_start,T_IS_c_1_start,T_IS_c_2_start,T_IS_c_3_start,T_IS_c_4_start,T_IS_c_5_start,T_GS_w_1_start,T_GS_w_2_start,T_GS_w_3_start,T_GS_c_1_start,T_GS_c_2_start,T_GS_c_3_start,T_GS_c_4_start,T_GS_c_5_start,T_GS_c_6_start,T_GS_c_7_start,T_CS_start,T_RLTS_start):
        self.m = model
        self.T_HP_HT_start = T_HP_HT_start
        self.T_HP_LT_start = T_HP_LT_start
        self.T_HS_start = T_HS_start
        self.T_HXA_start = T_HXA_start
        self.T_HXH_start = T_HXH_start
        self.T_HGC_start = T_HGC_start
        self.T_HXC_start = T_HXC_start
        self.T_HGS_start = T_HGS_start
        self.T_IS_w_1_start = T_IS_w_1_start
        self.T_IS_w_2_start = T_IS_w_2_start
        self.T_IS_w_3_start = T_IS_w_3_start
        self.T_IS_c_1_start = T_IS_c_1_start
        self.T_IS_c_2_start = T_IS_c_2_start
        self.T_IS_c_3_start = T_IS_c_3_start
        self.T_IS_c_4_start = T_IS_c_4_start
        self.T_IS_c_5_start = T_IS_c_5_start
        self.t_IS_air = 20 # fix
        self.T_GS_w_1_start = T_GS_w_1_start
        self.T_GS_w_2_start = T_GS_w_2_start
        self.T_GS_w_3_start = T_GS_w_3_start
        self.T_GS_c_1_start = T_GS_c_1_start
        self.T_GS_c_2_start = T_GS_c_2_start
        self.T_GS_c_3_start = T_GS_c_3_start
        self.T_GS_c_4_start = T_GS_c_4_start
        self.T_GS_c_5_start = T_GS_c_5_start
        self.T_GS_c_6_start = T_GS_c_6_start
        self.T_GS_c_7_start = T_GS_c_7_start
        self.t_GS_soil = 16 # Fix from investigation KIT
        self.t_GS_air = 16
        self.T_CS_start = T_CS_start
        self.T_RLTS_start = T_RLTS_start
        return self.m

    def setEndValues(self,model,End_Temp_Constraints,T_HS_end,T_CS_end,T_RLTS_end,End_Toggle_Constraints,B_HP_1_end,B_HP_2_end,B_HP_3_end,B_HP_4_end,V_HP_HXH_end,V_HP_HS_end,V_HP_HGC_end,V_HGCHXC_end,V_HXA_end,V_HXA_HXH_end,V_HS_IS_end,V_IS_HGS_end,V_HXA_HGC_end,V_GS_HGS_end,V_GS_CS_end):
        self.m = model
        self.End_Temp_Constraints = End_Temp_Constraints
        self.End_Toggle_Constraints = End_Toggle_Constraints
        
        if End_Temp_Constraints == True:
            self.T_HS_end = T_HS_end
            self.T_CS_end = T_CS_end
            self.T_RLTS_end = T_RLTS_end
        if End_Toggle_Constraints == True:
            self.B_HP_1_end = B_HP_1_end
            self.B_HP_2_end = B_HP_2_end
            self.B_HP_3_end = B_HP_3_end
            self.B_HP_4_end = B_HP_4_end
            self.V_HP_HXH_end = V_HP_HXH_end
            self.V_HP_HS_end = V_HP_HS_end
            self.V_HP_HGC_end = V_HP_HGC_end
            self.V_HGCHXC_end = V_HGCHXC_end
            self.V_HXA_end = V_HXA_end
            self.V_HXA_HXH_end = V_HXA_HXH_end
            self.V_HS_IS_end = V_HS_IS_end
            self.V_IS_HGS_end = V_IS_HGS_end
            self.V_HXA_HGC_end = V_HXA_HGC_end
            self.V_GS_HGS_end = V_GS_HGS_end
            self.V_GS_CS_end = V_GS_CS_end
            ## All but cold side HP HXC
        return self.m

    def setConstraints(self,model):
        self.m = model
        ## General cost constraint
        self.m.Constraint_Cost_I = pyo.Constraint(expr = self.m.C_TOT_I_ == sum(self.m.C_OP_I[i] for i in self.I[0:-1]))

        ## Cost constraints 
        self.m.Constraint_Cost_time_I = pyo.ConstraintList()
        for i in self.I[0:-1]:
            self.m.Constraint_Cost_time_I.add(self.m.C_OP_I[i] == self.StepSizeInSec2/self.t_hour_in_sec * ((self.m.E_HP_EL_in_I[i] + self.m.E_HXA_EL_I[i] + self.m.E_IS_EL_I[i] + self.m.E_GS_EL_I[i] + self.m.E_VP_EL_I[i]) * self.c_ELECTRICITY_buy_I[i]))

        ## General Slack constraint
        self.m.Constraint_Slack_I = pyo.Constraint(expr = self.m.S_TOT_I_ == sum(self.m.S_OP_I[i] for i in self.I[1:]))

        ## Slack constraints
        self.m.Constraint_Slack_time_I = pyo.ConstraintList()
        for i in self.I[1:]:
            self.m.Constraint_Slack_time_I.add(self.m.S_OP_I[i] == self.StepSizeInSec2/self.t_hour_in_sec * (self.s_T_HP * self.m.S_T_HP_I[i] + self.s_T_HS * self.m.S_T_HS_I[i] + sum(self.s_T_IS_W * self.m.S_T_IS_W_I_WR[i,r] for r in self.wr_IS) + sum(self.s_T_IS_C * self.m.S_T_IS_C_I_CR[i,r] for r in self.cr_IS) + self.s_T_HXH * self.m.S_T_HXH_I[i] + self.s_T_HGC * self.m.S_T_HGC_I[i] + self.s_T_HXC * self.m.S_T_HXC_I[i] + self.s_T_HGS * self.m.S_T_HGS_I[i] + sum(sum(self.s_T_GS_W * self.m.S_T_GS_W_I_WR_WC[i,c,r] for r in self.wr_GS) for c in self.wc_GS) + sum(sum(self.s_T_GS_C * self.m.S_T_GS_C_I_CR_CC[i,c,r] for r in self.cr_GS) for c in self.cc_GS) + self.s_T_CS * self.m.S_T_CS_I[i] + self.s_T_RLTS * self.m.S_T_RLTS_I[i] + self.s_T_HXA * self.m.S_T_HXA_I[i] + self.s_T_HXC * self.m.S_T_HXC_w_I[i] + self.s_T_HXH * self.m.S_T_HXH_b_I[i] + self.s_T_HXH * self.m.S_T_HXH_w_I[i] + self.s_T_HXC * self.m.S_T_HXC_b_I[i]))

        ## General toggle constraint
        self.m.Constraint_Toggle_I = pyo.Constraint(expr = self.m.T_TOT_I_ == sum(self.m.T_OP_I[i] for i in self.I[0:-1]))

        ## Toggle constraints
        self.m.Constraint_Toggle_time_I = pyo.ConstraintList()
        for i in self.I[0:-1]:
            self.m.Constraint_Toggle_time_I.add(self.m.T_OP_I[i] == self.m.Z_HP_I_3[i] * self.c_switch_HP_3 + self.m.Z_HXH_HS_I[i] * self.c_switch_HXH_HS + self.m.Z_HGC_HGCHXC_I[i] * self.c_switch_HGC_HGCHXC + self.m.Z_HXA_I[i] * self.c_switch_HXA + self.m.Z_HXH_HGC_I[i] * self.c_switch_HXH_HGC + self.m.Z_HS_HGS_I[i] * self.c_switch_HS_HGS + self.m.Z_HGS_CS_I[i] * self.c_switch_HGS_CS + self.m.Z_VP_I[i] * self.c_switch_VP_lin)

        ## HP
        self.m.Constraint_HP_I = pyo.ConstraintList()
        for i in self.I[0:-1]:
            self.m.Constraint_HP_I.add(sum(self.m.B_HP_H_I[h,i] for h in self.H) == 1) ## SOS1 constraint

            self.m.Constraint_HP_I.add(self.m.Q_HP_HT_I[i] == self.a_HP_HT_0 + self.a_HP_HT_1 * self.m.T_HP_HT_in_I[i] + self.a_HP_HT_2 * self.m.T_HP_LT_in_I[i]) ## Thermal equation HT
            self.m.Constraint_HP_I.add(self.m.Q_HP_LT_I[i] == self.a_HP_LT_0 + self.a_HP_LT_1 * self.m.T_HP_HT_in_I[i] + self.a_HP_LT_2 * self.m.T_HP_LT_in_I[i]) ## Thermal equation LT
            self.m.Constraint_HP_I.add(self.m.E_HP_EL_I[i] == self.a_HP_EL_0 + self.a_HP_EL_1 * self.m.T_HP_HT_in_I[i] + self.a_HP_EL_2 * self.m.T_HP_LT_in_I[i]) ## Electrical equation 
            
            for h in self.H:
                self.m.Constraint_HP_I.add(self.m.Z_HP_Q_HT_H_I[h,i] <= self.q_HP_HT_max) ## Big M to gain self.m.Z_HP_Q_HT_H_I[0][i] as Q dot
                self.m.Constraint_HP_I.add(self.m.Z_HP_Q_HT_H_I[h,i] >= self.q_HP_HT_min) ## Big M
                self.m.Constraint_HP_I.add(self.m.Z_HP_Q_HT_H_I[h,i] <= self.m.B_HP_H_I[h,i] * self.q_HP_HT_max) ## Big M 
                self.m.Constraint_HP_I.add(self.m.Z_HP_Q_HT_H_I[h,i] >= self.m.B_HP_H_I[h,i] * self.q_HP_HT_min) ## Big M 
                self.m.Constraint_HP_I.add(self.m.Z_HP_Q_HT_H_I[h,i] <= self.m.Q_HP_HT_I[i] - (1-self.m.B_HP_H_I[h,i]) * self.q_HP_HT_min) ## Big M 
                self.m.Constraint_HP_I.add(self.m.Z_HP_Q_HT_H_I[h,i] >= self.m.Q_HP_HT_I[i] - (1-self.m.B_HP_H_I[h,i]) * self.q_HP_HT_max) ## Big M 

                self.m.Constraint_HP_I.add(self.m.Z_HP_Q_LT_H_I[h,i] <= self.q_HP_LT_max) ## Big M to gain self.m.Z_HP_Q_LT_H_I[0][i] as Q dot
                self.m.Constraint_HP_I.add(self.m.Z_HP_Q_LT_H_I[h,i] >= self.q_HP_LT_min) ## Big M 
                self.m.Constraint_HP_I.add(self.m.Z_HP_Q_LT_H_I[h,i] <= self.m.B_HP_H_I[h,i] * self.q_HP_LT_max) ## Big M 
                self.m.Constraint_HP_I.add(self.m.Z_HP_Q_LT_H_I[h,i] >= self.m.B_HP_H_I[h,i] * self.q_HP_LT_min) ## Big M 
                self.m.Constraint_HP_I.add(self.m.Z_HP_Q_LT_H_I[h,i] <= self.m.Q_HP_LT_I[i] - (1-self.m.B_HP_H_I[h,i]) * self.q_HP_LT_min) ## Big M
                self.m.Constraint_HP_I.add(self.m.Z_HP_Q_LT_H_I[h,i] >= self.m.Q_HP_LT_I[i] - (1-self.m.B_HP_H_I[h,i]) * self.q_HP_LT_max) ## Big M  

                self.m.Constraint_HP_I.add(self.m.Z_HP_E_EL_H_I[h,i] <= self.q_HP_EL_max) ## Big M to gain self.m.Z_HP_Q_H_I[0][i] as Q dot
                self.m.Constraint_HP_I.add(self.m.Z_HP_E_EL_H_I[h,i] >= self.q_HP_EL_min) ## Big M 
                self.m.Constraint_HP_I.add(self.m.Z_HP_E_EL_H_I[h,i] <= self.m.B_HP_H_I[h,i] * self.q_HP_EL_max) ## Big M 
                self.m.Constraint_HP_I.add(self.m.Z_HP_E_EL_H_I[h,i] >= self.m.B_HP_H_I[h,i] * self.q_HP_EL_min) ## Big M 
                self.m.Constraint_HP_I.add(self.m.Z_HP_E_EL_H_I[h,i] <= self.m.E_HP_EL_I[i] - (1-self.m.B_HP_H_I[h,i]) * self.q_HP_EL_min) ## Big M 
                self.m.Constraint_HP_I.add(self.m.Z_HP_E_EL_H_I[h,i] >= self.m.E_HP_EL_I[i] - (1-self.m.B_HP_H_I[h,i]) * self.q_HP_EL_max) ## Big M 

            self.m.Constraint_HP_I.add(self.m.Q_HP_HT_I[i] == sum(self.m.Z_HP_Q_HT_H_I[h,i] for h in self.H)) ## Thighten Relaxation problem
            self.m.Constraint_HP_I.add(self.m.Q_HP_LT_I[i] == sum(self.m.Z_HP_Q_LT_H_I[h,i] for h in self.H)) ## Thighten Relaxation problem
            self.m.Constraint_HP_I.add(self.m.E_HP_EL_I[i] == sum(self.m.Z_HP_E_EL_H_I[h,i] for h in self.H)) ## Thighten Relaxation problem
            
            self.m.Constraint_HP_I.add(self.m.T_HP_HT_out_I[i] == self.m.T_HP_HT_in_I[i] + 1/self.c_w * (sum(self.d_HP_power_H[h] * 1/self.mdot_HP_w_H[h] * self.m.Z_HP_Q_HT_H_I[h,i] for h in self.H[1:]))) ## HT last part for division by zero
            self.m.Constraint_HP_I.add(self.m.T_HP_LT_out_I[i] == self.m.T_HP_LT_in_I[i] - 1/self.c_b * (sum(self.d_HP_power_H[h] * 1/self.mdot_HP_b_H[h] * self.m.Z_HP_Q_LT_H_I[h,i] for h in self.H[1:]))) ## LT last part for division by zero
            self.m.Constraint_HP_I.add(self.m.E_HP_EL_in_I[i] == sum(self.d_HP_power_H[h] * self.m.Z_HP_E_EL_H_I[h,i] for h in self.H) + sum(self.e_HP_EL_pumps[h] * self.m.B_HP_H_I[h,i] for h in self.H)) ## EL

            self.m.Constraint_HP_I.add(self.m.T_HP_LT_in_I[i] >= -15) ## physical constraints
            self.m.Constraint_HP_I.add(self.m.T_HP_LT_out_I[i] >= -15) ## physical constraints
            self.m.Constraint_HP_I.add(self.m.T_HP_HT_in_I[i] <= 60) ## physical constraints

        for i in self.I[0:-(self.ControlPeriod3)]:
            if i%self.ControlPeriod3 == 0:
                for j in range(1,self.ControlPeriod3):
                    for h in self.H:
                        self.m.Constraint_HP_I.add(self.m.B_HP_H_I[h,i] == self.m.B_HP_H_I[h,i+j])

        for i in self.I[0:-2]:
            self.m.Constraint_HP_I.add(self.m.Z_HP_I_3[i] >= self.m.B_HP_H_I[1,i+1] + self.m.B_HP_H_I[0,i] - 1)
            self.m.Constraint_HP_I.add(self.m.Z_HP_I_3[i] >= self.m.B_HP_H_I[2,i+1] + self.m.B_HP_H_I[0,i] - 1)
            self.m.Constraint_HP_I.add(self.m.Z_HP_I_3[i] >= self.m.B_HP_H_I[3,i+1] + self.m.B_HP_H_I[0,i] - 1)
            self.m.Constraint_HP_I.add(self.m.Z_HP_I_3[i] >= self.m.B_HP_H_I[4,i+1] + self.m.B_HP_H_I[0,i] - 1)

            self.m.Constraint_HP_I.add(self.m.Z_HP_I_3[i] >= self.m.B_HP_H_I[3,i+1] + self.m.B_HP_H_I[1,i] - 1)
            self.m.Constraint_HP_I.add(self.m.Z_HP_I_3[i] >= self.m.B_HP_H_I[4,i+1] + self.m.B_HP_H_I[1,i] - 1)

            self.m.Constraint_HP_I.add(self.m.Z_HP_I_3[i] >= self.m.B_HP_H_I[3,i+1] + self.m.B_HP_H_I[2,i] - 1)
            self.m.Constraint_HP_I.add(self.m.Z_HP_I_3[i] >= self.m.B_HP_H_I[4,i+1] + self.m.B_HP_H_I[2,i] - 1)

            self.m.Constraint_HP_I.add(self.m.Z_HP_I_3[i] <= 1)
        
        if self.End_Toggle_Constraints == True:
            self.m.Constraint_HP_I.add(self.m.Z_HP_I_3[self.I[-2]] >= self.B_HP_1_end + self.m.B_HP_H_I[0,self.I[-2]] - 1)
            self.m.Constraint_HP_I.add(self.m.Z_HP_I_3[self.I[-2]] >= self.B_HP_2_end + self.m.B_HP_H_I[0,self.I[-2]] - 1)
            self.m.Constraint_HP_I.add(self.m.Z_HP_I_3[self.I[-2]] >= self.B_HP_3_end + self.m.B_HP_H_I[0,self.I[-2]] - 1)
            self.m.Constraint_HP_I.add(self.m.Z_HP_I_3[self.I[-2]] >= self.B_HP_4_end + self.m.B_HP_H_I[0,self.I[-2]] - 1)

            self.m.Constraint_HP_I.add(self.m.Z_HP_I_3[self.I[-2]] >= self.B_HP_3_end + self.m.B_HP_H_I[1,self.I[-2]] - 1)
            self.m.Constraint_HP_I.add(self.m.Z_HP_I_3[self.I[-2]] >= self.B_HP_4_end + self.m.B_HP_H_I[1,self.I[-2]] - 1)

            self.m.Constraint_HP_I.add(self.m.Z_HP_I_3[self.I[-2]] >= self.B_HP_3_end + self.m.B_HP_H_I[2,self.I[-2]] - 1)
            self.m.Constraint_HP_I.add(self.m.Z_HP_I_3[self.I[-2]] >= self.B_HP_4_end + self.m.B_HP_H_I[2,self.I[-2]] - 1)

            self.m.Constraint_HP_I.add(self.m.Z_HP_I_3[self.I[-2]] <= 1)
        
        # HP HT Tank
        self.m.Constraint_HP_I.add(self.m.T_HP_HT_I[0] == self.T_HP_HT_start) ## Start temperature

        for i in self.I[0:-1]:
            self.m.Constraint_HP_I.add(self.m.T_HP_HT_I[i+1] == self.m.T_HP_HT_I[i] + self.StepSizeInSec2 * (1/(self.m_HP_HT_w * self.c_w) * (self.c_w * sum(self.mdot_HP_w_H[h] * self.m.Z_HP_HT_in_H_I[h,i] for h in self.H) - self.c_w * sum(self.mdot_HP_w_H[h] * self.m.Z_HP_HXH_H_I[h,i] for h in self.H) - self.c_w * sum(self.mdot_HP_w_H[h] * self.m.Z_HP_HS_H_I[h,i] for h in self.H))) + self.StepSizeInSec2 * self.alpha_HP_time * (self.t_default - self.m.T_HP_HT_I[i+1])/(self.m_HP_HT_w * self.c_w)) ## General energy flow
        
        for i in self.I[0:-1]:
            for h in self.H:
                self.m.Constraint_HP_I.add(self.m.Z_HP_HT_in_H_I[h,i] <= self.T_HP_delta_max) ## Big M constraint input
                self.m.Constraint_HP_I.add(self.m.Z_HP_HT_in_H_I[h,i] >= self.T_HP_delta_min) ## Big M constraint input
                self.m.Constraint_HP_I.add(self.m.Z_HP_HT_in_H_I[h,i] <= self.T_HP_delta_max * self.m.B_HP_H_I[h,i]) ## Big M constraint input
                self.m.Constraint_HP_I.add(self.m.Z_HP_HT_in_H_I[h,i] >= self.T_HP_delta_min * self.m.B_HP_H_I[h,i]) ## Big M constraint input
                self.m.Constraint_HP_I.add(self.m.Z_HP_HT_in_H_I[h,i] <= (self.m.T_HP_HT_out_I[i] - self.m.T_HP_HT_I[i+1]) - (1 - self.m.B_HP_H_I[h,i]) * self.T_HP_delta_min) ## Big M constraint input
                self.m.Constraint_HP_I.add(self.m.Z_HP_HT_in_H_I[h,i] >= (self.m.T_HP_HT_out_I[i] - self.m.T_HP_HT_I[i+1]) - (1 - self.m.B_HP_H_I[h,i]) * self.T_HP_delta_max) ## Big M constraint input

        for i in self.I[0:-1]:
            self.m.Constraint_HP_I.add(self.m.T_HP_HT_in_I[i] == self.m.T_HP_HT_I[i+1])

        for i in self.I[1:]:
            self.m.Constraint_HP_I.add(self.m.T_HP_HT_I[i] <= self.T_HP_HT_max + self.m.S_T_HP_I[i]) ## Temperature range tank
            self.m.Constraint_HP_I.add(self.m.T_HP_HT_I[i] >= self.T_HP_HT_min - self.m.S_T_HP_I[i]) ## Temperature range tank

        # HP LT Tank
        self.m.Constraint_HP_I.add(self.m.T_HP_LT_I[0] == self.T_HP_LT_start) ## Start temperature

        for i in self.I[0:-1]:
            self.m.Constraint_HP_I.add(self.m.T_HP_LT_I[i+1] == self.m.T_HP_LT_I[i] + self.StepSizeInSec2 * (1/(self.m_HP_LT_b * self.c_b) * (self.c_b * sum(self.mdot_HP_b_H[h] * self.m.Z_HP_LT_in_H_I[h,i] for h in self.H) + self.c_b * sum(self.mdot_HP_b_H[h] * self.m.Z_HP_HGC_2_H_I[h,i] for h in self.H))) + self.StepSizeInSec2 * self.alpha_HP_time * (self.t_default - self.m.T_HP_LT_I[i+1])/(self.m_HP_LT_b * self.c_b)) ## General energy flow
        
        for i in self.I[0:-1]:
            for h in self.H:    
                self.m.Constraint_HP_I.add(self.m.Z_HP_LT_in_H_I[h,i] <= self.T_HP_delta_max) ## Big M constraint input
                self.m.Constraint_HP_I.add(self.m.Z_HP_LT_in_H_I[h,i] >= self.T_HP_delta_min) ## Big M constraint input
                self.m.Constraint_HP_I.add(self.m.Z_HP_LT_in_H_I[h,i] <= self.T_HP_delta_max * self.m.B_HP_H_I[h,i]) ## Big M constraint input
                self.m.Constraint_HP_I.add(self.m.Z_HP_LT_in_H_I[h,i] >= self.T_HP_delta_min * self.m.B_HP_H_I[h,i]) ## Big M constraint input
                self.m.Constraint_HP_I.add(self.m.Z_HP_LT_in_H_I[h,i] <= (self.m.T_HP_LT_out_I[i] - self.m.T_HP_LT_I[i+1]) - (1 - self.m.B_HP_H_I[h,i]) * self.T_HP_delta_min) ## Big M constraint input
                self.m.Constraint_HP_I.add(self.m.Z_HP_LT_in_H_I[h,i] >= (self.m.T_HP_LT_out_I[i] - self.m.T_HP_LT_I[i+1]) - (1 - self.m.B_HP_H_I[h,i]) * self.T_HP_delta_max) ## Big M constraint input

        for i in self.I[0:-1]:
            self.m.Constraint_HP_I.add(self.m.T_HP_LT_in_I[i] == self.m.T_HP_LT_I[i+1])

        for i in self.I[1:]:
            self.m.Constraint_HP_I.add(self.m.T_HP_LT_I[i] <= self.T_HP_LT_max + self.m.S_T_HP_I[i]) ## Temperature range tank
            self.m.Constraint_HP_I.add(self.m.T_HP_LT_I[i] >= self.T_HP_LT_min - self.m.S_T_HP_I[i]) ## Temperature range tank
        
        ## HS
        self.m.Constraint_HS_I = pyo.ConstraintList()
        self.m.Constraint_HS_I.add(self.m.T_HS_I[0] == self.T_HS_start) ## Start temperature
        if self.End_Temp_Constraints == True:
            self.m.Constraint_HS_I.add(self.m.T_HS_I[self.I[-1]] >= self.T_HS_end - self.m.S_T_HS_I[self.I[-1]])

        for i in self.I[0:-1]: 
            self.m.Constraint_HS_I.add(self.m.T_HS_I[i+1] == self.m.T_HS_I[i] + self.StepSizeInSec2 * (1/(self.m_HS_w * self.c_w) * (self.c_w * sum(self.mdot_HP_w_H[h] * self.m.Z_HP_HS_H_I[h,i] for h in self.H) - self.c_w * self.mdot_IS_w * self.m.W_HS_IS_I[i])) - self.StepSizeInSec2 * self.q_dem_HS_I[i]/(self.m_HS_w * self.c_w) + self.StepSizeInSec2 * self.alpha_HS_time * (self.t_default - self.m.T_HS_I[i+1])/(self.m_HS_w * self.c_w)) ## General energy flow

        for i in self.I[1:]:
            self.m.Constraint_HS_I.add(self.m.T_HS_I[i] <= self.T_HS_max + self.m.S_T_HS_I[i]) ## Temperature range tank
            self.m.Constraint_HS_I.add(self.m.T_HS_I[i] >= self.T_HS_min - self.m.S_T_HS_I[i]) ## Temperature range tank

        ## CS
        self.m.Constraint_CS_I = pyo.ConstraintList()
        self.m.Constraint_CS_I.add(self.m.T_CS_I[0] == self.T_CS_start) ## Start temperature
        if self.End_Temp_Constraints == True:
            self.m.Constraint_CS_I.add(self.m.T_CS_I[self.I[-1]] <= self.T_CS_end + self.m.S_T_CS_I[self.I[-1]])

        for i in self.I[0:-1]:
            self.m.Constraint_CS_I.add(self.m.T_CS_I[i+1] == self.m.T_CS_I[i] + self.StepSizeInSec2 * (1/(self.m_CS_w * self.c_w) * (self.c_w * self.mdot_GS_w * self.m.W_GS_CS_I[i] + self.c_w * self.mdot_VP_tot * self.m.W_HXC_CS_I[i])) + self.StepSizeInSec2 * self.q_dem_CS_I[i]/(self.m_CS_w * self.c_w) + self.StepSizeInSec2 * self.alpha_CS_time * (self.t_default - self.m.T_CS_I[i+1])/(self.m_CS_w * self.c_w)) ## General energy flow

        for i in self.I[1:]:
            self.m.Constraint_CS_I.add(self.m.T_CS_I[i] <= self.T_CS_max + self.m.S_T_CS_I[i]) ## Temperature range tank
            self.m.Constraint_CS_I.add(self.m.T_CS_I[i] >= self.T_CS_min - self.m.S_T_CS_I[i]) ## Temperature range tank

        ## RLTS
        self.m.Constraint_RLTS_I = pyo.ConstraintList()
        self.m.Constraint_RLTS_I.add(self.m.T_RLTS_I[0] == self.T_RLTS_start) ## Start temperature
        if self.End_Temp_Constraints == True:
            self.m.Constraint_RLTS_I.add(self.m.T_RLTS_I[self.I[-1]] <= self.T_RLTS_end + self.m.S_T_RLTS_I[self.I[-1]])


        for i in self.I[0:-1]:
            self.m.Constraint_RLTS_I.add(self.m.T_RLTS_I[i+1] == self.m.T_RLTS_I[i] + self.StepSizeInSec2 * (1/(self.m_RLTS_w * self.c_w) * (self.c_w * self.mdot_VP_tot * self.m.W_HXC_RLTS_I[i])) + self.StepSizeInSec2 * self.q_dem_RLTS_I[i]/(self.m_RLTS_w * self.c_w) + self.StepSizeInSec2 * self.alpha_RLTS_time * (self.t_default - self.m.T_RLTS_I[i+1])/(self.m_RLTS_w * self.c_w)) ## General energy flow

        for i in self.I[1:]:
            self.m.Constraint_RLTS_I.add(self.m.T_RLTS_I[i] <= self.T_RLTS_max + self.m.S_T_RLTS_I[i]) ## Temperature range tank
            self.m.Constraint_RLTS_I.add(self.m.T_RLTS_I[i] >= self.T_RLTS_min - self.m.S_T_RLTS_I[i]) ## Temperature range tank

        ## IS
        self.m.Constraint_IS_I = pyo.ConstraintList()
        for i in self.I[:-1]:
            for r in self.cr_IS[1:]:
                self.m.Constraint_IS_I.add(self.m.Q_IS_C_NORTH_I_CR[i,r] == (self.m.T_IS_C_I_CR[i+1,r-1] - self.m.T_IS_C_I_CR[i+1,r]) * self.lambda_IS_c_c / self.height_IS * self.a_north_south_IS)
    
            for r in self.cr_IS[:-1]:
                self.m.Constraint_IS_I.add(self.m.Q_IS_C_SOUTH_I_CR[i,r] == (self.m.T_IS_C_I_CR[i+1,r+1] - self.m.T_IS_C_I_CR[i+1,r]) * self.lambda_IS_c_c / self.height_IS * self.a_north_south_IS)

            for r in self.wr_IS:
                self.m.Constraint_IS_I.add(self.m.Q_IS_C_W_I_WR[i,r] == (self.m.T_IS_W_I_WR[i+1,r] - self.m.T_IS_C_I_CR[i+1,r]) * self.alpha_IS_w_c * self.a_pipe_IS)

        ## Concrete borders
        for i in self.I[:-1]:
            self.m.Constraint_IS_I.add(self.m.Q_IS_C_NORTH_I_CR[i,0] == (self.t_IS_air - self.m.T_IS_C_I_CR[i+1,0]) * self.lambda_IS_c_a / self.height_IS * self.a_north_south_IS)
            self.m.Constraint_IS_I.add(self.m.Q_IS_C_SOUTH_I_CR[i,self.cr_IS[-1]] == (self.t_IS_air - self.m.T_IS_C_I_CR[i+1,self.cr_IS[-1]]) * self.lambda_IS_c_a / self.height_IS * self.a_north_south_IS)  

        ## Concrete temperature
            self.m.Constraint_IS_I.add(self.m.T_IS_C_I_CR[0,0] == self.T_IS_c_1_start)
            self.m.Constraint_IS_I.add(self.m.T_IS_C_I_CR[0,1] == self.T_IS_c_2_start)
            self.m.Constraint_IS_I.add(self.m.T_IS_C_I_CR[0,2] == self.T_IS_c_3_start)
            self.m.Constraint_IS_I.add(self.m.T_IS_C_I_CR[0,3] == self.T_IS_c_4_start)
            self.m.Constraint_IS_I.add(self.m.T_IS_C_I_CR[0,4] == self.T_IS_c_5_start)

        for i in self.I[:-1]:
            for r in self.cr_IS:
                if r in self.wr_IS:
                    self.m.Constraint_IS_I.add(self.m.T_IS_C_I_CR[i+1,r] == self.m.T_IS_C_I_CR[i,r] + self.StepSizeInSec2 * (1/(self.m_IS_c * self.c_c)) * (self.m.Q_IS_C_NORTH_I_CR[i,r] + self.m.Q_IS_C_SOUTH_I_CR[i,r] + self.m.Q_IS_C_W_I_WR[i,r]))
                else:
                    self.m.Constraint_IS_I.add(self.m.T_IS_C_I_CR[i+1,r] == self.m.T_IS_C_I_CR[i,r] + self.StepSizeInSec2 * (1/(self.m_IS_c * self.c_c)) * (self.m.Q_IS_C_NORTH_I_CR[i,r] + self.m.Q_IS_C_SOUTH_I_CR[i,r]))

        for i in self.I[1:]:
            for r in self.cr_IS:
                self.m.Constraint_IS_I.add(self.m.T_IS_C_I_CR[i,r] <= self.T_IS_max_c + self.m.S_T_IS_C_I_CR[i,r])
                self.m.Constraint_IS_I.add(self.m.T_IS_C_I_CR[i,r] >= self.T_IS_min_c - self.m.S_T_IS_C_I_CR[i,r])

        ## Water energy flows
        for i in self.I[:-1]:
            for r in self.wr_IS[1:-1]:
                self.m.Constraint_IS_I.add(self.m.Q_IS_W_I_WR[i,r] == (self.m.T_IS_W_I_WR[i+1,r-2] - self.m.T_IS_W_I_WR[i+1,r]) * self.mdot_IS_w/self.n_IS_blocks * self.c_w + (self.m.T_IS_W_I_WR[i+1,r+2] - self.m.T_IS_W_I_WR[i+1,r]) * self.mdot_IS_w/self.n_IS_blocks * self.c_w)
            for r in self.wr_IS[0:1]:
                self.m.Constraint_IS_I.add(self.m.Q_IS_W_I_WR[i,r] == (self.m.T_IS_W_I_WR[i+1,r+2] - self.m.T_IS_W_I_WR[i+1,r]) * self.mdot_IS_w/self.n_IS_blocks * self.c_w)
            # for r in self.wr_IS[-1]
            self.m.Constraint_IS_I.add(self.m.Q_IS_W_I_WR[i,self.wr_IS[-1]] == (self.m.T_IS_W_I_WR[i+1,self.wr_IS[-1]-2] - self.m.T_IS_W_I_WR[i+1,self.wr_IS[-1]]) * self.mdot_IS_w/self.n_IS_blocks * self.c_w)

            for r in self.wr_IS:
                self.m.Constraint_IS_I.add(self.m.Q_IS_W_C_I_WR[i,r] == (self.m.T_IS_C_I_CR[i+1,r] - self.m.T_IS_W_I_WR[i+1,r]) * self.alpha_IS_w_c * self.a_pipe_IS)  
        
        ## Water borders
        for i in self.I[:-1]:
            self.m.Constraint_IS_I.add(self.m.Q_IS_W_I_IN[i] == -(self.m.W_HS_IS_I_2[i] + self.m.W_IS_HGS_I_2[i]) * self.mdot_IS_w/self.n_IS_blocks * self.c_w)

        ## Water temperature
        self.m.Constraint_IS_I.add(self.m.T_IS_W_I_WR[0,0] == self.T_IS_w_1_start)
        self.m.Constraint_IS_I.add(self.m.T_IS_W_I_WR[0,2] == self.T_IS_w_2_start)
        self.m.Constraint_IS_I.add(self.m.T_IS_W_I_WR[0,4] == self.T_IS_w_3_start)

        for i in self.I[:-1]:
            for r in self.wr_IS:
                if r == 0:
                    self.m.Constraint_IS_I.add(self.m.T_IS_W_I_WR[i+1,r] == self.m.T_IS_W_I_WR[i,r] + self.StepSizeInSec2 * (1/(self.m_IS_w * self.c_w)) * (self.m.Q_IS_W_I_WR[i,r] + self.m.Q_IS_W_C_I_WR[i,r] + self.m.Q_IS_W_I_IN[i]))
                else:
                    self.m.Constraint_IS_I.add(self.m.T_IS_W_I_WR[i+1,r] == self.m.T_IS_W_I_WR[i,r] + self.StepSizeInSec2 * (1/(self.m_IS_w * self.c_w)) * (self.m.Q_IS_W_I_WR[i,r] + self.m.Q_IS_W_C_I_WR[i,r]))

        for i in self.I[1:]:
            for r in self.wr_IS:
                self.m.Constraint_IS_I.add(self.m.T_IS_W_I_WR[i,r] <= self.T_IS_max_w + self.m.S_T_IS_W_I_WR[i,r])
                self.m.Constraint_IS_I.add(self.m.T_IS_W_I_WR[i,r] >= self.T_IS_min_w - self.m.S_T_IS_W_I_WR[i,r])

        ## Water outflow
        for i in self.I:
                self.m.Constraint_IS_I.add(self.m.T_IS_I[i] == self.m.T_IS_W_I_WR[i,self.wr_IS[-1]])
        # Old
        for i in self.I[0:-1]:
            self.m.Constraint_IS_I.add(self.m.Z_IS_pump_I[i] >= self.m.V_HS_IS_I[i])
            self.m.Constraint_IS_I.add(self.m.Z_IS_pump_I[i] >= self.m.V_IS_HGS_I[i])
            self.m.Constraint_IS_I.add(self.m.Z_IS_pump_I[i] <= self.m.V_HS_IS_I[i] + self.m.V_IS_HGS_I[i])
            self.m.Constraint_IS_I.add(self.m.Z_IS_pump_I[i] <= 1)
            self.m.Constraint_IS_I.add(self.m.E_IS_EL_I[i] == self.m.Z_IS_pump_I[i] * self.e_IS_EL)

        # Flows must be smaller or equal to 1, so that we cant use massflows x2. 
        for i in self.I[0:-1]:
            self.m.Constraint_IS_I.add(self.m.V_HS_IS_I[i] + self.m.V_IS_HGS_I[i] <= 1)

        for i in self.I[:-2]:
            self.m.Constraint_IS_I.add(self.m.Z_HS_HGS_I[i] >= self.m.V_HS_IS_I[i] - self.m.V_HS_IS_I[i+1])
            self.m.Constraint_IS_I.add(self.m.Z_HS_HGS_I[i] >= self.m.V_HS_IS_I[i+1] - self.m.V_HS_IS_I[i])
            self.m.Constraint_IS_I.add(self.m.Z_HS_HGS_I[i] >= self.m.V_IS_HGS_I[i] - self.m.V_IS_HGS_I[i+1])
            self.m.Constraint_IS_I.add(self.m.Z_HS_HGS_I[i] >= self.m.V_IS_HGS_I[i+1] - self.m.V_IS_HGS_I[i])
            self.m.Constraint_IS_I.add(self.m.Z_HS_HGS_I[i] <= 1)
        
        if self.End_Toggle_Constraints == True:
            self.m.Constraint_IS_I.add(self.m.Z_HS_HGS_I[self.I[-2]] >= self.m.V_HS_IS_I[self.I[-2]] - self.V_HS_IS_end)
            self.m.Constraint_IS_I.add(self.m.Z_HS_HGS_I[self.I[-2]] >= self.V_HS_IS_end - self.m.V_HS_IS_I[self.I[-2]])
            self.m.Constraint_IS_I.add(self.m.Z_HS_HGS_I[self.I[-2]] >= self.m.V_IS_HGS_I[self.I[-2]] - self.V_IS_HGS_end)
            self.m.Constraint_IS_I.add(self.m.Z_HS_HGS_I[self.I[-2]] >= self.V_IS_HGS_end - self.m.V_IS_HGS_I[self.I[-2]])
            self.m.Constraint_IS_I.add(self.m.Z_HS_HGS_I[self.I[-2]] <= 1)

        for i in self.I[:-1]:
            self.m.Constraint_IS_I.add(self.m.W_HS_IS_I_2[i] == -self.m.W_HS_IS_I[i])
            self.m.Constraint_IS_I.add(self.m.W_IS_HGS_I_2[i] == self.m.W_IS_HGS_I[i])

        ## GS
        self.m.Constraint_GS_I = pyo.ConstraintList()
        ## Concrete energy flows
        for i in self.I[:-1]:
            for c in self.cc_GS:
                for r in self.cr_GS[1:]:
                    self.m.Constraint_GS_I.add(self.m.Q_GS_C_NORTH_I_CR_CC[i,c,r] == (self.m.T_GS_C_I_CR_CC[i+1,c,r-1] - self.m.T_GS_C_I_CR_CC[i+1,c,r]) * self.lambda_GS_c_c / self.height_GS * self.a_north_south_GS)
        
            for c in self.cc_GS:
                for r in self.cr_GS[:-1]:
                    self.m.Constraint_GS_I.add(self.m.Q_GS_C_SOUTH_I_CR_CC[i,c,r] == (self.m.T_GS_C_I_CR_CC[i+1,c,r+1] - self.m.T_GS_C_I_CR_CC[i+1,c,r]) * self.lambda_GS_c_c / self.height_GS * self.a_north_south_GS)

            for c in self.cc_GS[1:]:
                for r in self.cr_GS:
                    self.m.Constraint_GS_I.add(self.m.Q_GS_C_WEST_I_CR_CC[i,c,r] == (self.m.T_GS_C_I_CR_CC[i+1,c-1,r] - self.m.T_GS_C_I_CR_CC[i+1,c,r]) * self.lambda_GS_c_c / self.width_GS * self.a_east_west_GS)

            for c in self.cc_GS[:-1]:
                for r in self.cr_GS:
                    self.m.Constraint_GS_I.add(self.m.Q_GS_C_EAST_I_CR_CC[i,c,r] == (self.m.T_GS_C_I_CR_CC[i+1,c+1,r] - self.m.T_GS_C_I_CR_CC[i+1,c,r]) * self.lambda_GS_c_c / self.width_GS * self.a_east_west_GS)

            for c in self.wc_GS:
                for r in self.wr_GS:
                    self.m.Constraint_GS_I.add(self.m.Q_GS_C_W_I_WR_WC[i,c,r] == (self.m.T_GS_W_I_WR_WC[i+1,c,r] - self.m.T_GS_C_I_CR_CC[i+1,c,r]) * self.alpha_GS_w_c * self.a_pipe_GS)

        ## Concrete borders
        for i in self.I[:-1]:
            for c in self.cc_GS:
                self.m.Constraint_GS_I.add(self.m.Q_GS_C_NORTH_I_CR_CC[i,c,0] == (self.t_GS_air - self.m.T_GS_C_I_CR_CC[i+1,c,0]) * self.lambda_GS_c_a / self.height_GS * self.a_north_south_GS)

            for c in self.cc_GS:
                self.m.Constraint_GS_I.add(self.m.Q_GS_C_SOUTH_I_CR_CC[i,c,self.cr_GS[-1]] == (self.t_GS_soil - self.m.T_GS_C_I_CR_CC[i+1,c,self.cr_GS[-1]]) * self.lambda_GS_c_s / self.height_GS * self.a_north_south_GS)

            if len(self.cc_GS) > 1:
                for r in self.cr_GS:
                    self.m.Constraint_GS_I.add(self.m.Q_GS_C_WEST_I_CR_CC[i,0,r] == (self.m.T_GS_C_I_CR_CC[i+1,self.cc_GS[-1],r] - self.m.T_GS_C_I_CR_CC[i+1,0,r]) * self.lambda_GS_c_c / self.width_GS * self.a_east_west_GS)

                for r in self.cr_GS:
                    self.m.Constraint_GS_I.add(self.m.Q_GS_C_EAST_I_CR_CC[i,self.cc_GS[-1],r] == (self.m.T_GS_C_I_CR_CC[i+1,0,r] - self.m.T_GS_C_I_CR_CC[i+1,self.cc_GS[-1],r]) * self.lambda_GS_c_c / self.width_GS * self.a_east_west_GS)
            else:
                for r in self.cr_GS:
                    self.m.Constraint_GS_I.add(self.m.Q_GS_C_WEST_I_CR_CC[i,0,r] == 0)

                for r in self.cr_GS:
                    self.m.Constraint_GS_I.add(self.m.Q_GS_C_EAST_I_CR_CC[i,self.cc_GS[-1],r] == 0)    

        ## Concrete temperature
        self.m.Constraint_GS_I.add(self.m.T_GS_C_I_CR_CC[0,0,0] == self.T_GS_c_1_start)
        self.m.Constraint_GS_I.add(self.m.T_GS_C_I_CR_CC[0,0,1] == self.T_GS_c_2_start)
        self.m.Constraint_GS_I.add(self.m.T_GS_C_I_CR_CC[0,0,2] == self.T_GS_c_3_start)
        self.m.Constraint_GS_I.add(self.m.T_GS_C_I_CR_CC[0,0,3] == self.T_GS_c_4_start)
        self.m.Constraint_GS_I.add(self.m.T_GS_C_I_CR_CC[0,0,4] == self.T_GS_c_5_start)
        self.m.Constraint_GS_I.add(self.m.T_GS_C_I_CR_CC[0,0,5] == self.T_GS_c_6_start)
        self.m.Constraint_GS_I.add(self.m.T_GS_C_I_CR_CC[0,0,6] == self.T_GS_c_7_start)

        for i in self.I[:-1]:
            for c in self.cc_GS:
                for r in self.cr_GS:
                    if r in self.wr_GS:
                        self.m.Constraint_GS_I.add(self.m.T_GS_C_I_CR_CC[i+1,c,r] == self.m.T_GS_C_I_CR_CC[i,c,r] + self.StepSizeInSec2 * (1/(self.m_GS_c * self.c_c)) * (self.m.Q_GS_C_NORTH_I_CR_CC[i,c,r] + self.m.Q_GS_C_SOUTH_I_CR_CC[i,c,r] + self.m.Q_GS_C_WEST_I_CR_CC[i,c,r] + self.m.Q_GS_C_EAST_I_CR_CC[i,c,r] + self.m.Q_GS_C_W_I_WR_WC[i,c,r]))
                    else:
                        self.m.Constraint_GS_I.add(self.m.T_GS_C_I_CR_CC[i+1,c,r] == self.m.T_GS_C_I_CR_CC[i,c,r] + self.StepSizeInSec2 * (1/(self.m_GS_c * self.c_c)) * (self.m.Q_GS_C_NORTH_I_CR_CC[i,c,r] + self.m.Q_GS_C_SOUTH_I_CR_CC[i,c,r] + self.m.Q_GS_C_WEST_I_CR_CC[i,c,r] + self.m.Q_GS_C_EAST_I_CR_CC[i,c,r]))
        
        for i in self.I[1:]:
            for c in self.cc_GS:
                for r in self.cr_GS:
                    self.m.Constraint_GS_I.add(self.m.T_GS_C_I_CR_CC[i,c,r] <= self.T_GS_max_c + self.m.S_T_GS_C_I_CR_CC[i,c,r])
                    self.m.Constraint_GS_I.add(self.m.T_GS_C_I_CR_CC[i,c,r] >= self.T_GS_min_c - self.m.S_T_GS_C_I_CR_CC[i,c,r])

        ## Water energy flows
        for i in self.I[:-1]:
            for r in self.wr_GS[::2]:
                for c in self.wc_GS[1:]:
                    self.m.Constraint_GS_I.add(self.m.Q_GS_W_WEST_I_WR_WC[i,c,r] == (self.m.T_GS_W_I_WR_WC[i+1,c-1,r] - self.m.T_GS_W_I_WR_WC[i+1,c,r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w)
                for c in self.wc_GS[:-1]:    
                    self.m.Constraint_GS_I.add(self.m.Q_GS_W_EAST_I_WR_WC[i,c,r] == (self.m.T_GS_W_I_WR_WC[i+1,c+1,r] - self.m.T_GS_W_I_WR_WC[i+1,c,r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w)

            for r in self.wr_GS[1::2]:
                for c in self.wc_GS[:-1]:
                    self.m.Constraint_GS_I.add(self.m.Q_GS_W_EAST_I_WR_WC[i,c,r] == (self.m.T_GS_W_I_WR_WC[i+1,c+1,r] - self.m.T_GS_W_I_WR_WC[i+1,c,r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w)
                for c in self.wc_GS[1:]:    
                    self.m.Constraint_GS_I.add(self.m.Q_GS_W_WEST_I_WR_WC[i,c,r] == (self.m.T_GS_W_I_WR_WC[i+1,c-1,r] - self.m.T_GS_W_I_WR_WC[i+1,c,r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w)

            for c in self.wc_GS:
                for r in self.wr_GS:
                    self.m.Constraint_GS_I.add(self.m.Q_GS_W_C_I_WR_WC[i,c,r] == (self.m.T_GS_C_I_CR_CC[i+1,c,r] - self.m.T_GS_W_I_WR_WC[i+1,c,r]) * self.alpha_GS_w_c * self.a_pipe_GS)      
        
        ## Water borders
        for i in self.I[:-1]:
            for r in self.wr_GS[0::2]:
                if r > 1:
                    self.m.Constraint_GS_I.add(self.m.Q_GS_W_WEST_I_WR_WC[i,0,r] == (self.m.T_GS_W_I_WR_WC[i+1,0,r-2] - self.m.T_GS_W_I_WR_WC[i+1,0,r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w)
                    self.m.Constraint_GS_I.add(self.m.Q_GS_W_EAST_I_WR_WC[i,self.wc_GS[-1],r] == 0)
                else: ## Start inflow
                    self.m.Constraint_GS_I.add(self.m.Q_GS_W_WEST_I_WR_WC[i,0,1] == -(self.c_w * self.mdot_GS_w/self.n_GS_blocks * self.m.W_GS_HGS_I_2[i] + self.c_w * self.mdot_GS_w/self.n_GS_blocks * self.m.W_GS_CS_I_2[i]))
                    self.m.Constraint_GS_I.add(self.m.Q_GS_W_EAST_I_WR_WC[i,self.wc_GS[-1],r] == (self.m.T_GS_W_I_WR_WC[i+1,self.wc_GS[-1],r+2] - self.m.T_GS_W_I_WR_WC[i+1,self.wc_GS[-1],r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w)
            for r in self.wr_GS[1::2]:
                self.m.Constraint_GS_I.add(self.m.Q_GS_W_EAST_I_WR_WC[i,self.wc_GS[-1],r] == (self.m.T_GS_W_I_WR_WC[i+1,self.wc_GS[-1],r-2] - self.m.T_GS_W_I_WR_WC[i+1,self.wc_GS[-1],r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w)
                self.m.Constraint_GS_I.add(self.m.Q_GS_W_WEST_I_WR_WC[i,0,r] == (self.m.T_GS_W_I_WR_WC[i+1,0,r+2] - self.m.T_GS_W_I_WR_WC[i+1,0,r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w)

        ## Water temperature
        self.m.Constraint_GS_I.add(self.m.T_GS_W_I_WR_WC[0,0,1] == self.T_GS_w_1_start)
        self.m.Constraint_GS_I.add(self.m.T_GS_W_I_WR_WC[0,0,3] == self.T_GS_w_2_start)
        self.m.Constraint_GS_I.add(self.m.T_GS_W_I_WR_WC[0,0,5] == self.T_GS_w_3_start)

        for i in self.I[:-1]:
            for c in self.wc_GS:
                for r in self.wr_GS:
                    self.m.Constraint_GS_I.add(self.m.T_GS_W_I_WR_WC[i+1,c,r] == self.m.T_GS_W_I_WR_WC[i,c,r] + self.StepSizeInSec2 * (1/(self.m_GS_w * self.c_w)) * (self.m.Q_GS_W_WEST_I_WR_WC[i,c,r] + self.m.Q_GS_W_EAST_I_WR_WC[i,c,r] + self.m.Q_GS_W_C_I_WR_WC[i,c,r]))

        for i in self.I[1:]:
            for c in self.wc_GS:
                for r in self.wr_GS:
                    self.m.Constraint_GS_I.add(self.m.T_GS_W_I_WR_WC[i,c,r] <= self.T_GS_max_w + self.m.S_T_GS_W_I_WR_WC[i,c,r])
                    self.m.Constraint_GS_I.add(self.m.T_GS_W_I_WR_WC[i,c,r] >= self.T_GS_min_w - self.m.S_T_GS_W_I_WR_WC[i,c,r])

        ## Water outflow
        for i in self.I:
            if self.wr_GS[::2][-1] > self.wr_GS[1::2][-1]:
                self.m.Constraint_GS_I.add(self.m.T_GS_I[i] == self.m.T_GS_W_I_WR_WC[i,self.wc_GS[-1],self.wr_GS[-1]])
            else:
                self.m.Constraint_GS_I.add(self.m.T_GS_I[i] == self.m.T_GS_W_I_WR_WC[i,self.wc_GS[0],self.wr_GS[-1]])

        for i in self.I[0:-1]:
            self.m.Constraint_GS_I.add(self.m.E_GS_EL_I[i] == (self.m.V_GS_HGS_I[i] + self.m.V_GS_CS_I[i]) * self.e_GS_EL)

        for i in self.I[0:-1]:
            self.m.Constraint_GS_I.add(self.m.W_GS_HGS_I_2[i] == self.m.W_GS_HGS_I[i])
            self.m.Constraint_GS_I.add(self.m.W_GS_CS_I_2[i] == self.m.W_GS_CS_I[i])

        ## V_HP_HXH_HS
        self.m.Constraint_V_HP_HXH_HS_I = pyo.ConstraintList()
        for i in self.I[0:-1]:
            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.W_HP_HXH_I[i] >= sum(self.V_HP_HXH_min * self.m.Z_HP_HXH_N_I[n,i] + self.T_HP_HXH_min_N[n] * self.m.V_HP_HXH_N_I[n,i] - self.V_HP_HXH_min * self.T_HP_HXH_min_N[n] * self.m.B_T_HP_HXH_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.W_HP_HXH_I[i] >= sum(self.V_HP_HXH_max * self.m.Z_HP_HXH_N_I[n,i] + self.T_HP_HXH_max_N[n] * self.m.V_HP_HXH_N_I[n,i] - self.V_HP_HXH_max * self.T_HP_HXH_max_N[n] * self.m.B_T_HP_HXH_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.W_HP_HXH_I[i] <= sum(self.V_HP_HXH_max * self.m.Z_HP_HXH_N_I[n,i] + self.T_HP_HXH_min_N[n] * self.m.V_HP_HXH_N_I[n,i] - self.V_HP_HXH_max * self.T_HP_HXH_min_N[n] * self.m.B_T_HP_HXH_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.W_HP_HXH_I[i] <= sum(self.V_HP_HXH_min * self.m.Z_HP_HXH_N_I[n,i] + self.T_HP_HXH_max_N[n] * self.m.V_HP_HXH_N_I[n,i] - self.V_HP_HXH_min * self.T_HP_HXH_max_N[n] * self.m.B_T_HP_HXH_N_I[n,i] for n in self.N_MC))

            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.T_HP_HXH_I[i] == sum(self.m.Z_HP_HXH_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.V_HP_HXH_I[i] == sum(self.m.V_HP_HXH_N_I[n,i] for n in self.N_MC))

        for i in self.I[0:-1]:
            self.m.Constraint_V_HP_HXH_HS_I.add(sum(self.m.B_T_HP_HXH_N_I[n,i] for n in self.N_MC) == 1)

        for i in self.I[0:-1]:
            for n in self.N_MC: 
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.V_HP_HXH_N_I[n,i]  <= self.V_HP_HXH_max * self.m.B_T_HP_HXH_N_I[n,i])
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.V_HP_HXH_N_I[n,i]  >= self.V_HP_HXH_min * self.m.B_T_HP_HXH_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:                
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HXH_N_I[n,i] <= self.T_HP_HXH_max_N[n] * self.m.B_T_HP_HXH_N_I[n,i])
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HXH_N_I[n,i] >= self.T_HP_HXH_min_N[n] * self.m.B_T_HP_HXH_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HXH_N_I[n,i] <= self.T_upper_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HXH_N_I[n,i] >= self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HXH_N_I[n,i] <= self.T_upper_MC * self.m.B_T_HP_HXH_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HXH_N_I[n,i] >= self.T_lower_MC * self.m.B_T_HP_HXH_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HXH_N_I[n,i] <= self.m.T_HP_HXH_N_I[n,i] - (1 - self.m.B_T_HP_HXH_N_I[n,i]) * self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HXH_N_I[n,i] >= self.m.T_HP_HXH_N_I[n,i] - (1 - self.m.B_T_HP_HXH_N_I[n,i]) * self.T_upper_MC) ## Big M constraint input

        for i in self.I[0:-(self.ControlPeriod3)]:
            if i%self.ControlPeriod3 == 0:
                for j in range(1,self.ControlPeriod3):
                    self.m.Constraint_V_HP_HXH_HS_I.add(self.m.V_HP_HXH_I[i] == self.m.V_HP_HXH_I[i+j])
        
        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.T_HP_HXH_N_I[n,i] == self.m.T_HP_HT_I[i+1] - self.m.T_HXH_w_I[i+1])

        for i in self.I[0:-1]:
            for h in self.H:
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HXH_H_I[h,i] <= self.W_upper_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HXH_H_I[h,i] >= self.W_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HXH_H_I[h,i] <= self.W_upper_MC * self.m.B_HP_H_I[h,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HXH_H_I[h,i] >= self.W_lower_MC * self.m.B_HP_H_I[h,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HXH_H_I[h,i] <= self.m.W_HP_HXH_I[i] - (1 - self.m.B_HP_H_I[h,i]) * self.W_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HXH_H_I[h,i] >= self.m.W_HP_HXH_I[i] - (1 - self.m.B_HP_H_I[h,i]) * self.W_upper_MC) ## Big M constraint input

        for i in self.I[0:-1]:
            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.W_HP_HS_I[i] >= sum(self.V_HP_HS_min * self.m.Z_HP_HS_N_I[n,i] + self.T_HP_HS_min_N[n] * self.m.V_HP_HS_N_I[n,i] - self.V_HP_HS_min * self.T_HP_HS_min_N[n] * self.m.B_T_HP_HS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.W_HP_HS_I[i] >= sum(self.V_HP_HS_max * self.m.Z_HP_HS_N_I[n,i] + self.T_HP_HS_max_N[n] * self.m.V_HP_HS_N_I[n,i] - self.V_HP_HS_max * self.T_HP_HS_max_N[n] * self.m.B_T_HP_HS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.W_HP_HS_I[i] <= sum(self.V_HP_HS_max * self.m.Z_HP_HS_N_I[n,i] + self.T_HP_HS_min_N[n] * self.m.V_HP_HS_N_I[n,i] - self.V_HP_HS_max * self.T_HP_HS_min_N[n] * self.m.B_T_HP_HS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.W_HP_HS_I[i] <= sum(self.V_HP_HS_min * self.m.Z_HP_HS_N_I[n,i] + self.T_HP_HS_max_N[n] * self.m.V_HP_HS_N_I[n,i] - self.V_HP_HS_min * self.T_HP_HS_max_N[n] * self.m.B_T_HP_HS_N_I[n,i] for n in self.N_MC))

            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.T_HP_HS_I[i] == sum(self.m.Z_HP_HS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.V_HP_HS_I[i] == sum(self.m.V_HP_HS_N_I[n,i] for n in self.N_MC))

        for i in self.I[0:-1]:
            self.m.Constraint_V_HP_HXH_HS_I.add(sum(self.m.B_T_HP_HS_N_I[n,i] for n in self.N_MC) == 1)

        for i in self.I[0:-1]:
            for n in self.N_MC: 
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.V_HP_HS_N_I[n,i]  <= self.V_HP_HS_max * self.m.B_T_HP_HS_N_I[n,i])
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.V_HP_HS_N_I[n,i]  >= self.V_HP_HS_min * self.m.B_T_HP_HS_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:                
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HS_N_I[n,i] <= self.T_HP_HS_max_N[n] * self.m.B_T_HP_HS_N_I[n,i])
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HS_N_I[n,i] >= self.T_HP_HS_min_N[n] * self.m.B_T_HP_HS_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HS_N_I[n,i] <= self.T_upper_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HS_N_I[n,i] >= self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HS_N_I[n,i] <= self.T_upper_MC * self.m.B_T_HP_HS_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HS_N_I[n,i] >= self.T_lower_MC * self.m.B_T_HP_HS_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HS_N_I[n,i] <= self.m.T_HP_HS_N_I[n,i] - (1 - self.m.B_T_HP_HS_N_I[n,i]) * self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HS_N_I[n,i] >= self.m.T_HP_HS_N_I[n,i] - (1 - self.m.B_T_HP_HS_N_I[n,i]) * self.T_upper_MC) ## Big M constraint input

        for i in self.I[0:-(self.ControlPeriod3)]:
            if i%self.ControlPeriod3 == 0:
                for j in range(1,self.ControlPeriod3):
                    self.m.Constraint_V_HP_HXH_HS_I.add(self.m.V_HP_HS_I[i] == self.m.V_HP_HS_I[i+j])
        
        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.T_HP_HS_N_I[n,i] == self.m.T_HP_HT_I[i+1] - self.m.T_HS_I[i+1])

        for i in self.I[0:-1]:
            for h in self.H:
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HS_H_I[h,i] <= self.W_upper_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HS_H_I[h,i] >= self.W_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HS_H_I[h,i] <= self.W_upper_MC * self.m.B_HP_H_I[h,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HS_H_I[h,i] >= self.W_lower_MC * self.m.B_HP_H_I[h,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HS_H_I[h,i] <= self.m.W_HP_HS_I[i] - (1 - self.m.B_HP_H_I[h,i]) * self.W_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HP_HS_H_I[h,i] >= self.m.W_HP_HS_I[i] - (1 - self.m.B_HP_H_I[h,i]) * self.W_upper_MC) ## Big M constraint input

        for i in self.I[0:-1]:
            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.V_HP_HXH_I[i] + self.m.V_HP_HS_I[i] == 1) ## Always 1 for both ways

        for i in self.I[0:-1]:
            for h in self.H:
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HXH_HXH_w_I[h,i] <= self.T_upper_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HXH_HXH_w_I[h,i] >= self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HXH_HXH_w_I[h,i] <= self.T_upper_MC * self.m.B_HP_H_I[h,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HXH_HXH_w_I[h,i] >= self.T_lower_MC * self.m.B_HP_H_I[h,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HXH_HXH_w_I[h,i] <= (self.m.T_HXH_w_I[i+1] - self.m.T_HXH_w_out_I[i]) - (1 - self.m.B_HP_H_I[h,i]) * self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HXH_HXH_w_I[h,i] >= (self.m.T_HXH_w_I[i+1] - self.m.T_HXH_w_out_I[i]) - (1 - self.m.B_HP_H_I[h,i]) * self.T_upper_MC) ## Big M constraint input

        for i in self.I[:-2]:
            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HXH_HS_I[i] >= self.m.V_HP_HXH_I[i] - self.m.V_HP_HXH_I[i+1])
            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HXH_HS_I[i] >= self.m.V_HP_HXH_I[i+1] - self.m.V_HP_HXH_I[i])
            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HXH_HS_I[i] >= self.m.V_HP_HS_I[i] - self.m.V_HP_HS_I[i+1])
            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HXH_HS_I[i] >= self.m.V_HP_HS_I[i+1] - self.m.V_HP_HS_I[i])
            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HXH_HS_I[i] <= 1)
        
        if self.End_Toggle_Constraints == True:
            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HXH_HS_I[self.I[-2]] >= self.m.V_HP_HXH_I[self.I[-2]] - self.V_HP_HXH_end)
            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HXH_HS_I[self.I[-2]] >= self.V_HP_HXH_end - self.m.V_HP_HXH_I[self.I[-2]])
            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HXH_HS_I[self.I[-2]] >= self.m.V_HP_HS_I[self.I[-2]] - self.V_HP_HS_end)
            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HXH_HS_I[self.I[-2]] >= self.V_HP_HS_end - self.m.V_HP_HS_I[self.I[-2]])
            self.m.Constraint_V_HP_HXH_HS_I.add(self.m.Z_HXH_HS_I[self.I[-2]] <= 1)

        ## V_HXA_HXH_HGC
        self.m.Constraint_V_HXA_HXH_HGC_I = pyo.ConstraintList()
        for i in self.I[0:-1]:
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.W_HXA_HXH_I[i] >= sum(self.V_HXA_HXH_min * self.m.Z_HXA_HXH_N_I[n,i] + self.T_HXA_HXH_min_N[n] * self.m.V_HXA_HXH_N_I[n,i] - self.V_HXA_HXH_min * self.T_HXA_HXH_min_N[n] * self.m.B_T_HXA_HXH_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.W_HXA_HXH_I[i] >= sum(self.V_HXA_HXH_max * self.m.Z_HXA_HXH_N_I[n,i] + self.T_HXA_HXH_max_N[n] * self.m.V_HXA_HXH_N_I[n,i] - self.V_HXA_HXH_max * self.T_HXA_HXH_max_N[n] * self.m.B_T_HXA_HXH_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.W_HXA_HXH_I[i] <= sum(self.V_HXA_HXH_max * self.m.Z_HXA_HXH_N_I[n,i] + self.T_HXA_HXH_min_N[n] * self.m.V_HXA_HXH_N_I[n,i] - self.V_HXA_HXH_max * self.T_HXA_HXH_min_N[n] * self.m.B_T_HXA_HXH_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.W_HXA_HXH_I[i] <= sum(self.V_HXA_HXH_min * self.m.Z_HXA_HXH_N_I[n,i] + self.T_HXA_HXH_max_N[n] * self.m.V_HXA_HXH_N_I[n,i] - self.V_HXA_HXH_min * self.T_HXA_HXH_max_N[n] * self.m.B_T_HXA_HXH_N_I[n,i] for n in self.N_MC))

            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.T_HXA_HXH_I[i] == sum(self.m.Z_HXA_HXH_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.V_HXA_HXH_I[i] == sum(self.m.V_HXA_HXH_N_I[n,i] for n in self.N_MC))

        for i in self.I[0:-1]:
            self.m.Constraint_V_HXA_HXH_HGC_I.add(sum(self.m.B_T_HXA_HXH_N_I[n,i] for n in self.N_MC) == 1)

        for i in self.I[0:-1]:
            for n in self.N_MC: 
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.V_HXA_HXH_N_I[n,i]  <= self.V_HXA_HXH_max * self.m.B_T_HXA_HXH_N_I[n,i])
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.V_HXA_HXH_N_I[n,i]  >= self.V_HXA_HXH_min * self.m.B_T_HXA_HXH_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:                
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXA_HXH_N_I[n,i] <= self.T_HXA_HXH_max_N[n] * self.m.B_T_HXA_HXH_N_I[n,i])
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXA_HXH_N_I[n,i] >= self.T_HXA_HXH_min_N[n] * self.m.B_T_HXA_HXH_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXA_HXH_N_I[n,i] <= self.T_upper_MC) ## Big M constraint input
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXA_HXH_N_I[n,i] >= self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXA_HXH_N_I[n,i] <= self.T_upper_MC * self.m.B_T_HXA_HXH_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXA_HXH_N_I[n,i] >= self.T_lower_MC * self.m.B_T_HXA_HXH_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXA_HXH_N_I[n,i] <= self.m.T_HXA_HXH_N_I[n,i] - (1 - self.m.B_T_HXA_HXH_N_I[n,i]) * self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXA_HXH_N_I[n,i] >= self.m.T_HXA_HXH_N_I[n,i] - (1 - self.m.B_T_HXA_HXH_N_I[n,i]) * self.T_upper_MC) ## Big M constraint input

        for i in self.I[0:-(self.ControlPeriod3)]:
            if i%self.ControlPeriod3 == 0:
                for j in range(1,self.ControlPeriod3):
                    self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.V_HXA_HXH_I[i] == self.m.V_HXA_HXH_I[i+j])
        
        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.T_HXA_HXH_N_I[n,i] == self.m.T_HXA_I[i+1] - self.m.T_HXH_b_I[i+1])

        for i in self.I[0:-1]:
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.W_HXA_HGC_I[i] >= sum(self.V_HXA_HGC_min * self.m.Z_HXA_HGC_N_I[n,i] + self.T_HXA_HGC_min_N[n] * self.m.V_HXA_HGC_N_I[n,i] - self.V_HXA_HGC_min * self.T_HXA_HGC_min_N[n] * self.m.B_T_HXA_HGC_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.W_HXA_HGC_I[i] >= sum(self.V_HXA_HGC_max * self.m.Z_HXA_HGC_N_I[n,i] + self.T_HXA_HGC_max_N[n] * self.m.V_HXA_HGC_N_I[n,i] - self.V_HXA_HGC_max * self.T_HXA_HGC_max_N[n] * self.m.B_T_HXA_HGC_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.W_HXA_HGC_I[i] <= sum(self.V_HXA_HGC_max * self.m.Z_HXA_HGC_N_I[n,i] + self.T_HXA_HGC_min_N[n] * self.m.V_HXA_HGC_N_I[n,i] - self.V_HXA_HGC_max * self.T_HXA_HGC_min_N[n] * self.m.B_T_HXA_HGC_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.W_HXA_HGC_I[i] <= sum(self.V_HXA_HGC_min * self.m.Z_HXA_HGC_N_I[n,i] + self.T_HXA_HGC_max_N[n] * self.m.V_HXA_HGC_N_I[n,i] - self.V_HXA_HGC_min * self.T_HXA_HGC_max_N[n] * self.m.B_T_HXA_HGC_N_I[n,i] for n in self.N_MC))

            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.T_HXA_HGC_I[i] == sum(self.m.Z_HXA_HGC_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.V_HXA_HGC_I[i] == sum(self.m.V_HXA_HGC_N_I[n,i] for n in self.N_MC))

        for i in self.I[0:-1]:
            self.m.Constraint_V_HXA_HXH_HGC_I.add(sum(self.m.B_T_HXA_HGC_N_I[n,i] for n in self.N_MC) == 1)

        for i in self.I[0:-1]:
            for n in self.N_MC: 
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.V_HXA_HGC_N_I[n,i]  <= self.V_HXA_HGC_max * self.m.B_T_HXA_HGC_N_I[n,i])
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.V_HXA_HGC_N_I[n,i]  >= self.V_HXA_HGC_min * self.m.B_T_HXA_HGC_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:                
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXA_HGC_N_I[n,i] <= self.T_HXA_HGC_max_N[n] * self.m.B_T_HXA_HGC_N_I[n,i])
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXA_HGC_N_I[n,i] >= self.T_HXA_HGC_min_N[n] * self.m.B_T_HXA_HGC_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXA_HGC_N_I[n,i] <= self.T_upper_MC) ## Big M constraint input
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXA_HGC_N_I[n,i] >= self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXA_HGC_N_I[n,i] <= self.T_upper_MC * self.m.B_T_HXA_HGC_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXA_HGC_N_I[n,i] >= self.T_lower_MC * self.m.B_T_HXA_HGC_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXA_HGC_N_I[n,i] <= self.m.T_HXA_HGC_N_I[n,i] - (1 - self.m.B_T_HXA_HGC_N_I[n,i]) * self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXA_HGC_N_I[n,i] >= self.m.T_HXA_HGC_N_I[n,i] - (1 - self.m.B_T_HXA_HGC_N_I[n,i]) * self.T_upper_MC) ## Big M constraint input

        for i in self.I[0:-(self.ControlPeriod3)]:
            if i%self.ControlPeriod3 == 0:
                for j in range(1,self.ControlPeriod3):
                    self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.V_HXA_HGC_I[i] == self.m.V_HXA_HGC_I[i+j])
        
        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.T_HXA_HGC_N_I[n,i] == self.m.T_HXA_I[i+1] - self.m.T_HGC_I[i+1])

        for i in self.I[0:-1]:
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.V_HXA_HXH_I[i] + self.m.V_HXA_HGC_I[i] <= 1) ## Maximum of 1 for both ways, since pump independent on HXA or HP

        for i in self.I[:-2]:
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXH_HGC_I[i] >= self.m.V_HXA_HXH_I[i] - self.m.V_HXA_HXH_I[i+1])
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXH_HGC_I[i] >= self.m.V_HXA_HXH_I[i+1] - self.m.V_HXA_HXH_I[i])
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXH_HGC_I[i] >= self.m.V_HXA_HGC_I[i] - self.m.V_HXA_HGC_I[i+1])
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXH_HGC_I[i] >= self.m.V_HXA_HGC_I[i+1] - self.m.V_HXA_HGC_I[i])
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXH_HGC_I[i] <= 1)
        
        if self.End_Toggle_Constraints == True:
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXH_HGC_I[self.I[-2]] >= self.m.V_HXA_HXH_I[self.I[-2]] - self.V_HXA_HXH_end)
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXH_HGC_I[self.I[-2]] >= self.V_HXA_HXH_end - self.m.V_HXA_HXH_I[self.I[-2]])
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXH_HGC_I[self.I[-2]] >= self.m.V_HXA_HGC_I[self.I[-2]] - self.V_HXA_HGC_end)
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXH_HGC_I[self.I[-2]] >= self.V_HXA_HGC_end - self.m.V_HXA_HGC_I[self.I[-2]])
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXH_HGC_I[self.I[-2]] <= 1)

        for i in self.I[:-1]: 
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXH_HXH_b_I[i] <= self.T_upper_MC) ## Big M constraint input
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXH_HXH_b_I[i] >= self.T_lower_MC) ## Big M constraint input
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXH_HXH_b_I[i] <= self.T_upper_MC * sum(self.m.B_HP_H_I[h,i] * self.mdot_digit_HP_H[h] for h in self.H)) ## Big M constraint input
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXH_HXH_b_I[i] >= self.T_lower_MC * sum(self.m.B_HP_H_I[h,i] * self.mdot_digit_HP_H[h] for h in self.H)) ## Big M constraint input
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXH_HXH_b_I[i] <= (self.m.T_HXH_b_I[i+1] - self.m.T_HXH_b_out_I[i]) - (1 - sum(self.m.B_HP_H_I[h,i] * self.mdot_digit_HP_H[h] for h in self.H)) * self.T_lower_MC) ## Big M constraint input
            self.m.Constraint_V_HXA_HXH_HGC_I.add(self.m.Z_HXH_HXH_b_I[i] >= (self.m.T_HXH_b_I[i+1] - self.m.T_HXH_b_out_I[i]) - (1 - sum(self.m.B_HP_H_I[h,i] * self.mdot_digit_HP_H[h] for h in self.H)) * self.T_upper_MC) ## Big M constraint input

        ## VP_HS_IS 
        self.m.Constraint_VP_HS_IS_I = pyo.ConstraintList()
        for i in self.I[0:-1]:
            self.m.Constraint_VP_HS_IS_I.add(self.m.W_HS_IS_I_2[i] >= sum(self.V_HS_IS_min * self.m.Z_HS_IS_N_I_2[n,i] + self.T_HS_IS_min_N[n] * self.m.V_HS_IS_N_I_2[n,i] - self.V_HS_IS_min * self.T_HS_IS_min_N[n] * self.m.B_T_HS_IS_N_I_2[n,i] for n in self.N_MC))
            self.m.Constraint_VP_HS_IS_I.add(self.m.W_HS_IS_I_2[i] >= sum(self.V_HS_IS_max * self.m.Z_HS_IS_N_I_2[n,i] + self.T_HS_IS_max_N[n] * self.m.V_HS_IS_N_I_2[n,i] - self.V_HS_IS_max * self.T_HS_IS_max_N[n] * self.m.B_T_HS_IS_N_I_2[n,i] for n in self.N_MC))
            self.m.Constraint_VP_HS_IS_I.add(self.m.W_HS_IS_I_2[i] <= sum(self.V_HS_IS_max * self.m.Z_HS_IS_N_I_2[n,i] + self.T_HS_IS_min_N[n] * self.m.V_HS_IS_N_I_2[n,i] - self.V_HS_IS_max * self.T_HS_IS_min_N[n] * self.m.B_T_HS_IS_N_I_2[n,i] for n in self.N_MC))
            self.m.Constraint_VP_HS_IS_I.add(self.m.W_HS_IS_I_2[i] <= sum(self.V_HS_IS_min * self.m.Z_HS_IS_N_I_2[n,i] + self.T_HS_IS_max_N[n] * self.m.V_HS_IS_N_I_2[n,i] - self.V_HS_IS_min * self.T_HS_IS_max_N[n] * self.m.B_T_HS_IS_N_I_2[n,i] for n in self.N_MC))

            self.m.Constraint_VP_HS_IS_I.add(self.m.T_HS_IS_I_2[i] == sum(self.m.Z_HS_IS_N_I_2[n,i] for n in self.N_MC))
            self.m.Constraint_VP_HS_IS_I.add(self.m.V_HS_IS_I_2[i] == sum(self.m.V_HS_IS_N_I_2[n,i] for n in self.N_MC))

        for i in self.I[0:-1]:
            self.m.Constraint_VP_HS_IS_I.add(sum(self.m.B_T_HS_IS_N_I_2[n,i] for n in self.N_MC) == 1)

        for i in self.I[0:-1]:
            for n in self.N_MC: 
                self.m.Constraint_VP_HS_IS_I.add(self.m.V_HS_IS_N_I_2[n,i]  <= self.V_HS_IS_max * self.m.B_T_HS_IS_N_I_2[n,i])
                self.m.Constraint_VP_HS_IS_I.add(self.m.V_HS_IS_N_I_2[n,i]  >= self.V_HS_IS_min * self.m.B_T_HS_IS_N_I_2[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:                
                self.m.Constraint_VP_HS_IS_I.add(self.m.Z_HS_IS_N_I_2[n,i] <= self.T_HS_IS_max_N[n] * self.m.B_T_HS_IS_N_I_2[n,i])
                self.m.Constraint_VP_HS_IS_I.add(self.m.Z_HS_IS_N_I_2[n,i] >= self.T_HS_IS_min_N[n] * self.m.B_T_HS_IS_N_I_2[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_VP_HS_IS_I.add(self.m.Z_HS_IS_N_I_2[n,i] <= self.T_upper_MC) ## Big M constraint input
                self.m.Constraint_VP_HS_IS_I.add(self.m.Z_HS_IS_N_I_2[n,i] >= self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_VP_HS_IS_I.add(self.m.Z_HS_IS_N_I_2[n,i] <= self.T_upper_MC * self.m.B_T_HS_IS_N_I_2[n,i]) ## Big M constraint input
                self.m.Constraint_VP_HS_IS_I.add(self.m.Z_HS_IS_N_I_2[n,i] >= self.T_lower_MC * self.m.B_T_HS_IS_N_I_2[n,i]) ## Big M constraint input
                self.m.Constraint_VP_HS_IS_I.add(self.m.Z_HS_IS_N_I_2[n,i] <= self.m.T_HS_IS_N_I_2[n,i] - (1 - self.m.B_T_HS_IS_N_I_2[n,i]) * self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_VP_HS_IS_I.add(self.m.Z_HS_IS_N_I_2[n,i] >= self.m.T_HS_IS_N_I_2[n,i] - (1 - self.m.B_T_HS_IS_N_I_2[n,i]) * self.T_upper_MC) ## Big M constraint input

        for i in self.I[0:-(self.ControlPeriod3)]:
            if i%self.ControlPeriod3 == 0:
                for j in range(1,self.ControlPeriod3):
                    self.m.Constraint_VP_HS_IS_I.add(self.m.V_HS_IS_I_2[i] == self.m.V_HS_IS_I_2[i+j])
        
        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_VP_HS_IS_I.add(self.m.T_HS_IS_N_I_2[n,i] == self.m.T_IS_W_I_WR[i+1,0] - self.m.T_HS_I[i+1])

        for i in self.I[0:-1]:
            self.m.Constraint_VP_HS_IS_I.add(self.m.W_HS_IS_I[i] >= sum(self.V_HS_IS_min * self.m.Z_HS_IS_N_I[n,i] + self.T_HS_IS_min_N[n] * self.m.V_HS_IS_N_I[n,i] - self.V_HS_IS_min * self.T_HS_IS_min_N[n] * self.m.B_T_HS_IS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_VP_HS_IS_I.add(self.m.W_HS_IS_I[i] >= sum(self.V_HS_IS_max * self.m.Z_HS_IS_N_I[n,i] + self.T_HS_IS_max_N[n] * self.m.V_HS_IS_N_I[n,i] - self.V_HS_IS_max * self.T_HS_IS_max_N[n] * self.m.B_T_HS_IS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_VP_HS_IS_I.add(self.m.W_HS_IS_I[i] <= sum(self.V_HS_IS_max * self.m.Z_HS_IS_N_I[n,i] + self.T_HS_IS_min_N[n] * self.m.V_HS_IS_N_I[n,i] - self.V_HS_IS_max * self.T_HS_IS_min_N[n] * self.m.B_T_HS_IS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_VP_HS_IS_I.add(self.m.W_HS_IS_I[i] <= sum(self.V_HS_IS_min * self.m.Z_HS_IS_N_I[n,i] + self.T_HS_IS_max_N[n] * self.m.V_HS_IS_N_I[n,i] - self.V_HS_IS_min * self.T_HS_IS_max_N[n] * self.m.B_T_HS_IS_N_I[n,i] for n in self.N_MC))

            self.m.Constraint_VP_HS_IS_I.add(self.m.T_HS_IS_I[i] == sum(self.m.Z_HS_IS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_VP_HS_IS_I.add(self.m.V_HS_IS_I[i] == sum(self.m.V_HS_IS_N_I[n,i] for n in self.N_MC))

        for i in self.I[0:-1]:
            self.m.Constraint_VP_HS_IS_I.add(sum(self.m.B_T_HS_IS_N_I[n,i] for n in self.N_MC) == 1)

        for i in self.I[0:-1]:
            for n in self.N_MC: 
                self.m.Constraint_VP_HS_IS_I.add(self.m.V_HS_IS_N_I[n,i]  <= self.V_HS_IS_max * self.m.B_T_HS_IS_N_I[n,i])
                self.m.Constraint_VP_HS_IS_I.add(self.m.V_HS_IS_N_I[n,i]  >= self.V_HS_IS_min * self.m.B_T_HS_IS_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:                
                self.m.Constraint_VP_HS_IS_I.add(self.m.Z_HS_IS_N_I[n,i] <= self.T_HS_IS_max_N[n] * self.m.B_T_HS_IS_N_I[n,i])
                self.m.Constraint_VP_HS_IS_I.add(self.m.Z_HS_IS_N_I[n,i] >= self.T_HS_IS_min_N[n] * self.m.B_T_HS_IS_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_VP_HS_IS_I.add(self.m.Z_HS_IS_N_I[n,i] <= self.T_upper_MC) ## Big M constraint input
                self.m.Constraint_VP_HS_IS_I.add(self.m.Z_HS_IS_N_I[n,i] >= self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_VP_HS_IS_I.add(self.m.Z_HS_IS_N_I[n,i] <= self.T_upper_MC * self.m.B_T_HS_IS_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_VP_HS_IS_I.add(self.m.Z_HS_IS_N_I[n,i] >= self.T_lower_MC * self.m.B_T_HS_IS_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_VP_HS_IS_I.add(self.m.Z_HS_IS_N_I[n,i] <= self.m.T_HS_IS_N_I[n,i] - (1 - self.m.B_T_HS_IS_N_I[n,i]) * self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_VP_HS_IS_I.add(self.m.Z_HS_IS_N_I[n,i] >= self.m.T_HS_IS_N_I[n,i] - (1 - self.m.B_T_HS_IS_N_I[n,i]) * self.T_upper_MC) ## Big M constraint input

        for i in self.I[0:-(self.ControlPeriod3)]:
            if i%self.ControlPeriod3 == 0:
                for j in range(1,self.ControlPeriod3):
                    self.m.Constraint_VP_HS_IS_I.add(self.m.V_HS_IS_I[i] == self.m.V_HS_IS_I[i+j])
        
        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_VP_HS_IS_I.add(self.m.T_HS_IS_N_I[n,i] == self.m.T_HS_I[i+1] - self.m.T_IS_I[i+1])

        for i in self.I[0:-1]:
            self.m.Constraint_VP_HS_IS_I.add(self.m.V_HS_IS_I[i] == self.m.V_HS_IS_I_2[i])
        ## VP_IS_HGS
        self.m.Constraint_VP_IS_HGS_I = pyo.ConstraintList()
        for i in self.I[0:-1]:
            self.m.Constraint_VP_IS_HGS_I.add(self.m.W_IS_HGS_I_2[i] >= sum(self.V_IS_HGS_min * self.m.Z_IS_HGS_N_I_2[n,i] + self.T_IS_HGS_min_N[n] * self.m.V_IS_HGS_N_I_2[n,i] - self.V_IS_HGS_min * self.T_IS_HGS_min_N[n] * self.m.B_T_IS_HGS_N_I_2[n,i] for n in self.N_MC))
            self.m.Constraint_VP_IS_HGS_I.add(self.m.W_IS_HGS_I_2[i] >= sum(self.V_IS_HGS_max * self.m.Z_IS_HGS_N_I_2[n,i] + self.T_IS_HGS_max_N[n] * self.m.V_IS_HGS_N_I_2[n,i] - self.V_IS_HGS_max * self.T_IS_HGS_max_N[n] * self.m.B_T_IS_HGS_N_I_2[n,i] for n in self.N_MC))
            self.m.Constraint_VP_IS_HGS_I.add(self.m.W_IS_HGS_I_2[i] <= sum(self.V_IS_HGS_max * self.m.Z_IS_HGS_N_I_2[n,i] + self.T_IS_HGS_min_N[n] * self.m.V_IS_HGS_N_I_2[n,i] - self.V_IS_HGS_max * self.T_IS_HGS_min_N[n] * self.m.B_T_IS_HGS_N_I_2[n,i] for n in self.N_MC))
            self.m.Constraint_VP_IS_HGS_I.add(self.m.W_IS_HGS_I_2[i] <= sum(self.V_IS_HGS_min * self.m.Z_IS_HGS_N_I_2[n,i] + self.T_IS_HGS_max_N[n] * self.m.V_IS_HGS_N_I_2[n,i] - self.V_IS_HGS_min * self.T_IS_HGS_max_N[n] * self.m.B_T_IS_HGS_N_I_2[n,i] for n in self.N_MC))

            self.m.Constraint_VP_IS_HGS_I.add(self.m.T_IS_HGS_I_2[i] == sum(self.m.Z_IS_HGS_N_I_2[n,i] for n in self.N_MC))
            self.m.Constraint_VP_IS_HGS_I.add(self.m.V_IS_HGS_I_2[i] == sum(self.m.V_IS_HGS_N_I_2[n,i] for n in self.N_MC))

        for i in self.I[0:-1]:
            self.m.Constraint_VP_IS_HGS_I.add(sum(self.m.B_T_IS_HGS_N_I_2[n,i] for n in self.N_MC) == 1)

        for i in self.I[0:-1]:
            for n in self.N_MC: 
                self.m.Constraint_VP_IS_HGS_I.add(self.m.V_IS_HGS_N_I_2[n,i]  <= self.V_IS_HGS_max * self.m.B_T_IS_HGS_N_I_2[n,i])
                self.m.Constraint_VP_IS_HGS_I.add(self.m.V_IS_HGS_N_I_2[n,i]  >= self.V_IS_HGS_min * self.m.B_T_IS_HGS_N_I_2[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:                
                self.m.Constraint_VP_IS_HGS_I.add(self.m.Z_IS_HGS_N_I_2[n,i] <= self.T_IS_HGS_max_N[n] * self.m.B_T_IS_HGS_N_I_2[n,i])
                self.m.Constraint_VP_IS_HGS_I.add(self.m.Z_IS_HGS_N_I_2[n,i] >= self.T_IS_HGS_min_N[n] * self.m.B_T_IS_HGS_N_I_2[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_VP_IS_HGS_I.add(self.m.Z_IS_HGS_N_I_2[n,i] <= self.T_upper_MC) ## Big M constraint input
                self.m.Constraint_VP_IS_HGS_I.add(self.m.Z_IS_HGS_N_I_2[n,i] >= self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_VP_IS_HGS_I.add(self.m.Z_IS_HGS_N_I_2[n,i] <= self.T_upper_MC * self.m.B_T_IS_HGS_N_I_2[n,i]) ## Big M constraint input
                self.m.Constraint_VP_IS_HGS_I.add(self.m.Z_IS_HGS_N_I_2[n,i] >= self.T_lower_MC * self.m.B_T_IS_HGS_N_I_2[n,i]) ## Big M constraint input
                self.m.Constraint_VP_IS_HGS_I.add(self.m.Z_IS_HGS_N_I_2[n,i] <= self.m.T_IS_HGS_N_I_2[n,i] - (1 - self.m.B_T_IS_HGS_N_I_2[n,i]) * self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_VP_IS_HGS_I.add(self.m.Z_IS_HGS_N_I_2[n,i] >= self.m.T_IS_HGS_N_I_2[n,i] - (1 - self.m.B_T_IS_HGS_N_I_2[n,i]) * self.T_upper_MC) ## Big M constraint input

        for i in self.I[0:-(self.ControlPeriod3)]:
            if i%self.ControlPeriod3 == 0:
                for j in range(1,self.ControlPeriod3):
                    self.m.Constraint_VP_IS_HGS_I.add(self.m.V_IS_HGS_I_2[i] == self.m.V_IS_HGS_I_2[i+j])
        
        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_VP_IS_HGS_I.add(self.m.T_IS_HGS_N_I_2[n,i] == self.m.T_IS_W_I_WR[i+1,0] - self.m.T_HGS_I[i+1])

        for i in self.I[0:-1]:
            self.m.Constraint_VP_IS_HGS_I.add(self.m.W_IS_HGS_I[i] >= sum(self.V_IS_HGS_min * self.m.Z_IS_HGS_N_I[n,i] + self.T_IS_HGS_min_N[n] * self.m.V_IS_HGS_N_I[n,i] - self.V_IS_HGS_min * self.T_IS_HGS_min_N[n] * self.m.B_T_IS_HGS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_VP_IS_HGS_I.add(self.m.W_IS_HGS_I[i] >= sum(self.V_IS_HGS_max * self.m.Z_IS_HGS_N_I[n,i] + self.T_IS_HGS_max_N[n] * self.m.V_IS_HGS_N_I[n,i] - self.V_IS_HGS_max * self.T_IS_HGS_max_N[n] * self.m.B_T_IS_HGS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_VP_IS_HGS_I.add(self.m.W_IS_HGS_I[i] <= sum(self.V_IS_HGS_max * self.m.Z_IS_HGS_N_I[n,i] + self.T_IS_HGS_min_N[n] * self.m.V_IS_HGS_N_I[n,i] - self.V_IS_HGS_max * self.T_IS_HGS_min_N[n] * self.m.B_T_IS_HGS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_VP_IS_HGS_I.add(self.m.W_IS_HGS_I[i] <= sum(self.V_IS_HGS_min * self.m.Z_IS_HGS_N_I[n,i] + self.T_IS_HGS_max_N[n] * self.m.V_IS_HGS_N_I[n,i] - self.V_IS_HGS_min * self.T_IS_HGS_max_N[n] * self.m.B_T_IS_HGS_N_I[n,i] for n in self.N_MC))

            self.m.Constraint_VP_IS_HGS_I.add(self.m.T_IS_HGS_I[i] == sum(self.m.Z_IS_HGS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_VP_IS_HGS_I.add(self.m.V_IS_HGS_I[i] == sum(self.m.V_IS_HGS_N_I[n,i] for n in self.N_MC))

        for i in self.I[0:-1]:
            self.m.Constraint_VP_IS_HGS_I.add(sum(self.m.B_T_IS_HGS_N_I[n,i] for n in self.N_MC) == 1)

        for i in self.I[0:-1]:
            for n in self.N_MC: 
                self.m.Constraint_VP_IS_HGS_I.add(self.m.V_IS_HGS_N_I[n,i]  <= self.V_IS_HGS_max * self.m.B_T_IS_HGS_N_I[n,i])
                self.m.Constraint_VP_IS_HGS_I.add(self.m.V_IS_HGS_N_I[n,i]  >= self.V_IS_HGS_min * self.m.B_T_IS_HGS_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:                
                self.m.Constraint_VP_IS_HGS_I.add(self.m.Z_IS_HGS_N_I[n,i] <= self.T_IS_HGS_max_N[n] * self.m.B_T_IS_HGS_N_I[n,i])
                self.m.Constraint_VP_IS_HGS_I.add(self.m.Z_IS_HGS_N_I[n,i] >= self.T_IS_HGS_min_N[n] * self.m.B_T_IS_HGS_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_VP_IS_HGS_I.add(self.m.Z_IS_HGS_N_I[n,i] <= self.T_upper_MC) ## Big M constraint input
                self.m.Constraint_VP_IS_HGS_I.add(self.m.Z_IS_HGS_N_I[n,i] >= self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_VP_IS_HGS_I.add(self.m.Z_IS_HGS_N_I[n,i] <= self.T_upper_MC * self.m.B_T_IS_HGS_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_VP_IS_HGS_I.add(self.m.Z_IS_HGS_N_I[n,i] >= self.T_lower_MC * self.m.B_T_IS_HGS_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_VP_IS_HGS_I.add(self.m.Z_IS_HGS_N_I[n,i] <= self.m.T_IS_HGS_N_I[n,i] - (1 - self.m.B_T_IS_HGS_N_I[n,i]) * self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_VP_IS_HGS_I.add(self.m.Z_IS_HGS_N_I[n,i] >= self.m.T_IS_HGS_N_I[n,i] - (1 - self.m.B_T_IS_HGS_N_I[n,i]) * self.T_upper_MC) ## Big M constraint input

        for i in self.I[0:-(self.ControlPeriod3)]:
            if i%self.ControlPeriod3 == 0:
                for j in range(1,self.ControlPeriod3):
                    self.m.Constraint_VP_IS_HGS_I.add(self.m.V_IS_HGS_I[i] == self.m.V_IS_HGS_I[i+j])
        
        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_VP_IS_HGS_I.add(self.m.T_IS_HGS_N_I[n,i] == self.m.T_IS_I[i+1] - self.m.T_HGS_I[i+1])

        for i in self.I[0:-1]:
            self.m.Constraint_VP_IS_HGS_I.add(self.m.V_IS_HGS_I[i] == self.m.V_IS_HGS_I_2[i])
        ## V_HP_HGC_HGCHXC
        self.m.Constraint_V_HP_HGC_HGCHXC_I = pyo.ConstraintList()
        for i in self.I[0:-1]:
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.W_HP_HGC_I[i] >= sum(self.V_HP_HGC_min * self.m.Z_HP_HGC_N_I[n,i] + self.T_HP_HGC_min_N[n] * self.m.V_HP_HGC_N_I[n,i] - self.V_HP_HGC_min * self.T_HP_HGC_min_N[n] * self.m.B_T_HP_HGC_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.W_HP_HGC_I[i] >= sum(self.V_HP_HGC_max * self.m.Z_HP_HGC_N_I[n,i] + self.T_HP_HGC_max_N[n] * self.m.V_HP_HGC_N_I[n,i] - self.V_HP_HGC_max * self.T_HP_HGC_max_N[n] * self.m.B_T_HP_HGC_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.W_HP_HGC_I[i] <= sum(self.V_HP_HGC_max * self.m.Z_HP_HGC_N_I[n,i] + self.T_HP_HGC_min_N[n] * self.m.V_HP_HGC_N_I[n,i] - self.V_HP_HGC_max * self.T_HP_HGC_min_N[n] * self.m.B_T_HP_HGC_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.W_HP_HGC_I[i] <= sum(self.V_HP_HGC_min * self.m.Z_HP_HGC_N_I[n,i] + self.T_HP_HGC_max_N[n] * self.m.V_HP_HGC_N_I[n,i] - self.V_HP_HGC_min * self.T_HP_HGC_max_N[n] * self.m.B_T_HP_HGC_N_I[n,i] for n in self.N_MC))

            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.T_HP_HGC_I[i] == sum(self.m.Z_HP_HGC_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.V_HP_HGC_I[i] == sum(self.m.V_HP_HGC_N_I[n,i] for n in self.N_MC))

        for i in self.I[0:-1]:
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(sum(self.m.B_T_HP_HGC_N_I[n,i] for n in self.N_MC) == 1)

        for i in self.I[0:-1]:
            for n in self.N_MC: 
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.V_HP_HGC_N_I[n,i]  <= self.V_HP_HGC_max * self.m.B_T_HP_HGC_N_I[n,i])
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.V_HP_HGC_N_I[n,i]  >= self.V_HP_HGC_min * self.m.B_T_HP_HGC_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:                
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGC_N_I[n,i] <= self.T_HP_HGC_max_N[n] * self.m.B_T_HP_HGC_N_I[n,i])
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGC_N_I[n,i] >= self.T_HP_HGC_min_N[n] * self.m.B_T_HP_HGC_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGC_N_I[n,i] <= self.T_upper_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGC_N_I[n,i] >= self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGC_N_I[n,i] <= self.T_upper_MC * self.m.B_T_HP_HGC_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGC_N_I[n,i] >= self.T_lower_MC * self.m.B_T_HP_HGC_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGC_N_I[n,i] <= self.m.T_HP_HGC_N_I[n,i] - (1 - self.m.B_T_HP_HGC_N_I[n,i]) * self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGC_N_I[n,i] >= self.m.T_HP_HGC_N_I[n,i] - (1 - self.m.B_T_HP_HGC_N_I[n,i]) * self.T_upper_MC) ## Big M constraint input

        for i in self.I[0:-(self.ControlPeriod3)]:
            if i%self.ControlPeriod3 == 0:
                for j in range(1,self.ControlPeriod3):
                    self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.V_HP_HGC_I[i] == self.m.V_HP_HGC_I[i+j])
        
        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.T_HP_HGC_N_I[n,i] == self.m.T_HP_LT_I[i+1] - self.m.T_HGC_I[i+1])

        for i in self.I[0:-1]:
            for h in self.H:
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGC_H_I[h,i] <= self.W_upper_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGC_H_I[h,i] >= self.W_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGC_H_I[h,i] <= self.W_upper_MC * self.m.B_HP_H_I[h,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGC_H_I[h,i] >= self.W_lower_MC * self.m.B_HP_H_I[h,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGC_H_I[h,i] <= self.m.W_HP_HGC_I[i] - (1 - self.m.B_HP_H_I[h,i]) * self.W_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGC_H_I[h,i] >= self.m.W_HP_HGC_I[i] - (1 - self.m.B_HP_H_I[h,i]) * self.W_upper_MC) ## Big M constraint input

        for i in self.I[0:-1]:
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.W_HP_HGCHXC_I[i] >= sum(self.V_HP_HGCHXC_min * self.m.Z_HP_HGCHXC_N_I[n,i] + self.T_HP_HGCHXC_min_N[n] * self.m.V_HP_HGCHXC_N_I[n,i] - self.V_HP_HGCHXC_min * self.T_HP_HGCHXC_min_N[n] * self.m.B_T_HP_HGCHXC_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.W_HP_HGCHXC_I[i] >= sum(self.V_HP_HGCHXC_max * self.m.Z_HP_HGCHXC_N_I[n,i] + self.T_HP_HGCHXC_max_N[n] * self.m.V_HP_HGCHXC_N_I[n,i] - self.V_HP_HGCHXC_max * self.T_HP_HGCHXC_max_N[n] * self.m.B_T_HP_HGCHXC_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.W_HP_HGCHXC_I[i] <= sum(self.V_HP_HGCHXC_max * self.m.Z_HP_HGCHXC_N_I[n,i] + self.T_HP_HGCHXC_min_N[n] * self.m.V_HP_HGCHXC_N_I[n,i] - self.V_HP_HGCHXC_max * self.T_HP_HGCHXC_min_N[n] * self.m.B_T_HP_HGCHXC_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.W_HP_HGCHXC_I[i] <= sum(self.V_HP_HGCHXC_min * self.m.Z_HP_HGCHXC_N_I[n,i] + self.T_HP_HGCHXC_max_N[n] * self.m.V_HP_HGCHXC_N_I[n,i] - self.V_HP_HGCHXC_min * self.T_HP_HGCHXC_max_N[n] * self.m.B_T_HP_HGCHXC_N_I[n,i] for n in self.N_MC))

            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.T_HP_HGCHXC_I[i] == sum(self.m.Z_HP_HGCHXC_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.V_HP_HGCHXC_I[i] == sum(self.m.V_HP_HGCHXC_N_I[n,i] for n in self.N_MC))

        for i in self.I[0:-1]:
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(sum(self.m.B_T_HP_HGCHXC_N_I[n,i] for n in self.N_MC) == 1)

        for i in self.I[0:-1]:
            for n in self.N_MC: 
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.V_HP_HGCHXC_N_I[n,i]  <= self.V_HP_HGCHXC_max * self.m.B_T_HP_HGCHXC_N_I[n,i])
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.V_HP_HGCHXC_N_I[n,i]  >= self.V_HP_HGCHXC_min * self.m.B_T_HP_HGCHXC_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:                
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_N_I[n,i] <= self.T_HP_HGCHXC_max_N[n] * self.m.B_T_HP_HGCHXC_N_I[n,i])
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_N_I[n,i] >= self.T_HP_HGCHXC_min_N[n] * self.m.B_T_HP_HGCHXC_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_N_I[n,i] <= self.T_upper_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_N_I[n,i] >= self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_N_I[n,i] <= self.T_upper_MC * self.m.B_T_HP_HGCHXC_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_N_I[n,i] >= self.T_lower_MC * self.m.B_T_HP_HGCHXC_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_N_I[n,i] <= self.m.T_HP_HGCHXC_N_I[n,i] - (1 - self.m.B_T_HP_HGCHXC_N_I[n,i]) * self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_N_I[n,i] >= self.m.T_HP_HGCHXC_N_I[n,i] - (1 - self.m.B_T_HP_HGCHXC_N_I[n,i]) * self.T_upper_MC) ## Big M constraint input

        for i in self.I[0:-(self.ControlPeriod3)]:
            if i%self.ControlPeriod3 == 0:
                for j in range(1,self.ControlPeriod3):
                    self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.V_HP_HGCHXC_I[i] == self.m.V_HP_HGCHXC_I[i+j])
        
        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.T_HP_HGCHXC_N_I[n,i] == self.m.T_HXC_b_I[i+1] - self.m.T_HGC_I[i+1])

        for i in self.I[0:-1]:
            for h in self.H:
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_H_I[h,i] <= self.W_upper_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_H_I[h,i] >= self.W_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_H_I[h,i] <= self.W_upper_MC * self.m.B_HP_H_I[h,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_H_I[h,i] >= self.W_lower_MC * self.m.B_HP_H_I[h,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_H_I[h,i] <= self.m.W_HP_HGCHXC_I[i] - (1 - self.m.B_HP_H_I[h,i]) * self.W_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_H_I[h,i] >= self.m.W_HP_HGCHXC_I[i] - (1 - self.m.B_HP_H_I[h,i]) * self.W_upper_MC) ## Big M constraint input
        
        for i in self.I[0:-1]:
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.W_HP_HGCHXC_2_I[i] >= sum(self.V_HP_HGCHXC_min * self.m.Z_HP_HGCHXC_N_2_I[n,i] + self.T_HP_HGCHXC_min_N[n] * self.m.V_HP_HGCHXC_N_2_I[n,i] - self.V_HP_HGCHXC_min * self.T_HP_HGCHXC_min_N[n] * self.m.B_T_HP_HGCHXC_N_2_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.W_HP_HGCHXC_2_I[i] >= sum(self.V_HP_HGCHXC_max * self.m.Z_HP_HGCHXC_N_2_I[n,i] + self.T_HP_HGCHXC_max_N[n] * self.m.V_HP_HGCHXC_N_2_I[n,i] - self.V_HP_HGCHXC_max * self.T_HP_HGCHXC_max_N[n] * self.m.B_T_HP_HGCHXC_N_2_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.W_HP_HGCHXC_2_I[i] <= sum(self.V_HP_HGCHXC_max * self.m.Z_HP_HGCHXC_N_2_I[n,i] + self.T_HP_HGCHXC_min_N[n] * self.m.V_HP_HGCHXC_N_2_I[n,i] - self.V_HP_HGCHXC_max * self.T_HP_HGCHXC_min_N[n] * self.m.B_T_HP_HGCHXC_N_2_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.W_HP_HGCHXC_2_I[i] <= sum(self.V_HP_HGCHXC_min * self.m.Z_HP_HGCHXC_N_2_I[n,i] + self.T_HP_HGCHXC_max_N[n] * self.m.V_HP_HGCHXC_N_2_I[n,i] - self.V_HP_HGCHXC_min * self.T_HP_HGCHXC_max_N[n] * self.m.B_T_HP_HGCHXC_N_2_I[n,i] for n in self.N_MC))

            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.T_HP_HGCHXC_2_I[i] == sum(self.m.Z_HP_HGCHXC_N_2_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.V_HP_HGCHXC_2_I[i] == sum(self.m.V_HP_HGCHXC_N_2_I[n,i] for n in self.N_MC))

        for i in self.I[0:-1]:
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(sum(self.m.B_T_HP_HGCHXC_N_2_I[n,i] for n in self.N_MC) == 1)

        for i in self.I[0:-1]:
            for n in self.N_MC: 
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.V_HP_HGCHXC_N_2_I[n,i]  <= self.V_HP_HGCHXC_max * self.m.B_T_HP_HGCHXC_N_2_I[n,i])
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.V_HP_HGCHXC_N_2_I[n,i]  >= self.V_HP_HGCHXC_min * self.m.B_T_HP_HGCHXC_N_2_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:                
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_N_2_I[n,i] <= self.T_HP_HGCHXC_max_N[n] * self.m.B_T_HP_HGCHXC_N_2_I[n,i])
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_N_2_I[n,i] >= self.T_HP_HGCHXC_min_N[n] * self.m.B_T_HP_HGCHXC_N_2_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_N_2_I[n,i] <= self.T_upper_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_N_2_I[n,i] >= self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_N_2_I[n,i] <= self.T_upper_MC * self.m.B_T_HP_HGCHXC_N_2_I[n,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_N_2_I[n,i] >= self.T_lower_MC * self.m.B_T_HP_HGCHXC_N_2_I[n,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_N_2_I[n,i] <= self.m.T_HP_HGCHXC_N_2_I[n,i] - (1 - self.m.B_T_HP_HGCHXC_N_2_I[n,i]) * self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_N_2_I[n,i] >= self.m.T_HP_HGCHXC_N_2_I[n,i] - (1 - self.m.B_T_HP_HGCHXC_N_2_I[n,i]) * self.T_upper_MC) ## Big M constraint input

        for i in self.I[0:-(self.ControlPeriod3)]:
            if i%self.ControlPeriod3 == 0:
                for j in range(1,self.ControlPeriod3):
                    self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.V_HP_HGCHXC_2_I[i] == self.m.V_HP_HGCHXC_2_I[i+j])
        
        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.T_HP_HGCHXC_N_2_I[n,i] == self.m.T_HP_LT_I[i+1] - self.m.T_HXC_b_I[i+1])

        for i in self.I[0:-1]:
            for h in self.H:
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_2_H_I[h,i] <= self.W_upper_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_2_H_I[h,i] >= self.W_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_2_H_I[h,i] <= self.W_upper_MC * self.m.B_HP_H_I[h,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_2_H_I[h,i] >= self.W_lower_MC * self.m.B_HP_H_I[h,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_2_H_I[h,i] <= self.m.W_HP_HGCHXC_2_I[i] - (1 - self.m.B_HP_H_I[h,i]) * self.W_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGCHXC_2_H_I[h,i] >= self.m.W_HP_HGCHXC_2_I[i] - (1 - self.m.B_HP_H_I[h,i]) * self.W_upper_MC) ## Big M constraint input

        for i in self.I[:-1]:
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.V_HP_HGC_I[i] + self.m.V_HP_HGCHXC_I[i] == 1) ## Both flows for HP  
            for n in self.N_MC:
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.V_HP_HGC_N_I[n,i] + self.m.V_HP_HGCHXC_N_I[n,i] <= 1)
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.V_HP_HGCHXC_N_I[n,i] == self.m.V_HP_HGCHXC_N_2_I[n,i])
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.V_HP_HGC_N_I[n,i] == self.m.V_HP_HGC_N_I[n,i])

            for h in self.H:  # energy constraints  
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGC_2_H_I[h,i] + self.m.Z_HP_HGCHXC_H_I[h,i] + self.m.Z_HP_HGCHXC_2_H_I[h,i] + self.m.Z_HP_HGC_H_I[h,i] == 0)
                                                #NEW:            HP-side                       HGC-Side                    HXC-Side                          HGC-Side
        for i in self.I[:-2]:
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HGC_HGCHXC_I[i] >= self.m.V_HP_HGC_I[i] - self.m.V_HP_HGC_I[i+1])
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HGC_HGCHXC_I[i] >= self.m.V_HP_HGC_I[i+1] - self.m.V_HP_HGC_I[i])
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HGC_HGCHXC_I[i] >= self.m.V_HP_HGCHXC_I[i] - self.m.V_HP_HGCHXC_I[i+1])
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HGC_HGCHXC_I[i] >= self.m.V_HP_HGCHXC_I[i+1] - self.m.V_HP_HGCHXC_I[i])
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HGC_HGCHXC_I[i] <= 1)

        if self.End_Toggle_Constraints == True:
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HGC_HGCHXC_I[self.I[-2]] >= self.m.V_HP_HGC_I[self.I[-2]] - self.V_HP_HGC_end)
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HGC_HGCHXC_I[self.I[-2]] >= self.V_HP_HGC_end - self.m.V_HP_HGC_I[self.I[-2]])
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HGC_HGCHXC_I[self.I[-2]] >= self.m.V_HP_HGCHXC_I[self.I[-2]] - self.V_HGCHXC_end)
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HGC_HGCHXC_I[self.I[-2]] >= self.V_HGCHXC_end - self.m.V_HP_HGCHXC_I[self.I[-2]])
            self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HGC_HGCHXC_I[self.I[-2]] <= 1)

        for i in self.I[0:-1]:
            for h in self.H:
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HXC_HXC_b_I[h,i] <= self.W_upper_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HXC_HXC_b_I[h,i] >= self.W_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HXC_HXC_b_I[h,i] <= self.W_upper_MC * self.m.B_HP_H_I[h,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HXC_HXC_b_I[h,i] >= self.W_lower_MC * self.m.B_HP_H_I[h,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HXC_HXC_b_I[h,i] <= (self.m.T_HXC_b_I[i+1] - self.m.T_HXC_b_out_I[i]) - (1 - self.m.B_HP_H_I[h,i]) * self.W_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HXC_HXC_b_I[h,i] >= (self.m.T_HXC_b_I[i+1] - self.m.T_HXC_b_out_I[i]) - (1 - self.m.B_HP_H_I[h,i]) * self.W_upper_MC) ## Big M constraint input
        
        for i in self.I[0:-1]:
            for h in self.H:
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGC_2_H_I[h,i] <= self.W_upper_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGC_2_H_I[h,i] >= self.W_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGC_2_H_I[h,i] <= self.W_upper_MC * self.m.B_HP_H_I[h,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGC_2_H_I[h,i] >= self.W_lower_MC * self.m.B_HP_H_I[h,i]) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGC_2_H_I[h,i] <= (self.m.T_HGC_I[i+1] - self.m.T_HP_LT_I[i+1]) - (1 - self.m.B_HP_H_I[h,i]) * self.W_lower_MC) ## Big M constraint input
                self.m.Constraint_V_HP_HGC_HGCHXC_I.add(self.m.Z_HP_HGC_2_H_I[h,i] >= (self.m.T_HGC_I[i+1] - self.m.T_HP_LT_I[i+1]) - (1 - self.m.B_HP_H_I[h,i]) * self.W_upper_MC) ## Big M constraint input

        ## V_GS_HGS_CS
        self.m.Constraint_V_GS_HGS_CS_I = pyo.ConstraintList()
        # View of GS, since we have more than one storage
        for i in self.I[0:-1]:
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.W_GS_HGS_I_2[i] >= sum(self.V_GS_HGS_min * self.m.Z_GS_HGS_N_I_2[n,i] + self.T_GS_HGS_min_N[n] * self.m.V_GS_HGS_N_I_2[n,i] - self.V_GS_HGS_min * self.T_GS_HGS_min_N[n] * self.m.B_T_GS_HGS_N_I_2[n,i] for n in self.N_MC))
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.W_GS_HGS_I_2[i] >= sum(self.V_GS_HGS_max * self.m.Z_GS_HGS_N_I_2[n,i] + self.T_GS_HGS_max_N[n] * self.m.V_GS_HGS_N_I_2[n,i] - self.V_GS_HGS_max * self.T_GS_HGS_max_N[n] * self.m.B_T_GS_HGS_N_I_2[n,i] for n in self.N_MC))
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.W_GS_HGS_I_2[i] <= sum(self.V_GS_HGS_max * self.m.Z_GS_HGS_N_I_2[n,i] + self.T_GS_HGS_min_N[n] * self.m.V_GS_HGS_N_I_2[n,i] - self.V_GS_HGS_max * self.T_GS_HGS_min_N[n] * self.m.B_T_GS_HGS_N_I_2[n,i] for n in self.N_MC))
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.W_GS_HGS_I_2[i] <= sum(self.V_GS_HGS_min * self.m.Z_GS_HGS_N_I_2[n,i] + self.T_GS_HGS_max_N[n] * self.m.V_GS_HGS_N_I_2[n,i] - self.V_GS_HGS_min * self.T_GS_HGS_max_N[n] * self.m.B_T_GS_HGS_N_I_2[n,i] for n in self.N_MC))

            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.T_GS_HGS_I_2[i] == sum(self.m.Z_GS_HGS_N_I_2[n,i] for n in self.N_MC))
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.V_GS_HGS_I_2[i] == sum(self.m.V_GS_HGS_N_I_2[n,i] for n in self.N_MC))

        for i in self.I[0:-1]:
            self.m.Constraint_V_GS_HGS_CS_I.add(sum(self.m.B_T_GS_HGS_N_I_2[n,i] for n in self.N_MC) == 1)

        for i in self.I[0:-1]:
            for n in self.N_MC: 
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.V_GS_HGS_N_I_2[n,i]  <= self.V_GS_HGS_max * self.m.B_T_GS_HGS_N_I_2[n,i])
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.V_GS_HGS_N_I_2[n,i]  >= self.V_GS_HGS_min * self.m.B_T_GS_HGS_N_I_2[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:                
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_HGS_N_I_2[n,i] <= self.T_GS_HGS_max_N[n] * self.m.B_T_GS_HGS_N_I_2[n,i])
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_HGS_N_I_2[n,i] >= self.T_GS_HGS_min_N[n] * self.m.B_T_GS_HGS_N_I_2[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_HGS_N_I_2[n,i] <= self.T_upper_MC) ## Big M constraint input
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_HGS_N_I_2[n,i] >= self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_HGS_N_I_2[n,i] <= self.T_upper_MC * self.m.B_T_GS_HGS_N_I_2[n,i]) ## Big M constraint input
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_HGS_N_I_2[n,i] >= self.T_lower_MC * self.m.B_T_GS_HGS_N_I_2[n,i]) ## Big M constraint input
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_HGS_N_I_2[n,i] <= self.m.T_GS_HGS_N_I_2[n,i] - (1 - self.m.B_T_GS_HGS_N_I_2[n,i]) * self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_HGS_N_I_2[n,i] >= self.m.T_GS_HGS_N_I_2[n,i] - (1 - self.m.B_T_GS_HGS_N_I_2[n,i]) * self.T_upper_MC) ## Big M constraint input

        for i in self.I[0:-(self.ControlPeriod3)]:
            if i%self.ControlPeriod3 == 0:
                for j in range(1,self.ControlPeriod3):
                    self.m.Constraint_V_GS_HGS_CS_I.add(self.m.V_GS_HGS_I_2[i] == self.m.V_GS_HGS_I_2[i+j])
        
        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.T_GS_HGS_N_I_2[n,i] == self.m.T_GS_W_I_WR_WC[i+1,0,1] - self.m.T_HGS_I[i+1])

        for i in self.I[0:-1]:
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.W_GS_CS_I_2[i] >= sum(self.V_GS_CS_min * self.m.Z_GS_CS_N_I_2[n,i] + self.T_GS_CS_min_N[n] * self.m.V_GS_CS_N_I_2[n,i] - self.V_GS_CS_min * self.T_GS_CS_min_N[n] * self.m.B_T_GS_CS_N_I_2[n,i] for n in self.N_MC))
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.W_GS_CS_I_2[i] >= sum(self.V_GS_CS_max * self.m.Z_GS_CS_N_I_2[n,i] + self.T_GS_CS_max_N[n] * self.m.V_GS_CS_N_I_2[n,i] - self.V_GS_CS_max * self.T_GS_CS_max_N[n] * self.m.B_T_GS_CS_N_I_2[n,i] for n in self.N_MC))
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.W_GS_CS_I_2[i] <= sum(self.V_GS_CS_max * self.m.Z_GS_CS_N_I_2[n,i] + self.T_GS_CS_min_N[n] * self.m.V_GS_CS_N_I_2[n,i] - self.V_GS_CS_max * self.T_GS_CS_min_N[n] * self.m.B_T_GS_CS_N_I_2[n,i] for n in self.N_MC))
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.W_GS_CS_I_2[i] <= sum(self.V_GS_CS_min * self.m.Z_GS_CS_N_I_2[n,i] + self.T_GS_CS_max_N[n] * self.m.V_GS_CS_N_I_2[n,i] - self.V_GS_CS_min * self.T_GS_CS_max_N[n] * self.m.B_T_GS_CS_N_I_2[n,i] for n in self.N_MC))

            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.T_GS_CS_I_2[i] == sum(self.m.Z_GS_CS_N_I_2[n,i] for n in self.N_MC))
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.V_GS_CS_I_2[i] == sum(self.m.V_GS_CS_N_I_2[n,i] for n in self.N_MC))

        for i in self.I[0:-1]:
            self.m.Constraint_V_GS_HGS_CS_I.add(sum(self.m.B_T_GS_CS_N_I_2[n,i] for n in self.N_MC) == 1)

        for i in self.I[0:-1]:
            for n in self.N_MC: 
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.V_GS_CS_N_I_2[n,i]  <= self.V_GS_CS_max * self.m.B_T_GS_CS_N_I_2[n,i])
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.V_GS_CS_N_I_2[n,i]  >= self.V_GS_CS_min * self.m.B_T_GS_CS_N_I_2[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:                
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_CS_N_I_2[n,i] <= self.T_GS_CS_max_N[n] * self.m.B_T_GS_CS_N_I_2[n,i])
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_CS_N_I_2[n,i] >= self.T_GS_CS_min_N[n] * self.m.B_T_GS_CS_N_I_2[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_CS_N_I_2[n,i] <= self.T_upper_MC) ## Big M constraint input
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_CS_N_I_2[n,i] >= self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_CS_N_I_2[n,i] <= self.T_upper_MC * self.m.B_T_GS_CS_N_I_2[n,i]) ## Big M constraint input
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_CS_N_I_2[n,i] >= self.T_lower_MC * self.m.B_T_GS_CS_N_I_2[n,i]) ## Big M constraint input
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_CS_N_I_2[n,i] <= self.m.T_GS_CS_N_I_2[n,i] - (1 - self.m.B_T_GS_CS_N_I_2[n,i]) * self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_CS_N_I_2[n,i] >= self.m.T_GS_CS_N_I_2[n,i] - (1 - self.m.B_T_GS_CS_N_I_2[n,i]) * self.T_upper_MC) ## Big M constraint input

        for i in self.I[0:-(self.ControlPeriod3)]:
            if i%self.ControlPeriod3 == 0:
                for j in range(1,self.ControlPeriod3):
                    self.m.Constraint_V_GS_HGS_CS_I.add(self.m.V_GS_CS_I_2[i] == self.m.V_GS_CS_I_2[i+j])
        
        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.T_GS_CS_N_I_2[n,i] == self.m.T_GS_W_I_WR_WC[i+1,0,1] - self.m.T_CS_I[i+1])
        
        # View of components connected to HP 
        for i in self.I[0:-1]:
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.W_GS_HGS_I[i] >= sum(self.V_GS_HGS_min * self.m.Z_GS_HGS_N_I[n,i] + self.T_GS_HGS_min_N[n] * self.m.V_GS_HGS_N_I[n,i] - self.V_GS_HGS_min * self.T_GS_HGS_min_N[n] * self.m.B_T_GS_HGS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.W_GS_HGS_I[i] >= sum(self.V_GS_HGS_max * self.m.Z_GS_HGS_N_I[n,i] + self.T_GS_HGS_max_N[n] * self.m.V_GS_HGS_N_I[n,i] - self.V_GS_HGS_max * self.T_GS_HGS_max_N[n] * self.m.B_T_GS_HGS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.W_GS_HGS_I[i] <= sum(self.V_GS_HGS_max * self.m.Z_GS_HGS_N_I[n,i] + self.T_GS_HGS_min_N[n] * self.m.V_GS_HGS_N_I[n,i] - self.V_GS_HGS_max * self.T_GS_HGS_min_N[n] * self.m.B_T_GS_HGS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.W_GS_HGS_I[i] <= sum(self.V_GS_HGS_min * self.m.Z_GS_HGS_N_I[n,i] + self.T_GS_HGS_max_N[n] * self.m.V_GS_HGS_N_I[n,i] - self.V_GS_HGS_min * self.T_GS_HGS_max_N[n] * self.m.B_T_GS_HGS_N_I[n,i] for n in self.N_MC))

            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.T_GS_HGS_I[i] == sum(self.m.Z_GS_HGS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.V_GS_HGS_I[i] == sum(self.m.V_GS_HGS_N_I[n,i] for n in self.N_MC))

        for i in self.I[0:-1]:
            self.m.Constraint_V_GS_HGS_CS_I.add(sum(self.m.B_T_GS_HGS_N_I[n,i] for n in self.N_MC) == 1)

        for i in self.I[0:-1]:
            for n in self.N_MC: 
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.V_GS_HGS_N_I[n,i]  <= self.V_GS_HGS_max * self.m.B_T_GS_HGS_N_I[n,i])
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.V_GS_HGS_N_I[n,i]  >= self.V_GS_HGS_min * self.m.B_T_GS_HGS_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:                
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_HGS_N_I[n,i] <= self.T_GS_HGS_max_N[n] * self.m.B_T_GS_HGS_N_I[n,i])
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_HGS_N_I[n,i] >= self.T_GS_HGS_min_N[n] * self.m.B_T_GS_HGS_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_HGS_N_I[n,i] <= self.T_upper_MC) ## Big M constraint input
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_HGS_N_I[n,i] >= self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_HGS_N_I[n,i] <= self.T_upper_MC * self.m.B_T_GS_HGS_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_HGS_N_I[n,i] >= self.T_lower_MC * self.m.B_T_GS_HGS_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_HGS_N_I[n,i] <= self.m.T_GS_HGS_N_I[n,i] - (1 - self.m.B_T_GS_HGS_N_I[n,i]) * self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_HGS_N_I[n,i] >= self.m.T_GS_HGS_N_I[n,i] - (1 - self.m.B_T_GS_HGS_N_I[n,i]) * self.T_upper_MC) ## Big M constraint input

        for i in self.I[0:-(self.ControlPeriod3)]:
            if i%self.ControlPeriod3 == 0:
                for j in range(1,self.ControlPeriod3):
                    self.m.Constraint_V_GS_HGS_CS_I.add(self.m.V_GS_HGS_I[i] == self.m.V_GS_HGS_I[i+j])
        
        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.T_GS_HGS_N_I[n,i] == self.m.T_GS_I[i+1] - self.m.T_HGS_I[i+1])

        for i in self.I[0:-1]:
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.W_GS_CS_I[i] >= sum(self.V_GS_CS_min * self.m.Z_GS_CS_N_I[n,i] + self.T_GS_CS_min_N[n] * self.m.V_GS_CS_N_I[n,i] - self.V_GS_CS_min * self.T_GS_CS_min_N[n] * self.m.B_T_GS_CS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.W_GS_CS_I[i] >= sum(self.V_GS_CS_max * self.m.Z_GS_CS_N_I[n,i] + self.T_GS_CS_max_N[n] * self.m.V_GS_CS_N_I[n,i] - self.V_GS_CS_max * self.T_GS_CS_max_N[n] * self.m.B_T_GS_CS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.W_GS_CS_I[i] <= sum(self.V_GS_CS_max * self.m.Z_GS_CS_N_I[n,i] + self.T_GS_CS_min_N[n] * self.m.V_GS_CS_N_I[n,i] - self.V_GS_CS_max * self.T_GS_CS_min_N[n] * self.m.B_T_GS_CS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.W_GS_CS_I[i] <= sum(self.V_GS_CS_min * self.m.Z_GS_CS_N_I[n,i] + self.T_GS_CS_max_N[n] * self.m.V_GS_CS_N_I[n,i] - self.V_GS_CS_min * self.T_GS_CS_max_N[n] * self.m.B_T_GS_CS_N_I[n,i] for n in self.N_MC))

            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.T_GS_CS_I[i] == sum(self.m.Z_GS_CS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.V_GS_CS_I[i] == sum(self.m.V_GS_CS_N_I[n,i] for n in self.N_MC))

        for i in self.I[0:-1]:
            self.m.Constraint_V_GS_HGS_CS_I.add(sum(self.m.B_T_GS_CS_N_I[n,i] for n in self.N_MC) == 1)

        for i in self.I[0:-1]:
            for n in self.N_MC: 
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.V_GS_CS_N_I[n,i]  <= self.V_GS_CS_max * self.m.B_T_GS_CS_N_I[n,i])
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.V_GS_CS_N_I[n,i]  >= self.V_GS_CS_min * self.m.B_T_GS_CS_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:                
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_CS_N_I[n,i] <= self.T_GS_CS_max_N[n] * self.m.B_T_GS_CS_N_I[n,i])
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_CS_N_I[n,i] >= self.T_GS_CS_min_N[n] * self.m.B_T_GS_CS_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_CS_N_I[n,i] <= self.T_upper_MC) ## Big M constraint input
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_CS_N_I[n,i] >= self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_CS_N_I[n,i] <= self.T_upper_MC * self.m.B_T_GS_CS_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_CS_N_I[n,i] >= self.T_lower_MC * self.m.B_T_GS_CS_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_CS_N_I[n,i] <= self.m.T_GS_CS_N_I[n,i] - (1 - self.m.B_T_GS_CS_N_I[n,i]) * self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_GS_CS_N_I[n,i] >= self.m.T_GS_CS_N_I[n,i] - (1 - self.m.B_T_GS_CS_N_I[n,i]) * self.T_upper_MC) ## Big M constraint input

        for i in self.I[0:-(self.ControlPeriod3)]:
            if i%self.ControlPeriod3 == 0:
                for j in range(1,self.ControlPeriod3):
                    self.m.Constraint_V_GS_HGS_CS_I.add(self.m.V_GS_CS_I[i] == self.m.V_GS_CS_I[i+j])
        
        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_V_GS_HGS_CS_I.add(self.m.T_GS_CS_N_I[n,i] == self.m.T_GS_I[i+1] - self.m.T_CS_I[i+1])

        # Connection of both valve positions
        for i in self.I[0:-1]:
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.V_GS_HGS_I[i] == self.m.V_GS_HGS_I_2[i]) ## Maximum of 1 for both ways
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.V_GS_CS_I[i] == self.m.V_GS_CS_I_2[i]) ## Maximum of 1 for both ways

        # Maximum of 1 for all ways 
        for i in self.I[0:-1]:
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.V_GS_HGS_I[i] + self.m.V_GS_CS_I[i] <= 1) ## Maximum of 1 for both ways

        for i in self.I[:-2]:
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_HGS_CS_I[i] >= self.m.V_GS_HGS_I[i] - self.m.V_GS_HGS_I[i+1])
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_HGS_CS_I[i] >= self.m.V_GS_HGS_I[i+1] - self.m.V_GS_HGS_I[i])
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_HGS_CS_I[i] >= self.m.V_GS_CS_I[i] - self.m.V_GS_CS_I[i+1])
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_HGS_CS_I[i] >= self.m.V_GS_CS_I[i+1] - self.m.V_GS_CS_I[i])
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_HGS_CS_I[i] <= 1)

        if self.End_Toggle_Constraints == True:
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_HGS_CS_I[self.I[-2]] >= self.m.V_GS_HGS_I[self.I[-2]] - self.V_GS_HGS_end)
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_HGS_CS_I[self.I[-2]] >= self.V_GS_HGS_end - self.m.V_GS_HGS_I[self.I[-2]])
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_HGS_CS_I[self.I[-2]] >= self.m.V_GS_CS_I[self.I[-2]] - self.V_GS_CS_end)
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_HGS_CS_I[self.I[-2]] >= self.V_GS_CS_end - self.m.V_GS_CS_I[self.I[-2]])
            self.m.Constraint_V_GS_HGS_CS_I.add(self.m.Z_HGS_CS_I[self.I[-2]] <= 1)
 
        ## HXA
        self.m.Constraint_HXA_I = pyo.ConstraintList()
        for i in self.I[0:-1]:
            self.m.Constraint_HXA_I.add(self.m.T_HXAR_in_I[i] == self.temp_amb_I[i]) ## General temperature connection

        for i in self.I[0:-1]:
            self.m.Constraint_HXA_I.add(self.m.W_HXA_I[i] >= sum(self.V_HXA_min * self.m.Z_HXAR_N_I[n,i] + self.T_HXAR_min_N[n] * self.m.P_HXA_N_I[n,i] - self.V_HXA_min * self.T_HXAR_min_N[n] * self.m.B_T_HXAR_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_HXA_I.add(self.m.W_HXA_I[i] >= sum(self.V_HXA_max * self.m.Z_HXAR_N_I[n,i] + self.T_HXAR_max_N[n] * self.m.P_HXA_N_I[n,i] - self.V_HXA_max * self.T_HXAR_max_N[n] * self.m.B_T_HXAR_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_HXA_I.add(self.m.W_HXA_I[i] <= sum(self.V_HXA_max * self.m.Z_HXAR_N_I[n,i] + self.T_HXAR_min_N[n] * self.m.P_HXA_N_I[n,i] - self.V_HXA_max * self.T_HXAR_min_N[n] * self.m.B_T_HXAR_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_HXA_I.add(self.m.W_HXA_I[i] <= sum(self.V_HXA_min * self.m.Z_HXAR_N_I[n,i] + self.T_HXAR_max_N[n] * self.m.P_HXA_N_I[n,i] - self.V_HXA_min * self.T_HXAR_max_N[n] * self.m.B_T_HXAR_N_I[n,i] for n in self.N_MC))

            self.m.Constraint_HXA_I.add(self.m.T_HXAR_I[i] == sum(self.m.Z_HXAR_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_HXA_I.add(self.m.P_HXA_I[i] == sum(self.m.P_HXA_N_I[n,i] for n in self.N_MC))

        for i in self.I[0:-1]:
            self.m.Constraint_HXA_I.add(sum(self.m.B_T_HXAR_N_I[n,i] for n in self.N_MC) == 1)

        for i in self.I[0:-1]:
            for n in self.N_MC: 
                self.m.Constraint_HXA_I.add(self.m.P_HXA_N_I[n,i]  <= self.V_HXA_max * self.m.B_T_HXAR_N_I[n,i] * (1-self.temp_frost_I[i]))
                self.m.Constraint_HXA_I.add(self.m.P_HXA_N_I[n,i]  >= self.V_HXA_min * self.m.B_T_HXAR_N_I[n,i] * (1-self.temp_frost_I[i]))

        for i in self.I[0:-1]:
            for n in self.N_MC:                
                self.m.Constraint_HXA_I.add(self.m.Z_HXAR_N_I[n,i] <= self.T_HXAR_max_N[n] * self.m.B_T_HXAR_N_I[n,i])
                self.m.Constraint_HXA_I.add(self.m.Z_HXAR_N_I[n,i] >= self.T_HXAR_min_N[n] * self.m.B_T_HXAR_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_HXA_I.add(self.m.Z_HXAR_N_I[n,i] <= self.T_upper_MC) ## Big M constraint input
                self.m.Constraint_HXA_I.add(self.m.Z_HXAR_N_I[n,i] >= self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_HXA_I.add(self.m.Z_HXAR_N_I[n,i] <= self.T_upper_MC * self.m.B_T_HXAR_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_HXA_I.add(self.m.Z_HXAR_N_I[n,i] >= self.T_lower_MC * self.m.B_T_HXAR_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_HXA_I.add(self.m.Z_HXAR_N_I[n,i] <= self.m.T_HXAR_N_I[n,i] - (1 - self.m.B_T_HXAR_N_I[n,i]) * self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_HXA_I.add(self.m.Z_HXAR_N_I[n,i] >= self.m.T_HXAR_N_I[n,i] - (1 - self.m.B_T_HXAR_N_I[n,i]) * self.T_upper_MC) ## Big M constraint input

        for i in self.I[0:-(self.ControlPeriod3)]:
            if i%self.ControlPeriod3 == 0:
                for j in range(1,self.ControlPeriod3):
                    self.m.Constraint_HXA_I.add(self.m.P_HXA_I[i] == self.m.P_HXA_I[i+j])
        
        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_HXA_I.add(self.m.T_HXAR_N_I[n,i] == self.m.T_HXAR_in_I[i] - self.m.T_HXA_I[i+1])

        for i in self.I[0:-1]:
            self.m.Constraint_HXA_I.add(self.m.E_HXA_EL_I[i] == self.m.P_HXA_I[i] * self.e_HXA_EL_device + (self.m.V_HXA_HXH_I[i] + self.m.V_HXA_HGC_I[i]) * self.e_HXA_EL_pump)

        for i in self.I[:-2]:
            self.m.Constraint_HXA_I.add(self.m.Z_HXA_I[i] >= self.m.P_HXA_I[i] - self.m.P_HXA_I[i+1])
            self.m.Constraint_HXA_I.add(self.m.Z_HXA_I[i] >= self.m.P_HXA_I[i+1] - self.m.P_HXA_I[i])
            self.m.Constraint_HXA_I.add(self.m.Z_HXA_I[i] <= 1)

        if self.End_Toggle_Constraints == True:
            self.m.Constraint_HXA_I.add(self.m.Z_HXA_I[self.I[-2]] >= self.m.P_HXA_I[self.I[-2]] - self.V_HXA_end)
            self.m.Constraint_HXA_I.add(self.m.Z_HXA_I[self.I[-2]] >= self.V_HXA_end - self.m.P_HXA_I[self.I[-2]])
            self.m.Constraint_HXA_I.add(self.m.Z_HXA_I[self.I[-2]] <= 1)

        # HXA tank
        self.m.Constraint_HXA_I.add(self.m.T_HXA_I[0] == self.T_HXA_start) ## Start temperature

        for i in self.I[0:-1]:
            self.m.Constraint_HXA_I.add(self.m.T_HXA_I[i+1] == self.m.T_HXA_I[i] + self.StepSizeInSec2 * (1/(self.m_HXA_b * self.c_b) * (self.alpha_factor_HXA * self.c_a * self.mdot_HXA_a * self.m.W_HXA_I[i] - self.c_b * self.mdot_HXA_b * self.m.W_HXA_HXH_I[i] - self.c_b * self.mdot_HXA_b * self.m.W_HXA_HGC_I[i])) + self.StepSizeInSec2 * self.alpha_HXA_time * (self.temp_amb_I[i] - self.m.T_HXA_I[i+1])/(self.m_HXA_b * self.c_b)) ## General energy flow
            
        for i in self.I[1:]:
            self.m.Constraint_HXA_I.add(self.m.T_HXA_I[i] <= self.T_HXA_max + self.m.S_T_HXA_I[i]) ## Temperature range tank
            self.m.Constraint_HXA_I.add(self.m.T_HXA_I[i] >= self.T_HXA_min - self.m.S_T_HXA_I[i]) ## Temperature range tank

        ## HXH
        self.m.Constraint_HXH_I = pyo.ConstraintList()

        ## HXH Tank broil
        self.m.Constraint_HXH_I.add(self.m.T_HXH_b_I[0] == self.T_HXH_start) ## Start temperature

        for i in self.I[0:-1]:
            self.m.Constraint_HXH_I.add(self.m.T_HXH_b_I[i+1] == self.m.T_HXH_b_I[i] + self.StepSizeInSec2 * (1/(self.m_HXH_b * self.c_b) * (self.m.W_HXA_HXH_I[i] * self.mdot_HXA_b * self.c_b - self.m.Z_HXH_HXH_b_I[i] * self.mdot_HXA_b * self.c_b)) + self.StepSizeInSec2 * self.alpha_HXH_time * (self.t_default - self.m.T_HXH_b_I[i+1])/(self.m_HXH_b * self.c_b)) ## General energy flow

        for i in self.I[1:]:    
            self.m.Constraint_HXH_I.add(self.m.T_HXH_b_I[i] <= self.T_HXH_max + self.m.S_T_HXH_b_I[i]) ## Temperature range tank
            self.m.Constraint_HXH_I.add(self.m.T_HXH_b_I[i] >= self.T_HXH_min - self.m.S_T_HXH_b_I[i]) ## Temperature range tank  

        ## HXH Tank water
        self.m.Constraint_HXH_I.add(self.m.T_HXH_w_I[0] == self.T_HXH_start) ## Start temperature

        for i in self.I[0:-1]:
            self.m.Constraint_HXH_I.add(self.m.T_HXH_w_I[i+1] == self.m.T_HXH_w_I[i] + self.StepSizeInSec2 * (1/(self.m_HXH_w * self.c_w) * (self.c_w * sum(self.mdot_HP_w_H[h] * self.m.Z_HP_HXH_H_I[h,i] for h in self.H) - self.c_w * sum(self.mdot_HP_w_H[h] * self.m.Z_HXH_HXH_w_I[h,i] for h in self.H))) + self.StepSizeInSec2 * self.alpha_HXH_time * (self.t_default - self.m.T_HXH_w_I[i+1])/(self.m_HXH_w * self.c_w)) ## General energy flow

        for i in self.I[1:]:    
            self.m.Constraint_HXH_I.add(self.m.T_HXH_w_I[i] <= self.T_HXH_max + self.m.S_T_HXH_w_I[i]) ## Temperature range tank
            self.m.Constraint_HXH_I.add(self.m.T_HXH_w_I[i] >= self.T_HXH_min - self.m.S_T_HXH_w_I[i]) ## Temperature range tank 

        ## HXH     
        for i in self.I[:-1]:
            self.m.Constraint_HXH_I.add(0 == sum(self.m.Z_HXH_HXH_w_I[h,i] * self.mdot_HP_w_H[h] * self.c_w for h in self.H) + self.m.Qdot_HXH_w_b_I[i])
            self.m.Constraint_HXH_I.add(0 == self.m.Z_HXH_HXH_b_I[i] * self.mdot_HXA_b * self.c_b - self.m.Qdot_HXH_w_b_I[i])
            self.m.Constraint_HXH_I.add(self.m.Qdot_HXH_w_b_I[i] == ((self.m.T_HXH_w_I[i+1] + self.m.T_HXH_w_out_I[i])/2 - (self.m.T_HXH_b_I[i+1] + self.m.T_HXH_b_out_I[i])/2 ) * self.a_HXH_w_b * self.alpha_HXH_w_b)

        for i in self.I[:-1]:
            self.m.Constraint_HXH_I.add(self.m.T_HXH_I[i] == (self.m.T_HXH_w_I[i] + self.m.T_HXH_w_out_I[i] + self.m.T_HXH_b_I[i] + self.m.T_HXH_b_out_I[i])/4)
        
        for i in self.I[1:-1]:
            self.m.Constraint_HXH_I.add(self.m.T_HXH_I[i] <= self.T_HXH_max + self.m.S_T_HXH_I[i]) ## Temperature range 
            self.m.Constraint_HXH_I.add(self.m.T_HXH_I[i] >= self.T_HXH_min - self.m.S_T_HXH_I[i]) ## Temperature range 

        ## HGC
        self.m.Constraint_HGC_I = pyo.ConstraintList()
        self.m.Constraint_HGC_I.add(self.m.T_HGC_I[0] == self.T_HGC_start) ## Start temperature

        for i in self.I[0:-1]:
            self.m.Constraint_HGC_I.add(self.m.T_HGC_I[i+1] == self.m.T_HGC_I[i] + self.StepSizeInSec2 * (1/(self.m_HGC_b * self.c_b) * (self.c_b * sum(self.mdot_HP_b_H[h] * self.m.Z_HP_HGC_H_I[h,i] for h in self.H) + self.c_b * sum(self.mdot_HP_b_H[h] * self.m.Z_HP_HGCHXC_H_I[h,i] for h in self.H) + self.c_b * self.mdot_HXA_b * self.m.W_HXA_HGC_I[i])) + self.StepSizeInSec2 * self.alpha_HGC_time * (self.t_default - self.m.T_HGC_I[i+1])/(self.m_HGC_b * self.c_b)) ## General energy flow

        for i in self.I[1:]:    
            self.m.Constraint_HGC_I.add(self.m.T_HGC_I[i] <= self.T_HGC_max + self.m.S_T_HGC_I[i]) ## Temperature range tank
            self.m.Constraint_HGC_I.add(self.m.T_HGC_I[i] >= self.T_HGC_min - self.m.S_T_HGC_I[i]) ## Temperature range tank  
        
        ## HXC
        self.m.Constraint_HXC_I = pyo.ConstraintList()

        ## HXC Tank water 
        self.m.Constraint_HXC_I.add(self.m.T_HXC_w_I[0] == self.T_HXC_start) ## Start temperature

        for i in self.I[0:-1]:
            self.m.Constraint_HXC_I.add(self.m.T_HXC_w_I[i+1] == self.m.T_HXC_w_I[i] + self.StepSizeInSec2 * (1/(self.m_HXC_w * self.c_w) * -(self.m.W_HXC_HGS_I[i] * self.mdot_VP_tot * self.c_w + self.m.W_HXC_CS_I[i] * self.mdot_VP_tot * self.c_w + self.m.W_HXC_RLTS_I[i] * self.mdot_VP_tot * self.c_w + self.m.Z_HXC_HXC_w_I[i] * self.mdot_VP_tot * self.c_w)) + self.StepSizeInSec2 * self.alpha_HXC_time * (self.t_default - self.m.T_HXC_w_I[i+1])/(self.m_HXC_w * self.c_w)) ## General energy flow

        for i in self.I[1:]:    
            self.m.Constraint_HXC_I.add(self.m.T_HXC_w_I[i] <= self.T_HXC_max + self.m.S_T_HXC_w_I[i]) ## Temperature range tank
            self.m.Constraint_HXC_I.add(self.m.T_HXC_w_I[i] >= self.T_HXC_min - self.m.S_T_HXC_w_I[i]) ## Temperature range tank  

        ## HXC Tank broil
        self.m.Constraint_HXC_I.add(self.m.T_HXC_b_I[0] == self.T_HXC_start) ## Start temperature

        for i in self.I[0:-1]:
            self.m.Constraint_HXC_I.add(self.m.T_HXC_b_I[i+1] == self.m.T_HXC_b_I[i] + self.StepSizeInSec2 * (1/(self.m_HXC_b * self.c_b) * (sum(self.m.Z_HP_HGCHXC_2_H_I[h,i] * self.mdot_HP_b_H[h] * self.c_b for h in self.H) - sum(self.m.Z_HXC_HXC_b_I[h,i]* self.mdot_HP_b_H[h] * self.c_b for h in self.H))) + self.StepSizeInSec2 * self.alpha_HXC_time * (self.t_default - self.m.T_HXC_b_I[i+1])/(self.m_HXC_b * self.c_b)) ## General energy flow

        for i in self.I[1:]:    
            self.m.Constraint_HXC_I.add(self.m.T_HXC_b_I[i] <= self.T_HXC_max + self.m.S_T_HXC_b_I[i]) ## Temperature range tank
            self.m.Constraint_HXC_I.add(self.m.T_HXC_b_I[i] >= self.T_HXC_min - self.m.S_T_HXC_b_I[i]) ## Temperature range tank  

        ## HXC
        for i in self.I[:-1]:
            self.m.Constraint_HXC_I.add(0 == self.m.Z_HXC_HXC_w_I[i] * self.mdot_VP_tot * self.c_w + self.m.Qdot_HXC_w_b_I[i]) 
            self.m.Constraint_HXC_I.add(0 == sum(self.m.Z_HXC_HXC_b_I[h,i]* self.mdot_HP_b_H[h] * self.c_b for h in self.H) - self.m.Qdot_HXC_w_b_I[i])
            self.m.Constraint_HXC_I.add(self.m.Qdot_HXC_w_b_I[i] == ((self.m.T_HXC_w_I[i+1] + self.m.T_HXC_w_out_I[i])/2 - (self.m.T_HXC_b_I[i+1] + self.m.T_HXC_b_out_I[i])/2) * self.a_HXC_w_b * self.alpha_HXC_w_b) 

        for i in self.I[:-1]:    
            self.m.Constraint_HXC_I.add(self.m.T_HXC_I[i] == (self.m.T_HXC_w_I[i] + self.m.T_HXC_w_out_I[i] + self.m.T_HXC_b_I[i] + self.m.T_HXC_b_out_I[i])/4)
        
        for i in self.I[1:-1]:
            self.m.Constraint_HXC_I.add(self.m.T_HXC_I[i] <= self.T_HXC_max + self.m.S_T_HXC_I[i]) 
            self.m.Constraint_HXC_I.add(self.m.T_HXC_I[i] >= self.T_HXC_min - self.m.S_T_HXC_I[i]) 

        ## HGS
        self.m.Constraint_HGS_I = pyo.ConstraintList()
        self.m.Constraint_HGS_I.add(self.m.T_HGS_I[0] == self.T_HGS_start) ## Start temperature

        for i in self.I[0:-1]:
            self.m.Constraint_HGS_I.add(self.m.T_HGS_I[i+1] == self.m.T_HGS_I[i] + self.StepSizeInSec2 * (1/(self.m_HGS_w * self.c_w) * (self.c_w * self.mdot_IS_w * self.m.W_IS_HGS_I[i] + self.c_w * self.mdot_GS_w * self.m.W_GS_HGS_I[i] + self.c_w * self.mdot_VP_tot * self.m.W_HXC_HGS_I[i])) + self.StepSizeInSec2 * self.alpha_HGS_time * (self.t_default - self.m.T_HGS_I[i+1])/(self.m_HGS_w * self.c_w)) ## General energy flow

        for i in self.I[1:]:    
            self.m.Constraint_HGS_I.add(self.m.T_HGS_I[i] <= self.T_HGS_max + self.m.S_T_HGS_I[i]) ## Temperature range tank
            self.m.Constraint_HGS_I.add(self.m.T_HGS_I[i] >= self.T_HGS_min - self.m.S_T_HGS_I[i]) ## Temperature range tank  

        ## VP_HXC_HGS_CS_RLTS
        self.m.Constraint_VP_I = pyo.ConstraintList()
        for i in self.I[0:-1]:
            self.m.Constraint_VP_I.add(self.m.W_HXC_HGS_I[i] >= sum(self.V_HXC_HGS_min * self.m.Z_HXC_HGS_N_I[n,i] + self.T_HXC_HGS_min_N[n] * self.m.V_HXC_HGS_N_I[n,i] - self.V_HXC_HGS_min * self.T_HXC_HGS_min_N[n] * self.m.B_T_HXC_HGS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_VP_I.add(self.m.W_HXC_HGS_I[i] >= sum(self.V_HXC_HGS_max * self.m.Z_HXC_HGS_N_I[n,i] + self.T_HXC_HGS_max_N[n] * self.m.V_HXC_HGS_N_I[n,i] - self.V_HXC_HGS_max * self.T_HXC_HGS_max_N[n] * self.m.B_T_HXC_HGS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_VP_I.add(self.m.W_HXC_HGS_I[i] <= sum(self.V_HXC_HGS_max * self.m.Z_HXC_HGS_N_I[n,i] + self.T_HXC_HGS_min_N[n] * self.m.V_HXC_HGS_N_I[n,i] - self.V_HXC_HGS_max * self.T_HXC_HGS_min_N[n] * self.m.B_T_HXC_HGS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_VP_I.add(self.m.W_HXC_HGS_I[i] <= sum(self.V_HXC_HGS_min * self.m.Z_HXC_HGS_N_I[n,i] + self.T_HXC_HGS_max_N[n] * self.m.V_HXC_HGS_N_I[n,i] - self.V_HXC_HGS_min * self.T_HXC_HGS_max_N[n] * self.m.B_T_HXC_HGS_N_I[n,i] for n in self.N_MC))

            self.m.Constraint_VP_I.add(self.m.T_HXC_HGS_I[i] == sum(self.m.Z_HXC_HGS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_VP_I.add(self.m.V_HXC_HGS_I[i] == sum(self.m.V_HXC_HGS_N_I[n,i] for n in self.N_MC))

        for i in self.I[0:-1]:
            self.m.Constraint_VP_I.add(sum(self.m.B_T_HXC_HGS_N_I[n,i] for n in self.N_MC) == 1)

        for i in self.I[0:-1]:
            for n in self.N_MC: 
                self.m.Constraint_VP_I.add(self.m.V_HXC_HGS_N_I[n,i]  <= self.V_HXC_HGS_max * self.m.B_T_HXC_HGS_N_I[n,i])
                self.m.Constraint_VP_I.add(self.m.V_HXC_HGS_N_I[n,i]  >= self.V_HXC_HGS_min * self.m.B_T_HXC_HGS_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:                
                self.m.Constraint_VP_I.add(self.m.Z_HXC_HGS_N_I[n,i] <= self.T_HXC_HGS_max_N[n] * self.m.B_T_HXC_HGS_N_I[n,i])
                self.m.Constraint_VP_I.add(self.m.Z_HXC_HGS_N_I[n,i] >= self.T_HXC_HGS_min_N[n] * self.m.B_T_HXC_HGS_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_VP_I.add(self.m.Z_HXC_HGS_N_I[n,i] <= self.T_upper_MC) ## Big M constraint input
                self.m.Constraint_VP_I.add(self.m.Z_HXC_HGS_N_I[n,i] >= self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_VP_I.add(self.m.Z_HXC_HGS_N_I[n,i] <= self.T_upper_MC * self.m.B_T_HXC_HGS_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_VP_I.add(self.m.Z_HXC_HGS_N_I[n,i] >= self.T_lower_MC * self.m.B_T_HXC_HGS_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_VP_I.add(self.m.Z_HXC_HGS_N_I[n,i] <= self.m.T_HXC_HGS_N_I[n,i] - (1 - self.m.B_T_HXC_HGS_N_I[n,i]) * self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_VP_I.add(self.m.Z_HXC_HGS_N_I[n,i] >= self.m.T_HXC_HGS_N_I[n,i] - (1 - self.m.B_T_HXC_HGS_N_I[n,i]) * self.T_upper_MC) ## Big M constraint input

        for i in self.I[0:-(self.ControlPeriod3)]:
            if i%self.ControlPeriod3 == 0:
                for j in range(1,self.ControlPeriod3):
                    self.m.Constraint_VP_I.add(self.m.V_HXC_HGS_I[i] == self.m.V_HXC_HGS_I[i+j])
        
        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_VP_I.add(self.m.T_HXC_HGS_N_I[n,i] == self.m.T_HXC_w_I[i+1] - self.m.T_HGS_I[i+1])

        for i in self.I[0:-1]:
            self.m.Constraint_VP_I.add(self.m.W_HXC_CS_I[i] >= sum(self.V_HXC_CS_min * self.m.Z_HXC_CS_N_I[n,i] + self.T_HXC_CS_min_N[n] * self.m.V_HXC_CS_N_I[n,i] - self.V_HXC_CS_min * self.T_HXC_CS_min_N[n] * self.m.B_T_HXC_CS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_VP_I.add(self.m.W_HXC_CS_I[i] >= sum(self.V_HXC_CS_max * self.m.Z_HXC_CS_N_I[n,i] + self.T_HXC_CS_max_N[n] * self.m.V_HXC_CS_N_I[n,i] - self.V_HXC_CS_max * self.T_HXC_CS_max_N[n] * self.m.B_T_HXC_CS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_VP_I.add(self.m.W_HXC_CS_I[i] <= sum(self.V_HXC_CS_max * self.m.Z_HXC_CS_N_I[n,i] + self.T_HXC_CS_min_N[n] * self.m.V_HXC_CS_N_I[n,i] - self.V_HXC_CS_max * self.T_HXC_CS_min_N[n] * self.m.B_T_HXC_CS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_VP_I.add(self.m.W_HXC_CS_I[i] <= sum(self.V_HXC_CS_min * self.m.Z_HXC_CS_N_I[n,i] + self.T_HXC_CS_max_N[n] * self.m.V_HXC_CS_N_I[n,i] - self.V_HXC_CS_min * self.T_HXC_CS_max_N[n] * self.m.B_T_HXC_CS_N_I[n,i] for n in self.N_MC))

            self.m.Constraint_VP_I.add(self.m.T_HXC_CS_I[i] == sum(self.m.Z_HXC_CS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_VP_I.add(self.m.V_HXC_CS_I[i] == sum(self.m.V_HXC_CS_N_I[n,i] for n in self.N_MC))

        for i in self.I[0:-1]:
            self.m.Constraint_VP_I.add(sum(self.m.B_T_HXC_CS_N_I[n,i] for n in self.N_MC) == 1)

        for i in self.I[0:-1]:
            for n in self.N_MC: 
                self.m.Constraint_VP_I.add(self.m.V_HXC_CS_N_I[n,i]  <= self.V_HXC_CS_max * self.m.B_T_HXC_CS_N_I[n,i])
                self.m.Constraint_VP_I.add(self.m.V_HXC_CS_N_I[n,i]  >= self.V_HXC_CS_min * self.m.B_T_HXC_CS_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:                
                self.m.Constraint_VP_I.add(self.m.Z_HXC_CS_N_I[n,i] <= self.T_HXC_CS_max_N[n] * self.m.B_T_HXC_CS_N_I[n,i])
                self.m.Constraint_VP_I.add(self.m.Z_HXC_CS_N_I[n,i] >= self.T_HXC_CS_min_N[n] * self.m.B_T_HXC_CS_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_VP_I.add(self.m.Z_HXC_CS_N_I[n,i] <= self.T_upper_MC) ## Big M constraint input
                self.m.Constraint_VP_I.add(self.m.Z_HXC_CS_N_I[n,i] >= self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_VP_I.add(self.m.Z_HXC_CS_N_I[n,i] <= self.T_upper_MC * self.m.B_T_HXC_CS_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_VP_I.add(self.m.Z_HXC_CS_N_I[n,i] >= self.T_lower_MC * self.m.B_T_HXC_CS_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_VP_I.add(self.m.Z_HXC_CS_N_I[n,i] <= self.m.T_HXC_CS_N_I[n,i] - (1 - self.m.B_T_HXC_CS_N_I[n,i]) * self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_VP_I.add(self.m.Z_HXC_CS_N_I[n,i] >= self.m.T_HXC_CS_N_I[n,i] - (1 - self.m.B_T_HXC_CS_N_I[n,i]) * self.T_upper_MC) ## Big M constraint input

        for i in self.I[0:-(self.ControlPeriod3)]:
            if i%self.ControlPeriod3 == 0:
                for j in range(1,self.ControlPeriod3):
                    self.m.Constraint_VP_I.add(self.m.V_HXC_CS_I[i] == self.m.V_HXC_CS_I[i+j])
        
        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_VP_I.add(self.m.T_HXC_CS_N_I[n,i] == self.m.T_HXC_w_I[i+1] - self.m.T_CS_I[i+1])

        for i in self.I[0:-1]:
            self.m.Constraint_VP_I.add(self.m.W_HXC_RLTS_I[i] >= sum(self.V_HXC_RLTS_min * self.m.Z_HXC_RLTS_N_I[n,i] + self.T_HXC_RLTS_min_N[n] * self.m.V_HXC_RLTS_N_I[n,i] - self.V_HXC_RLTS_min * self.T_HXC_RLTS_min_N[n] * self.m.B_T_HXC_RLTS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_VP_I.add(self.m.W_HXC_RLTS_I[i] >= sum(self.V_HXC_RLTS_max * self.m.Z_HXC_RLTS_N_I[n,i] + self.T_HXC_RLTS_max_N[n] * self.m.V_HXC_RLTS_N_I[n,i] - self.V_HXC_RLTS_max * self.T_HXC_RLTS_max_N[n] * self.m.B_T_HXC_RLTS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_VP_I.add(self.m.W_HXC_RLTS_I[i] <= sum(self.V_HXC_RLTS_max * self.m.Z_HXC_RLTS_N_I[n,i] + self.T_HXC_RLTS_min_N[n] * self.m.V_HXC_RLTS_N_I[n,i] - self.V_HXC_RLTS_max * self.T_HXC_RLTS_min_N[n] * self.m.B_T_HXC_RLTS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_VP_I.add(self.m.W_HXC_RLTS_I[i] <= sum(self.V_HXC_RLTS_min * self.m.Z_HXC_RLTS_N_I[n,i] + self.T_HXC_RLTS_max_N[n] * self.m.V_HXC_RLTS_N_I[n,i] - self.V_HXC_RLTS_min * self.T_HXC_RLTS_max_N[n] * self.m.B_T_HXC_RLTS_N_I[n,i] for n in self.N_MC))

            self.m.Constraint_VP_I.add(self.m.T_HXC_RLTS_I[i] == sum(self.m.Z_HXC_RLTS_N_I[n,i] for n in self.N_MC))
            self.m.Constraint_VP_I.add(self.m.V_HXC_RLTS_I[i] == sum(self.m.V_HXC_RLTS_N_I[n,i] for n in self.N_MC))

        for i in self.I[0:-1]:
            self.m.Constraint_VP_I.add(sum(self.m.B_T_HXC_RLTS_N_I[n,i] for n in self.N_MC) == 1)

        for i in self.I[0:-1]:
            for n in self.N_MC: 
                self.m.Constraint_VP_I.add(self.m.V_HXC_RLTS_N_I[n,i]  <= self.V_HXC_RLTS_max * self.m.B_T_HXC_RLTS_N_I[n,i])
                self.m.Constraint_VP_I.add(self.m.V_HXC_RLTS_N_I[n,i]  >= self.V_HXC_RLTS_min * self.m.B_T_HXC_RLTS_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:                
                self.m.Constraint_VP_I.add(self.m.Z_HXC_RLTS_N_I[n,i] <= self.T_HXC_RLTS_max_N[n] * self.m.B_T_HXC_RLTS_N_I[n,i])
                self.m.Constraint_VP_I.add(self.m.Z_HXC_RLTS_N_I[n,i] >= self.T_HXC_RLTS_min_N[n] * self.m.B_T_HXC_RLTS_N_I[n,i])

        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_VP_I.add(self.m.Z_HXC_RLTS_N_I[n,i] <= self.T_upper_MC) ## Big M constraint input
                self.m.Constraint_VP_I.add(self.m.Z_HXC_RLTS_N_I[n,i] >= self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_VP_I.add(self.m.Z_HXC_RLTS_N_I[n,i] <= self.T_upper_MC * self.m.B_T_HXC_RLTS_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_VP_I.add(self.m.Z_HXC_RLTS_N_I[n,i] >= self.T_lower_MC * self.m.B_T_HXC_RLTS_N_I[n,i]) ## Big M constraint input
                self.m.Constraint_VP_I.add(self.m.Z_HXC_RLTS_N_I[n,i] <= self.m.T_HXC_RLTS_N_I[n,i] - (1 - self.m.B_T_HXC_RLTS_N_I[n,i]) * self.T_lower_MC) ## Big M constraint input
                self.m.Constraint_VP_I.add(self.m.Z_HXC_RLTS_N_I[n,i] >= self.m.T_HXC_RLTS_N_I[n,i] - (1 - self.m.B_T_HXC_RLTS_N_I[n,i]) * self.T_upper_MC) ## Big M constraint input

        for i in self.I[0:-(self.ControlPeriod3)]:
            if i%self.ControlPeriod3 == 0:
                for j in range(1,self.ControlPeriod3):
                    self.m.Constraint_VP_I.add(self.m.V_HXC_RLTS_I[i] == self.m.V_HXC_RLTS_I[i+j])
        
        for i in self.I[0:-1]:
            for n in self.N_MC:
                self.m.Constraint_VP_I.add(self.m.T_HXC_RLTS_N_I[n,i] == self.m.T_HXC_w_I[i+1] - self.m.T_RLTS_I[i+1])

        for i in self.I[0:-1]:
            self.m.Constraint_VP_I.add(self.m.V_HXC_HGS_I[i] + self.m.V_HXC_CS_I[i] + self.m.V_HXC_RLTS_I[i] <= 1) ## Maximum of 1 for all ways

        for i in self.I[:-2]:
            self.m.Constraint_VP_I.add(self.m.Z_VP_I[i] >= self.m.V_HXC_HGS_I[i] - self.m.V_HXC_HGS_I[i+1])
            self.m.Constraint_VP_I.add(self.m.Z_VP_I[i] >= self.m.V_HXC_HGS_I[i+1] - self.m.V_HXC_HGS_I[i])
            self.m.Constraint_VP_I.add(self.m.Z_VP_I[i] >= self.m.V_HXC_CS_I[i] - self.m.V_HXC_CS_I[i+1])
            self.m.Constraint_VP_I.add(self.m.Z_VP_I[i] >= self.m.V_HXC_CS_I[i+1] - self.m.V_HXC_CS_I[i])
            self.m.Constraint_VP_I.add(self.m.Z_VP_I[i] >= self.m.V_HXC_RLTS_I[i] - self.m.V_HXC_RLTS_I[i+1])
            self.m.Constraint_VP_I.add(self.m.Z_VP_I[i] >= self.m.V_HXC_RLTS_I[i+1] - self.m.V_HXC_RLTS_I[i])
            self.m.Constraint_VP_I.add(self.m.Z_VP_I[i] <= 1)

        for i in self.I[0:-1]:
            self.m.Constraint_VP_I.add(self.m.E_VP_EL_I[i] == (self.m.V_HXC_HGS_I[i] + self.m.V_HXC_CS_I[i] + self.m.V_HXC_RLTS_I[i]) * self.e_VP_EL)

        for i in self.I[:-1]: 
            self.m.Constraint_VP_I.add(self.m.Z_HXC_HXC_w_I[i] <= self.T_upper_MC) ## Big M constraint input
            self.m.Constraint_VP_I.add(self.m.Z_HXC_HXC_w_I[i] >= self.T_lower_MC) ## Big M constraint input
            self.m.Constraint_VP_I.add(self.m.Z_HXC_HXC_w_I[i] <= self.T_upper_MC * sum(self.m.B_HP_H_I[h,i] * self.mdot_digit_HP_H[h] for h in self.H)) ## Big M constraint input
            self.m.Constraint_VP_I.add(self.m.Z_HXC_HXC_w_I[i] >= self.T_lower_MC * sum(self.m.B_HP_H_I[h,i] * self.mdot_digit_HP_H[h] for h in self.H)) ## Big M constraint input
            self.m.Constraint_VP_I.add(self.m.Z_HXC_HXC_w_I[i] <= (self.m.T_HXC_w_I[i+1] - self.m.T_HXC_w_out_I[i]) - (1 - sum(self.m.B_HP_H_I[h,i] * self.mdot_digit_HP_H[h] for h in self.H)) * self.T_lower_MC) ## Big M constraint input
            self.m.Constraint_VP_I.add(self.m.Z_HXC_HXC_w_I[i] >= (self.m.T_HXC_w_I[i+1] - self.m.T_HXC_w_out_I[i]) - (1 - sum(self.m.B_HP_H_I[h,i] * self.mdot_digit_HP_H[h] for h in self.H)) * self.T_upper_MC) ## Big M constraint input

        return self.m

    def setObjective(self,model):
        self.m = model
        self.m.OBJ = pyo.Objective(expr = self.m.C_TOT_I_ + self.m.S_TOT_I_ + self.m.T_TOT_I_)
        return self.m

    def setWarmstart(self,model,available=True,file=None):
        self.m = model
        if available ==True:
            self.warmstart_available = True
            self.dFwarmStart = file
            for i in self.I[0:-1]:
                for h in self.H:
                    self.m.B_HP_H_I[h,i] = self.dFwarmStart[("B_HP_"+str(h)+"_I")][i]
            for i in self.I[0:-1]:
                for n in self.N_MC:
                    self.m.B_T_HXAR_N_I[n,i] = self.dFwarmStart[("B_T_HXAR_"+str(n)+"_I")][i]
            for i in self.I[0:-1]:
                for n in self.N_MC:
                    self.m.B_T_HP_HXH_N_I[n,i] = self.dFwarmStart[("B_T_HP_HXH_"+str(n)+"_I")][i]
            for i in self.I[0:-1]:
                for n in self.N_MC:
                    self.m.B_T_HP_HS_N_I[n,i] = self.dFwarmStart[("B_T_HP_HS_"+str(n)+"_I")][i]
            for i in self.I[0:-1]:
                for n in self.N_MC:
                    self.m.B_T_HXA_HXH_N_I[n,i] = self.dFwarmStart[("B_T_HXA_HXH_"+str(n)+"_I")][i]
            for i in self.I[0:-1]:
                for n in self.N_MC:
                    self.m.B_T_HXA_HGC_N_I[n,i] = self.dFwarmStart[("B_T_HXA_HGC_"+str(n)+"_I")][i]                
            for i in self.I[0:-1]:
                for n in self.N_MC:
                    self.m.B_T_HS_IS_N_I[n,i] = self.dFwarmStart[("B_T_HS_IS_"+str(n)+"_I")][i]
            for i in self.I[0:-1]:
                for n in self.N_MC:
                    self.m.B_T_HS_IS_N_I_2[n,i] = self.dFwarmStart[("B_T_HS_IS_"+str(n)+"_I_2")][i]
            for i in self.I[0:-1]:
                for n in self.N_MC:
                    self.m.B_T_HP_HGC_N_I[n,i] = self.dFwarmStart[("B_T_HP_HGC_"+str(n)+"_I")][i]
            for i in self.I[0:-1]:
                for n in self.N_MC:
                    self.m.B_T_HP_HGCHXC_N_I[n,i] = self.dFwarmStart[("B_T_HP_HGCHXC_"+str(n)+"_I")][i]
            for i in self.I[0:-1]:
                for n in self.N_MC:
                    self.m.B_T_HP_HGCHXC_N_2_I[n,i] = self.dFwarmStart[("B_T_HP_HGCHXC_"+str(n)+"_2_I")][i]
            for i in self.I[0:-1]:
                for n in self.N_MC:
                    self.m.B_T_IS_HGS_N_I[n,i] = self.dFwarmStart[("B_T_IS_HGS_"+str(n)+"_I")][i]
            for i in self.I[0:-1]:
                for n in self.N_MC:
                    self.m.B_T_IS_HGS_N_I_2[n,i] = self.dFwarmStart[("B_T_IS_HGS_"+str(n)+"_I_2")][i]
            for i in self.I[0:-1]:
                for n in self.N_MC:
                    self.m.B_T_GS_HGS_N_I[n,i] = self.dFwarmStart[("B_T_GS_HGS_"+str(n)+"_I")][i]
            for i in self.I[0:-1]:
                for n in self.N_MC:
                    self.m.B_T_GS_CS_N_I[n,i] = self.dFwarmStart[("B_T_GS_CS_"+str(n)+"_I")][i]
            for i in self.I[0:-1]:
                for n in self.N_MC:
                    self.m.B_T_GS_HGS_N_I_2[n,i] = self.dFwarmStart[("B_T_GS_HGS_"+str(n)+"_I_2")][i]
            for i in self.I[0:-1]:
                for n in self.N_MC:
                    self.m.B_T_GS_CS_N_I_2[n,i] = self.dFwarmStart[("B_T_GS_CS_"+str(n)+"_I_2")][i]
            for i in self.I[0:-1]:
                for n in self.N_MC:
                    self.m.B_T_HXC_HGS_N_I[n,i] = self.dFwarmStart[("B_T_HXC_HGS_"+str(n)+"_I")][i]
            for i in self.I[0:-1]:
                for n in self.N_MC:
                    self.m.B_T_HXC_CS_N_I[n,i] = self.dFwarmStart[("B_T_HXC_CS_"+str(n)+"_I")][i]
            for i in self.I[0:-1]:
                for n in self.N_MC:
                    self.m.B_T_HXC_RLTS_N_I[n,i] = self.dFwarmStart[("B_T_HXC_RLTS_"+str(n)+"_I")][i]
        else:
            self.warmstart_available = False
        
        return self.m


    def setSolverAndRunOptimization(self,solver = 0, showSolverOutput = 0, writeILP = 0, writeMPSfile = 0):
        if writeMPSfile == 1:
            self.m.write(filename = "WB.mps", io_options = {"symbolic_solver_labels":True})

        if solver == 0:
            self.opt = pyo.SolverFactory('gurobi', solver_io="python")
            self.opt.options['TimeLimit'] = 180
            self.opt.options['threads'] = 8
            #self.opt.options['MIPFocus'] = 1
            #self.opt.options['ObjBound'] = 50
            #self.opt.options['Cutoff'] = 500
            if writeILP == 1:
                self.opt.options['resultFile'] = 'test.ilp'
        elif solver == 1:
            self.opt = pyo.SolverFactory('cbc')
            self.opt.options['Sec'] = 180
        elif solver == 2:
            self.opt = pyo.SolverFactory('glpk')
            self.opt.options['tmlim'] = 200
            #self.opt.options['mipgap'] = 1e-6 # not needed atm
        
        if writeILP == 1:
            self.results = self.opt.solve(self.m,warmstart=self.warmstart_available,tee=True,symbolic_solver_labels=True) 
        else:
            self.results = self.opt.solve(self.m,warmstart=self.warmstart_available,tee=True)

        if showSolverOutput == 1:
            print(self.results)

    def getResults(self,model,source=None,savePath="",singleFile=False):
        self.m = model
        self.safeFile = pd.DataFrame(columns = ["C_OP_I","C_HP_I","C_HXA_I","C_IS_I","C_GS_I","C_VP_I","B_HP_4_I","B_HP_3_I","B_HP_2_I","B_HP_1_I","B_HP_0_I","P_HP_HXH_I","P_HP_HS_I","P_HP_HGC_H_I","P_HP_HGCHXC_H_I","P_HP_HGC_2_H_I","P_HP_HGCHXC_2_H_I","P_HXA_I","P_HXA_HXH_I","P_HXA_HGC_I",
        "P_HS_IS_I","P_IS_HGS_I","P_GS_HGS_I","P_GS_CS_I","P_VP_HGS_I","P_VP_CS_I","P_VP_RLTS_I","E_HP_EL_I","Q_HP_HT_I","Q_HP_LT_I","E_HP_EL_in_I","T_HP_HT_I","T_HP_LT_I","T_HS_I","T_HXH_I","T_HXH_b_I","T_HXH_w_I",
        "T_HGC_I","T_HXA_I","T_HXC_I","T_HXC_b_I","T_HXC_w_I","T_IS_I","T_ISw_I","T_ISc_I","T_IS_W_0_I","T_IS_W_1_I","T_IS_W_2_I","T_IS_C_0_I","T_IS_C_1_I","T_IS_C_2_I","T_IS_C_3_I","T_IS_C_4_I",
        "T_GS_I","T_GSw_I","T_GSc_I","T_GS_W_0_I","T_GS_W_1_I","T_GS_W_2_I","T_GS_C_0_I","T_GS_C_1_I","T_GS_C_2_I","T_GS_C_3_I","T_GS_C_4_I","T_GS_C_5_I","T_GS_C_6_I",
        "T_HGS_I","T_CS_I","T_RLTS_I","S_OP_I","S_T_HP_I","S_T_HS_I","S_T_HXH_I","S_T_HGC_I","S_T_HXA_I","S_T_HXC_I","S_T_IS_I","S_T_GS_I","S_T_HGS_I",
        "S_T_CS_I","S_T_RLTS_I","q_dem_HS_I","q_dem_CS_I","q_dem_RLTS_I","temp_amb_I","B_T_HXAR_0_I","B_T_HXAR_1_I","B_T_HP_HXH_0_I","B_T_HP_HXH_1_I","B_T_HP_HS_0_I","B_T_HP_HS_1_I","B_T_HXA_HXH_0_I","B_T_HXA_HXH_1_I",
        "B_T_HXA_HGC_0_I","B_T_HXA_HGC_1_I","B_T_HS_IS_0_I","B_T_HS_IS_1_I","B_T_HS_IS_0_I_2","B_T_HS_IS_1_I_2","B_T_HP_HGC_0_I","B_T_HP_HGC_1_I","B_T_HP_HGCHXC_0_I","B_T_HP_HGCHXC_1_I","B_T_HP_HGCHXC_0_2_I","B_T_HP_HGCHXC_1_2_I",
        "B_T_IS_HGS_0_I","B_T_IS_HGS_1_I","B_T_IS_HGS_0_I_2","B_T_IS_HGS_1_I_2","B_T_GS_HGS_0_I","B_T_GS_HGS_1_I","B_T_GS_CS_0_I","B_T_GS_CS_1_I","B_T_GS_HGS_0_I_2","B_T_GS_HGS_1_I_2","B_T_GS_CS_0_I_2","B_T_GS_CS_1_I_2",
        "B_T_HXC_HGS_0_I","B_T_HXC_HGS_1_I","B_T_HXC_CS_0_I","B_T_HXC_CS_1_I","B_T_HXC_RLTS_0_I","B_T_HXC_RLTS_1_I"]) 
        try: # Everything but slack constraints
            self.safeFile = self.safeFile.append({"C_OP_I":self.m.C_OP_I[0](),"C_HP_I":self.m.E_HP_EL_in_I[0]()*self.c_ELECTRICITY_buy_I[0]*self.StepSizeInSec2/self.t_hour_in_sec,
            "C_HXA_I":self.m.E_HXA_EL_I[0]()*self.c_ELECTRICITY_buy_I[0]*self.StepSizeInSec2/self.t_hour_in_sec,"C_IS_I":self.m.E_IS_EL_I[0]()*self.c_ELECTRICITY_buy_I[0]*self.StepSizeInSec2/self.t_hour_in_sec,
            "C_GS_I":self.m.E_GS_EL_I[0]()*self.c_ELECTRICITY_buy_I[0]*self.StepSizeInSec2/self.t_hour_in_sec,"C_VP_I":self.m.E_VP_EL_I[0]()*self.c_ELECTRICITY_buy_I[0]*self.StepSizeInSec2/self.t_hour_in_sec,
            "B_HP_4_I":self.m.B_HP_H_I[4 ,0](),"B_HP_3_I":self.m.B_HP_H_I[3 ,0](),"B_HP_2_I":self.m.B_HP_H_I[2 ,0](),"B_HP_1_I":self.m.B_HP_H_I[1 ,0](),"B_HP_0_I":self.m.B_HP_H_I[0 ,0](),
            "P_HP_HXH_I":(self.c_w * sum(self.mdot_HP_w_H[h] * self.m.Z_HP_HXH_H_I[h,0]() for h in self.H)),"P_HP_HS_I":(self.c_w * sum(self.mdot_HP_w_H[h] * self.m.Z_HP_HS_H_I[h,0]() for h in self.H)),"P_HP_HGC_H_I":(self.c_b * sum(self.mdot_HP_b_H[h] * self.m.Z_HP_HGC_H_I[h,0]() for h in self.H)),
            "P_HP_HGCHXC_H_I":(self.c_b * sum(self.mdot_HP_b_H[h] * self.m.Z_HP_HGCHXC_H_I[h,0]() for h in self.H)),"P_HP_HGC_2_H_I":(self.c_b * sum(self.mdot_HP_b_H[h] * self.m.Z_HP_HGC_2_H_I[h,0]() for h in self.H)),"P_HP_HGCHXC_2_H_I":(self.c_b * sum(self.mdot_HP_b_H[h] * self.m.Z_HP_HGCHXC_2_H_I[h,0]() for h in self.H)),"P_HXA_I":(self.alpha_factor_HXA * self.c_a * self.mdot_HXA_a * self.m.W_HXA_I[0]()),
            "P_HXA_HXH_I":(self.m.W_HXA_HXH_I[0]() * self.mdot_HXA_b * self.c_b),"P_HXA_HGC_I":(self.m.W_HXA_HGC_I[0]() * self.c_b * self.mdot_HXA_b),"P_HS_IS_I":(self.m.W_HS_IS_I[0]() * self.c_w * self.mdot_IS_w),
            "P_IS_HGS_I":(self.m.W_IS_HGS_I[0]() * self.c_w * self.mdot_IS_w),"P_GS_HGS_I":(self.m.W_GS_HGS_I[0]() * self.c_w * self.mdot_GS_w),"P_GS_CS_I":(self.m.W_GS_CS_I[0]() * self.c_w * self.mdot_GS_w),
            "P_VP_HGS_I":(self.m.W_HXC_HGS_I[0]() * self.mdot_VP_tot * self.c_w),"P_VP_CS_I":(self.m.W_HXC_CS_I[0]() * self.mdot_VP_tot * self.c_w),"P_VP_RLTS_I":(self.m.W_HXC_RLTS_I[0]() * self.mdot_VP_tot * self.c_w),
            "E_HP_EL_I":self.m.E_HP_EL_I[0](),"Q_HP_HT_I":self.m.Q_HP_HT_I[0](),"Q_HP_LT_I":self.m.Q_HP_LT_I[0](),"E_HP_EL_in_I":self.m.E_HP_EL_in_I[0](),"T_HP_HT_I":self.m.T_HP_HT_I[0](),
            "T_HP_LT_I":self.m.T_HP_LT_I[0](),"T_HS_I":self.m.T_HS_I[0](),"T_HXH_I":self.m.T_HXH_I[0](),"T_HXH_b_I":self.m.T_HXH_b_I[0](),"T_HXH_w_I":self.m.T_HXH_w_I[0](),"T_HGC_I":self.m.T_HGC_I[0](),
            "T_HXA_I":self.m.T_HXA_I[0](),"T_HXC_I":self.m.T_HXC_I[0](),"T_HXC_b_I":self.m.T_HXC_b_I[0](),"T_HXC_w_I":self.m.T_HXC_w_I[0](),"T_IS_I":self.m.T_IS_I[0](),"T_ISw_I":(sum(self.m.T_IS_W_I_WR[0,r]() for r in self.wr_IS)/len(self.wr_IS)),"T_ISc_I":(sum(self.m.T_IS_C_I_CR[0,r]() for r in self.cr_IS)/len(self.cr_IS)),
            "T_IS_W_0_I":self.m.T_IS_W_I_WR[0,0](),"T_IS_W_1_I":self.m.T_IS_W_I_WR[0,2](),"T_IS_W_2_I":self.m.T_IS_W_I_WR[0,4](),"T_IS_C_0_I":self.m.T_IS_C_I_CR[0,0](),"T_IS_C_1_I":self.m.T_IS_C_I_CR[0,1](),"T_IS_C_2_I":self.m.T_IS_C_I_CR[0,2](),"T_IS_C_3_I":self.m.T_IS_C_I_CR[0,3](),"T_IS_C_4_I":self.m.T_IS_C_I_CR[0,4](),
            "T_GS_I":self.m.T_GS_I[0](),"T_GSw_I":(sum(sum(self.m.T_GS_W_I_WR_WC[0,c,r]() for c in self.wc_GS) for r in self.wr_GS)/(len(self.wc_GS)*len(self.wr_GS))),"T_GSc_I":(sum(sum(self.m.T_GS_C_I_CR_CC[0,c,r]() for r in self.cr_GS) for c in self.cc_GS)/(len(self.cr_GS)*len(self.cc_GS))),
            "T_GS_W_0_I":sum(self.m.T_GS_W_I_WR_WC[0,c,1]() for c in self.wc_GS),"T_GS_W_1_I":sum(self.m.T_GS_W_I_WR_WC[0,c,3]() for c in self.wc_GS),"T_GS_W_2_I":sum(self.m.T_GS_W_I_WR_WC[0,c,5]() for c in self.wc_GS),
            "T_GS_C_0_I":sum(self.m.T_GS_C_I_CR_CC[0,c,0]() for c in self.cc_GS),"T_GS_C_1_I":sum(self.m.T_GS_C_I_CR_CC[0,c,1]() for c in self.cc_GS),"T_GS_C_2_I":sum(self.m.T_GS_C_I_CR_CC[0,c,2]() for c in self.cc_GS),"T_GS_C_3_I":sum(self.m.T_GS_C_I_CR_CC[0,c,3]() for c in self.cc_GS),"T_GS_C_4_I":sum(self.m.T_GS_C_I_CR_CC[0,c,4]() for c in self.cc_GS),"T_GS_C_5_I":sum(self.m.T_GS_C_I_CR_CC[0,c,5]() for c in self.cc_GS),"T_GS_C_6_I":sum(self.m.T_GS_C_I_CR_CC[0,c,6]() for c in self.cc_GS),
            "T_HGS_I":self.m.T_HGS_I[0](),"T_CS_I":self.m.T_CS_I[0](),"T_RLTS_I":self.m.T_RLTS_I[0](),"q_dem_HS_I":self.q_dem_HS_I[0],"q_dem_CS_I":self.q_dem_CS_I[0],"q_dem_RLTS_I":self.q_dem_RLTS_I[0],"temp_amb_I":self.temp_amb_I[0],"B_T_HXAR_0_I":self.m.B_T_HXAR_N_I[0,0](),"B_T_HXAR_1_I":self.m.B_T_HXAR_N_I[1,0](),"B_T_HP_HXH_0_I":self.m.B_T_HP_HXH_N_I[0,0](),"B_T_HP_HXH_1_I":self.m.B_T_HP_HXH_N_I[1,0](),
            "B_T_HP_HS_0_I":self.m.B_T_HP_HS_N_I[0,0](),"B_T_HP_HS_1_I":self.m.B_T_HP_HS_N_I[1,0](),"B_T_HXA_HXH_0_I":self.m.B_T_HXA_HXH_N_I[0,0](),"B_T_HXA_HXH_1_I":self.m.B_T_HXA_HXH_N_I[1,0](),"B_T_HXA_HGC_0_I":self.m.B_T_HXA_HGC_N_I[0,0](),"B_T_HXA_HGC_1_I":self.m.B_T_HXA_HGC_N_I[1,0](),"B_T_HS_IS_0_I":self.m.B_T_HS_IS_N_I[0,0](),"B_T_HS_IS_1_I":self.m.B_T_HS_IS_N_I[1,0](),
            "B_T_HS_IS_0_I_2":self.m.B_T_HS_IS_N_I_2[0,0](),"B_T_HS_IS_1_I_2":self.m.B_T_HS_IS_N_I_2[1,0](),"B_T_HP_HGC_0_I":self.m.B_T_HP_HGC_N_I[0,0](),"B_T_HP_HGC_1_I":self.m.B_T_HP_HGC_N_I[1,0](),"B_T_HP_HGCHXC_0_I":self.m.B_T_HP_HGCHXC_N_I[0,0](),"B_T_HP_HGCHXC_1_I":self.m.B_T_HP_HGCHXC_N_I[1,0](),"B_T_HP_HGCHXC_0_2_I":self.m.B_T_HP_HGCHXC_N_2_I[0,0](),"B_T_HP_HGCHXC_1_2_I":self.m.B_T_HP_HGCHXC_N_2_I[1,0](),
            "B_T_IS_HGS_0_I":self.m.B_T_IS_HGS_N_I[0,0](),"B_T_IS_HGS_1_I":self.m.B_T_IS_HGS_N_I[1,0](),"B_T_IS_HGS_0_I_2":self.m.B_T_IS_HGS_N_I_2[0,0](),"B_T_IS_HGS_1_I_2":self.m.B_T_IS_HGS_N_I_2[1,0](),"B_T_GS_HGS_0_I":self.m.B_T_GS_HGS_N_I[0,0](),"B_T_GS_HGS_1_I":self.m.B_T_GS_HGS_N_I[1,0](),"B_T_GS_CS_0_I":self.m.B_T_GS_CS_N_I[0,0](),"B_T_GS_CS_1_I":self.m.B_T_GS_CS_N_I[1,0](),
            "B_T_GS_HGS_0_I_2":self.m.B_T_GS_HGS_N_I_2[0,0](),"B_T_GS_HGS_1_I_2":self.m.B_T_GS_HGS_N_I_2[1,0](),"B_T_GS_CS_0_I_2":self.m.B_T_GS_CS_N_I_2[0,0](),"B_T_GS_CS_1_I_2":self.m.B_T_GS_CS_N_I_2[1,0](),"B_T_HXC_HGS_0_I":self.m.B_T_HXC_HGS_N_I[0,0](),"B_T_HXC_HGS_1_I":self.m.B_T_HXC_HGS_N_I[1,0](),
            "B_T_HXC_CS_0_I":self.m.B_T_HXC_CS_N_I[0,0](),"B_T_HXC_CS_1_I":self.m.B_T_HXC_CS_N_I[1,0](),"B_T_HXC_RLTS_0_I":self.m.B_T_HXC_RLTS_N_I[0,0](),"B_T_HXC_RLTS_1_I":self.m.B_T_HXC_RLTS_N_I[1,0]()}, ignore_index=True)
             # ALL
            for i in self.I[1:-1]:
                self.safeFile = self.safeFile.append({"C_OP_I":self.m.C_OP_I[i](),"C_HP_I":self.m.E_HP_EL_in_I[i]()*self.c_ELECTRICITY_buy_I[i]*self.StepSizeInSec2/self.t_hour_in_sec,
                "C_HXA_I":self.m.E_HXA_EL_I[i]()*self.c_ELECTRICITY_buy_I[i]*self.StepSizeInSec2/self.t_hour_in_sec,"C_IS_I":self.m.E_IS_EL_I[i]()*self.c_ELECTRICITY_buy_I[i]*self.StepSizeInSec2/self.t_hour_in_sec,
                "C_GS_I":self.m.E_GS_EL_I[i]()*self.c_ELECTRICITY_buy_I[i]*self.StepSizeInSec2/self.t_hour_in_sec,"C_VP_I":self.m.E_VP_EL_I[i]()*self.c_ELECTRICITY_buy_I[i]*self.StepSizeInSec2/self.t_hour_in_sec,
                "B_HP_4_I":self.m.B_HP_H_I[4 ,i](),"B_HP_3_I":self.m.B_HP_H_I[3 ,i](),"B_HP_2_I":self.m.B_HP_H_I[2 ,i](),"B_HP_1_I":self.m.B_HP_H_I[1 ,i](),"B_HP_0_I":self.m.B_HP_H_I[0 ,i](),
                "P_HP_HXH_I":(self.c_w * sum(self.mdot_HP_w_H[h] * self.m.Z_HP_HXH_H_I[h,i]() for h in self.H)),"P_HP_HS_I":(self.c_w * sum(self.mdot_HP_w_H[h] * self.m.Z_HP_HS_H_I[h,i]() for h in self.H)),"P_HP_HGC_H_I":(self.c_b * sum(self.mdot_HP_b_H[h] * self.m.Z_HP_HGC_H_I[h,i]() for h in self.H)),
                "P_HP_HGCHXC_H_I":(self.c_b * sum(self.mdot_HP_b_H[h] * self.m.Z_HP_HGCHXC_H_I[h,i]() for h in self.H)),"P_HP_HGC_2_H_I":(self.c_b * sum(self.mdot_HP_b_H[h] * self.m.Z_HP_HGC_2_H_I[h,i]() for h in self.H)),"P_HP_HGCHXC_2_H_I":(self.c_b * sum(self.mdot_HP_b_H[h] * self.m.Z_HP_HGCHXC_2_H_I[h,i]() for h in self.H)),"P_HXA_I":(self.alpha_factor_HXA * self.c_a * self.mdot_HXA_a * self.m.W_HXA_I[i]()),
                "P_HXA_HXH_I":(self.m.W_HXA_HXH_I[i]() * self.mdot_HXA_b * self.c_b),"P_HXA_HGC_I":(self.m.W_HXA_HGC_I[i]() * self.c_b * self.mdot_HXA_b),"P_HS_IS_I":(self.m.W_HS_IS_I[i]() * self.c_w * self.mdot_IS_w),
                "P_IS_HGS_I":(self.m.W_IS_HGS_I[i]() * self.c_w * self.mdot_IS_w),"P_GS_HGS_I":(self.m.W_GS_HGS_I[i]() * self.c_w * self.mdot_GS_w),"P_GS_CS_I":(self.m.W_GS_CS_I[i]() * self.c_w * self.mdot_GS_w),
                "P_VP_HGS_I":(self.m.W_HXC_HGS_I[i]() * self.mdot_VP_tot * self.c_w),"P_VP_CS_I":(self.m.W_HXC_CS_I[i]() * self.mdot_VP_tot * self.c_w),"P_VP_RLTS_I":(self.m.W_HXC_RLTS_I[i]() * self.mdot_VP_tot * self.c_w),
                "E_HP_EL_I":self.m.E_HP_EL_I[i](),"Q_HP_HT_I":self.m.Q_HP_HT_I[i](),"Q_HP_LT_I":self.m.Q_HP_LT_I[i](),"E_HP_EL_in_I":self.m.E_HP_EL_in_I[i](),"T_HP_HT_I":self.m.T_HP_HT_I[i](),
                "T_HP_LT_I":self.m.T_HP_LT_I[i](),"T_HS_I":self.m.T_HS_I[i](),"T_HXH_I":self.m.T_HXH_I[i](),"T_HXH_b_I":self.m.T_HXH_b_I[i](),"T_HXH_w_I":self.m.T_HXH_w_I[i](),"T_HGC_I":self.m.T_HGC_I[i](),
                "T_HXA_I":self.m.T_HXA_I[i](),"T_HXC_I":self.m.T_HXC_I[i](),"T_HXC_b_I":self.m.T_HXC_b_I[i](),"T_HXC_w_I":self.m.T_HXC_w_I[i](),"T_IS_I":self.m.T_IS_I[i](),"T_ISw_I":(sum(self.m.T_IS_W_I_WR[i,r]() for r in self.wr_IS)/len(self.wr_IS)),"T_ISc_I":(sum(self.m.T_IS_C_I_CR[i,r]() for r in self.cr_IS)/len(self.cr_IS)),
                "T_IS_W_0_I":self.m.T_IS_W_I_WR[i,0](),"T_IS_W_1_I":self.m.T_IS_W_I_WR[i,2](),"T_IS_W_2_I":self.m.T_IS_W_I_WR[i,4](),"T_IS_C_0_I":self.m.T_IS_C_I_CR[i,0](),"T_IS_C_1_I":self.m.T_IS_C_I_CR[i,1](),"T_IS_C_2_I":self.m.T_IS_C_I_CR[i,2](),"T_IS_C_3_I":self.m.T_IS_C_I_CR[i,3](),"T_IS_C_4_I":self.m.T_IS_C_I_CR[i,4](),
                "T_GS_I":self.m.T_GS_I[i](),"T_GSw_I":(sum(sum(self.m.T_GS_W_I_WR_WC[i,c,r]() for c in self.wc_GS) for r in self.wr_GS)/(len(self.wc_GS)*len(self.wr_GS))),"T_GSc_I":(sum(sum(self.m.T_GS_C_I_CR_CC[i,c,r]() for r in self.cr_GS) for c in self.cc_GS)/(len(self.cr_GS)*len(self.cc_GS))),
                "T_GS_W_0_I":sum(self.m.T_GS_W_I_WR_WC[i,c,1]() for c in self.wc_GS),"T_GS_W_1_I":sum(self.m.T_GS_W_I_WR_WC[i,c,3]() for c in self.wc_GS),"T_GS_W_2_I":sum(self.m.T_GS_W_I_WR_WC[i,c,5]() for c in self.wc_GS),
                "T_GS_C_0_I":sum(self.m.T_GS_C_I_CR_CC[i,c,0]() for c in self.cc_GS),"T_GS_C_1_I":sum(self.m.T_GS_C_I_CR_CC[i,c,1]() for c in self.cc_GS),"T_GS_C_2_I":sum(self.m.T_GS_C_I_CR_CC[i,c,2]() for c in self.cc_GS),"T_GS_C_3_I":sum(self.m.T_GS_C_I_CR_CC[i,c,3]() for c in self.cc_GS),"T_GS_C_4_I":sum(self.m.T_GS_C_I_CR_CC[i,c,4]() for c in self.cc_GS),"T_GS_C_5_I":sum(self.m.T_GS_C_I_CR_CC[i,c,5]() for c in self.cc_GS),"T_GS_C_6_I":sum(self.m.T_GS_C_I_CR_CC[i,c,6]() for c in self.cc_GS),
                "T_HGS_I":self.m.T_HGS_I[i](),"T_CS_I":self.m.T_CS_I[i](),"T_RLTS_I":self.m.T_RLTS_I[i](),"S_OP_I":self.m.S_OP_I[i](),"S_T_HP_I":self.m.S_T_HP_I[i](),"S_T_HS_I":self.m.S_T_HS_I[i](),"S_T_HXH_I":self.m.S_T_HXH_I[i]() + self.m.S_T_HXH_b_I[i]() + self.m.S_T_HXH_w_I[i](),"S_T_HGC_I":self.m.S_T_HGC_I[i](),
                "S_T_HXA_I":self.m.S_T_HXA_I[i](),"S_T_HXC_I":self.m.S_T_HXC_I[i]() + self.m.S_T_HXC_w_I[i]() + self.m.S_T_HXC_b_I[i](),"S_T_IS_I":sum(self.m.S_T_IS_W_I_WR[i,r]() for r in self.wr_IS)+sum(self.m.S_T_IS_C_I_CR[i,r]() for r in self.cr_IS),"S_T_GS_I":sum(sum(self.m.S_T_GS_W_I_WR_WC[i,c,r]() for r in self.wr_GS) for c in self.wc_GS)+sum(sum(self.m.S_T_GS_C_I_CR_CC[i,c,r]() for r in self.cr_GS)for c in self.cc_GS),
                "S_T_HGS_I":self.m.S_T_HGS_I[i](),"S_T_CS_I":self.m.S_T_CS_I[i](),"S_T_RLTS_I":self.m.S_T_RLTS_I[i](),"q_dem_HS_I":self.q_dem_HS_I[i],"q_dem_CS_I":self.q_dem_CS_I[i],"q_dem_RLTS_I":self.q_dem_RLTS_I[i],"temp_amb_I":self.temp_amb_I[i],"B_T_HXAR_0_I":self.m.B_T_HXAR_N_I[0,i](),"B_T_HXAR_1_I":self.m.B_T_HXAR_N_I[1,i](),"B_T_HP_HXH_0_I":self.m.B_T_HP_HXH_N_I[0,i](),"B_T_HP_HXH_1_I":self.m.B_T_HP_HXH_N_I[1,i](),
                "B_T_HP_HS_0_I":self.m.B_T_HP_HS_N_I[0,i](),"B_T_HP_HS_1_I":self.m.B_T_HP_HS_N_I[1,i](),"B_T_HXA_HXH_0_I":self.m.B_T_HXA_HXH_N_I[0,i](),"B_T_HXA_HXH_1_I":self.m.B_T_HXA_HXH_N_I[1,i](),"B_T_HXA_HGC_0_I":self.m.B_T_HXA_HGC_N_I[0,i](),"B_T_HXA_HGC_1_I":self.m.B_T_HXA_HGC_N_I[1,i](),"B_T_HS_IS_0_I":self.m.B_T_HS_IS_N_I[0,i](),"B_T_HS_IS_1_I":self.m.B_T_HS_IS_N_I[1,i](),
                "B_T_HS_IS_0_I_2":self.m.B_T_HS_IS_N_I_2[0,i](),"B_T_HS_IS_1_I_2":self.m.B_T_HS_IS_N_I_2[1,i](),"B_T_HP_HGC_0_I":self.m.B_T_HP_HGC_N_I[0,i](),"B_T_HP_HGC_1_I":self.m.B_T_HP_HGC_N_I[1,i](),"B_T_HP_HGCHXC_0_I":self.m.B_T_HP_HGCHXC_N_I[0,i](),"B_T_HP_HGCHXC_1_I":self.m.B_T_HP_HGCHXC_N_I[1,i](),"B_T_HP_HGCHXC_0_2_I":self.m.B_T_HP_HGCHXC_N_2_I[0,i](),"B_T_HP_HGCHXC_1_2_I":self.m.B_T_HP_HGCHXC_N_2_I[1,i](),
                "B_T_IS_HGS_0_I":self.m.B_T_IS_HGS_N_I[0,i](),"B_T_IS_HGS_1_I":self.m.B_T_IS_HGS_N_I[1,i](),"B_T_IS_HGS_0_I_2":self.m.B_T_IS_HGS_N_I_2[0,i](),"B_T_IS_HGS_1_I_2":self.m.B_T_IS_HGS_N_I_2[1,i](),"B_T_GS_HGS_0_I":self.m.B_T_GS_HGS_N_I[0,i](),"B_T_GS_HGS_1_I":self.m.B_T_GS_HGS_N_I[1,i](),"B_T_GS_CS_0_I":self.m.B_T_GS_CS_N_I[0,i](),"B_T_GS_CS_1_I":self.m.B_T_GS_CS_N_I[1,i](),
                "B_T_GS_HGS_0_I_2":self.m.B_T_GS_HGS_N_I_2[0,i](),"B_T_GS_HGS_1_I_2":self.m.B_T_GS_HGS_N_I_2[1,i](),"B_T_GS_CS_0_I_2":self.m.B_T_GS_CS_N_I_2[0,i](),"B_T_GS_CS_1_I_2":self.m.B_T_GS_CS_N_I_2[1,i](),"B_T_HXC_HGS_0_I":self.m.B_T_HXC_HGS_N_I[0,i](),"B_T_HXC_HGS_1_I":self.m.B_T_HXC_HGS_N_I[1,i](),
                "B_T_HXC_CS_0_I":self.m.B_T_HXC_CS_N_I[0,i](),"B_T_HXC_CS_1_I":self.m.B_T_HXC_CS_N_I[1,i](),"B_T_HXC_RLTS_0_I":self.m.B_T_HXC_RLTS_N_I[0,i](),"B_T_HXC_RLTS_1_I":self.m.B_T_HXC_RLTS_N_I[1,i]()}, ignore_index=True)
             # Storage temperatures and slacks
            self.safeFile = self.safeFile.append({"T_HP_HT_I":self.m.T_HP_HT_I[self.I[-1]](),"T_HP_LT_I":self.m.T_HP_LT_I[self.I[-1]](),"T_HS_I":self.m.T_HS_I[self.I[-1]](),"T_HXH_b_I":self.m.T_HXH_b_I[self.I[-1]](),"T_HXH_w_I":self.m.T_HXH_w_I[self.I[-1]](),"T_HGC_I":self.m.T_HGC_I[self.I[-1]](),
            "T_HXA_I":self.m.T_HXA_I[self.I[-1]](),"T_HXC_b_I":self.m.T_HXC_b_I[self.I[-1]](),"T_HXC_w_I":self.m.T_HXC_w_I[self.I[-1]](),"T_IS_I":self.m.T_IS_I[self.I[-1]](),"T_ISw_I":(sum(self.m.T_IS_W_I_WR[self.I[-1],r]() for r in self.wr_IS)/len(self.wr_IS)),"T_ISc_I":(sum(self.m.T_IS_C_I_CR[self.I[-1],r]() for r in self.cr_IS)/len(self.cr_IS)),
            "T_IS_W_0_I":self.m.T_IS_W_I_WR[self.I[-1],0](),"T_IS_W_1_I":self.m.T_IS_W_I_WR[self.I[-1],2](),"T_IS_W_2_I":self.m.T_IS_W_I_WR[self.I[-1],4](),"T_IS_C_0_I":self.m.T_IS_C_I_CR[self.I[-1],0](),"T_IS_C_1_I":self.m.T_IS_C_I_CR[self.I[-1],1](),"T_IS_C_2_I":self.m.T_IS_C_I_CR[self.I[-1],2](),"T_IS_C_3_I":self.m.T_IS_C_I_CR[self.I[-1],3](),"T_IS_C_4_I":self.m.T_IS_C_I_CR[self.I[-1],4](),
            "T_GS_I":self.m.T_GS_I[self.I[-1]](),"T_GSw_I":(sum(sum(self.m.T_GS_W_I_WR_WC[self.I[-1],c,r]() for c in self.wc_GS) for r in self.wr_GS)/(len(self.wc_GS)*len(self.wr_GS))),"T_GSc_I":(sum(sum(self.m.T_GS_C_I_CR_CC[self.I[-1],c,r]() for r in self.cr_GS) for c in self.cc_GS)/(len(self.cr_GS)*len(self.cc_GS))),
            "T_GS_W_0_I":sum(self.m.T_GS_W_I_WR_WC[self.I[-1],c,1]() for c in self.wc_GS),"T_GS_W_1_I":sum(self.m.T_GS_W_I_WR_WC[self.I[-1],c,3]() for c in self.wc_GS),"T_GS_W_2_I":sum(self.m.T_GS_W_I_WR_WC[self.I[-1],c,5]() for c in self.wc_GS),
            "T_GS_C_0_I":sum(self.m.T_GS_C_I_CR_CC[self.I[-1],c,0]() for c in self.cc_GS),"T_GS_C_1_I":sum(self.m.T_GS_C_I_CR_CC[self.I[-1],c,1]() for c in self.cc_GS),"T_GS_C_2_I":sum(self.m.T_GS_C_I_CR_CC[self.I[-1],c,2]() for c in self.cc_GS),"T_GS_C_3_I":sum(self.m.T_GS_C_I_CR_CC[self.I[-1],c,3]() for c in self.cc_GS),"T_GS_C_4_I":sum(self.m.T_GS_C_I_CR_CC[self.I[-1],c,4]() for c in self.cc_GS),"T_GS_C_5_I":sum(self.m.T_GS_C_I_CR_CC[self.I[-1],c,5]() for c in self.cc_GS),"T_GS_C_6_I":sum(self.m.T_GS_C_I_CR_CC[self.I[-1],c,6]() for c in self.cc_GS),
            "T_HGS_I":self.m.T_HGS_I[self.I[-1]](),"T_CS_I":self.m.T_CS_I[self.I[-1]](),"T_RLTS_I":self.m.T_RLTS_I[self.I[-1]](),"S_OP_I":self.m.S_OP_I[self.I[-1]](),"S_T_HP_I":self.m.S_T_HP_I[self.I[-1]](),"S_T_HS_I":self.m.S_T_HS_I[self.I[-1]](),"S_T_HXH_I":self.m.S_T_HXH_I[self.I[-1]]() + self.m.S_T_HXH_b_I[self.I[-1]]() + self.m.S_T_HXH_w_I[self.I[-1]](),"S_T_HGC_I":self.m.S_T_HGC_I[self.I[-1]](),
            "S_T_HXA_I":self.m.S_T_HXA_I[self.I[-1]](),"S_T_HXC_I":self.m.S_T_HXC_I[self.I[-1]]() + self.m.S_T_HXC_w_I[self.I[-1]]() + self.m.S_T_HXC_b_I[self.I[-1]](),"S_T_IS_I":sum(self.m.S_T_IS_W_I_WR[self.I[-1],r]() for r in self.wr_IS)+sum(self.m.S_T_IS_C_I_CR[self.I[-1],r]() for r in self.cr_IS),"S_T_GS_I":sum(sum(self.m.S_T_GS_W_I_WR_WC[self.I[-1],c,r]() for r in self.wr_GS) for c in self.wc_GS)+sum(sum(self.m.S_T_GS_C_I_CR_CC[self.I[-1],c,r]() for r in self.cr_GS)for c in self.cc_GS),"S_T_HGS_I":self.m.S_T_HGS_I[self.I[-1]](),"S_T_CS_I":self.m.S_T_CS_I[i](),
            "S_T_RLTS_I":self.m.S_T_RLTS_I[self.I[-1]]()}, ignore_index=True)
           
            self.safeFile = self.safeFile.round(4)
            if singleFile == True:
                source.setOptimizationResults(dataFrame=self.safeFile,savePath=savePath)
            return self.safeFile
        except:
            raise RuntimeError("Optimization I didn't come to a solution.")

if __name__ == "__main__":
    test = Linear_Binary_Model()
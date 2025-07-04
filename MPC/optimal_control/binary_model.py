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

class Binary_Model():
    
    def __init__(self):
        self.warmstart_available = False

    def setProfiles(self,profileForecastHeat,profileForecastCool,profileForecastDry,profileForecastWeather,profileForecastPrice,profileForecastFrost):
        self.q_dem_HS_T = profileForecastHeat
        self.q_dem_CS_T = profileForecastCool
        self.q_dem_RLTS_T = profileForecastDry
        self.temp_amb_T = profileForecastWeather
        self.c_ELECTRICITY_buy_T = profileForecastPrice
        self.temp_frost_T = profileForecastFrost

    def setParams(self,timeSteps,stepSizeInSec,controlPeriod1,controlPeriod2,tControlPeriodSwitch):
        ## Time
        self.T = timeSteps
        self.StepSizeInSec = stepSizeInSec 
        ## Control steps
        self.ControlPeriod1 = controlPeriod1
        self.ControlPeriod2 = controlPeriod2
        self.TControlPeriodSwitch1 = tControlPeriodSwitch
        ## HP
        self.H = list(range(0,5))
        ## VP_HXC_HGS_CS_RLTS
        self.V_1 = list(range(0,8))
        self.V_2 = list(range(0,14))
        ## General
        self.c_w = 4.18
        self.c_b = 3.56
        self.c_a = 1.01
        self.c_c = 0.879
        self.t_conection_delta = 50
        self.t_default = 20
        self.t_hour_in_sec = 3600
        ## Slack constants
        self.s_T_HP = 500
        self.s_T_HS = 500
        self.s_T_HXH = 500
        self.s_T_HGC = 500
        self.s_T_HXC = 500
        self.s_T_HGS = 500
        self.s_T_CS = 500
        self.s_T_RLTS = 500
        self.s_T_HXA = 500
        self.s_T_GS_C = 500
        self.s_T_GS_W = 500
        self.s_T_IS_C = 500
        self.s_T_IS_W = 500
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
        self.c_switch_HP_1 = 1 # since two switches at once, 1 € per switch // 2 € per on/off
        self.c_switch_HP_2 = 0.5 # since two switches at once, 1 € per switch // 2 € per on/off
        ## HS
        self.alpha_HS_time = 0.01 
        self.T_HS_max = 40 
        self.T_HS_min = 33 
        self.m_HS_w = 6000 
        self.T_HS_delta_max = 100 
        self.T_HS_delta_min = -100 
        ## HXA
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
        self.c_switch_HXA = 0.1 # with 0.1 10 times more expensive than normal switches, since HXA has a long range and needs lot of energy to start
        ## V_HP_HXH_HS
        self.c_switch_HXH_HS = 0.01
        ## HXH
        self.T_HXH_max = 60
        self.T_HXH_min = 0 
        self.T_HXH_delta_max = 100 
        self.T_HXH_delta_min = -100 
        self.a_HXH_w_b = 44.12 # see datasheet m2
        self.alpha_HXH_w_b = 4 # see datasheet W/m2 K
        ## V_HXA_HXH_HGC
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
        self.mdot_IS_w_2 = self.mdot_IS_w/2
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
        self.T_GS_delta_max = 100 
        self.T_GS_delta_min = -100
        self.n_GS_blocks = 306 
        self.mdot_GS_w = 50.00
        self.mdot_GS_w_2 = self.mdot_GS_w/2
        self.e_GS_EL = (5 * 1.3) # see datasheet
        ## VP_HS_IS
        ## Nothing needed!
        ## V_HP_HGC_HGCHXC
        self.c_switch_HGC_HGCHXC = 0.01
        ## VP_IS_HGS
        ## Nothing needed!
        ## V_GS_HGS_CS
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
        self.mdot_VP_RLTS_V_1 = [0, 0,      0,      0,     16.000, 8.000, 8.000, 5.333]
        self.mdot_VP_CS_V_1 =   [0, 0,      16.000, 8.000, 0,      0,     8.000, 5.333]
        self.mdot_VP_HGS_V_1 =  [0, 16.000, 0,      8.000, 0,      8.000, 0,     5.333]
        self.mdot_VP_RLTS_V_2 = [0, 0,      0,      0,     5.333,  4.000, 4.000, 3.200,  8.000, 8.000, 6.400,  8.000,  6.400, 5.333]
        self.mdot_VP_CS_V_2 =   [0, 0,      5.333,  8.000, 0,      4.000, 8.000, 6.400,  0,     4.000, 3.200,  8.000,  6.400, 5.333]
        self.mdot_VP_HGS_V_2 =  [0, 16.000, 10.667, 8.000, 10.667, 8.000, 4.000, 6.400,  8.000, 4.000, 6.4000, 0,      3.200, 5.333]
        self.mdot_VP_tot = 16.00
        self.c_switch_VP = 0.005
        self.e_VP_EL = 3.7

    def setVariables(self,model,binary=1):
        self.m = model
        ## General variables
        self.m.C_TOT_T_ = pyo.Var(domain=pyo.NonNegativeReals)
        self.m.C_OP_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals)
        ## Slack variables
        self.m.S_TOT_T_ = pyo.Var(domain=pyo.NonNegativeReals)
        self.m.S_OP_T = pyo.Var(self.T[1:], domain=pyo.NonNegativeReals)
        ## Toggle variables (Switching variables)
        self.m.T_TOT_T_ = pyo.Var(domain=pyo.NonNegativeReals)
        self.m.T_OP_T = pyo.Var(self.T, domain=pyo.NonNegativeReals)
        ## HP
        self.m.E_HP_EL_in_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals)
        self.m.E_HP_EL_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals)
        self.m.Q_HP_HT_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals)
        self.m.Q_HP_LT_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals)
        if binary == 0:
            self.m.B_HP_H_T = pyo.Var(self.H, self.T[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_HP_H_T = pyo.Var(self.H, self.T[0:-1], domain=pyo.Binary)
        if self.TControlPeriodSwitch1 > 0:
            self.m.Z_HP_T_1 = pyo.Var(self.T, domain=pyo.NonNegativeReals)
        self.m.Z_HP_T_2 = pyo.Var(self.T, domain=pyo.NonNegativeReals)
        self.m.Z_HP_Q_HT_H_T = pyo.Var(self.H, self.T[0:-1], domain=pyo.NonNegativeReals)
        self.m.Z_HP_Q_LT_H_T = pyo.Var(self.H, self.T[0:-1], domain=pyo.NonNegativeReals)
        self.m.Z_HP_E_EL_H_T = pyo.Var(self.H, self.T[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_HP_HT_T = pyo.Var(self.T, domain=pyo.Reals)
        self.m.T_HP_HT_in_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HP_HT_in_H_T = pyo.Var(self.H, self.T[0:-1], domain=pyo.Reals)
        self.m.T_HP_HT_out_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.T_HP_LT_T = pyo.Var(self.T, domain=pyo.Reals)
        self.m.T_HP_LT_in_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HP_LT_in_H_T = pyo.Var(self.H, self.T[0:-1], domain=pyo.Reals)
        self.m.T_HP_LT_out_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.T_HP_HT_in_HXH_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HP_HT_H_in_HXH_T = pyo.Var(self.H, self.T[0:-1], domain=pyo.Reals)
        self.m.T_HP_HT_in_HS_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HP_HT_H_in_HS_T = pyo.Var(self.H, self.T[0:-1], domain=pyo.Reals)
        self.m.T_HP_LT_in_HGC_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HP_LT_H_in_HGC_T = pyo.Var(self.H, self.T[0:-1], domain=pyo.Reals)
        self.m.S_T_HP_T = pyo.Var(self.T[1:], domain=pyo.NonNegativeReals)
        ## HS
        self.m.T_HS_T = pyo.Var(self.T, domain=pyo.Reals)
        self.m.T_HS_HT_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HS_HT_H_T = pyo.Var(self.H, self.T[0:-1], domain=pyo.Reals)
        self.m.T_HS_LT_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HS_LT_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HS_LT_2_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.S_T_HS_T = pyo.Var(self.T[1:], domain=pyo.NonNegativeReals)
        ## HXA
        self.m.T_HXA_in_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        if binary == 0:
            self.m.B_HXA_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_HXA_T = pyo.Var(self.T[0:-1], domain=pyo.Binary)
        self.m.Z_HXA_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        self.m.E_HXA_EL_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals)
        self.m.T_HXA_T = pyo.Var(self.T, domain=pyo.Reals)
        self.m.T_HXA_HXH_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_T_HXA_HXH_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.T_HXA_HGC_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_T_HXA_HGC_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HXA_out_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.S_T_HXA_T = pyo.Var(self.T[1:], domain=pyo.NonNegativeReals)
        self.m.Z_HXA_s_T = pyo.Var(self.T, domain=pyo.NonNegativeReals)
        ## V_HP_HXH_HS
        if binary == 0:
            self.m.B_HXH_HS_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_HXH_HS_T = pyo.Var(self.T[0:-1], domain=pyo.Binary)
        self.m.Z_HXH_HS_T = pyo.Var(self.T, domain=pyo.NonNegativeReals)
        self.m.Z_HP_HXH_H_T = pyo.Var(self.H,self.T[0:-1],domain=pyo.NonNegativeReals)
        self.m.Z_HP_HS_H_T = pyo.Var(self.H,self.T[0:-1],domain=pyo.NonNegativeReals)
        ## HXH
        self.m.T_HXH_T = pyo.Var(self.T, domain=pyo.Reals)
        self.m.T_HXH_w_in_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.T_HXH_w_out_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HXH_w_H_T = pyo.Var(self.H,self.T[0:-1], domain=pyo.Reals)
        self.m.T_HXH_b_in_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.T_HXH_b_out_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HXH_b_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Qdot_HXH_w_b_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.S_T_HXH_T = pyo.Var(self.T[1:], domain=pyo.NonNegativeReals)
        ## V_HXA_HXH_HGC
        if binary == 0:
            self.m.B_HXH_HGC_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_HXH_HGC_T = pyo.Var(self.T[0:-1], domain=pyo.Binary)
        self.m.Z_HXH_HGC_T = pyo.Var(self.T, domain=pyo.NonNegativeReals) 
        self.m.Z_HXA_HXH_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals) 
        self.m.Z_HXA_HGC_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals)
        ## HGC
        self.m.T_HGC_T = pyo.Var(self.T, domain=pyo.Reals)
        self.m.T_HGC_HP_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HGC_HP_H_T = pyo.Var(self.H, self.T[0:-1], domain=pyo.Reals)
        self.m.T_HGC_HXC_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HGC_HXC_H_T = pyo.Var(self.H, self.T[0:-1], domain=pyo.Reals)
        self.m.T_HGC_HXA_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HGC_HXA_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.S_T_HGC_T = pyo.Var(self.T[1:], domain=pyo.NonNegativeReals)
        ## HXC
        self.m.T_HXC_T = pyo.Var(self.T, domain=pyo.Reals)
        self.m.T_HXC_b_in_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.T_HXC_b_out_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HXC_b_H_T = pyo.Var(self.H, self.T[0:-1], domain=pyo.Reals)
        self.m.T_HXC_w_delta_in_T = pyo.Var(self.T[0:-1], domain=pyo.Reals) 
        self.m.T_HXC_w_out_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.T_HXC_HGS_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        if self.TControlPeriodSwitch1 > 0:
            self.m.Z_HXC_HGS_V_T_1 = pyo.Var(self.V_1, self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HXC_HGS_V_T_2 = pyo.Var(self.V_2, self.T[0:-1], domain=pyo.Reals)
        if self.TControlPeriodSwitch1 > 0:
            self.m.Z_HGS_V_T_1 = pyo.Var(self.V_1, self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HGS_V_T_2 = pyo.Var(self.V_2, self.T[0:-1], domain=pyo.Reals)
        self.m.T_HXC_CS_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        if self.TControlPeriodSwitch1 > 0:
            self.m.Z_HXC_CS_V_T_1 = pyo.Var(self.V_1, self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HXC_CS_V_T_2 = pyo.Var(self.V_2, self.T[0:-1], domain=pyo.Reals)
        if self.TControlPeriodSwitch1 > 0:
            self.m.Z_CS_V_T_1 = pyo.Var(self.V_1, self.T[0:-1], domain=pyo.Reals)
        self.m.Z_CS_V_T_2 = pyo.Var(self.V_2, self.T[0:-1], domain=pyo.Reals)
        self.m.T_HXC_RLTS_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        if self.TControlPeriodSwitch1 > 0:
            self.m.Z_HXC_RLTS_V_T_1 = pyo.Var(self.V_1, self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HXC_RLTS_V_T_2 = pyo.Var(self.V_2, self.T[0:-1], domain=pyo.Reals)
        if self.TControlPeriodSwitch1 > 0:
            self.m.Z_RLTS_V_T_1 = pyo.Var(self.V_1, self.T[0:-1], domain=pyo.Reals)
        self.m.Z_RLTS_V_T_2 = pyo.Var(self.V_2, self.T[0:-1], domain=pyo.Reals)
        self.m.Qdot_HXC_w_b_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.S_T_HXC_T = pyo.Var(self.T[1:], domain=pyo.NonNegativeReals)
        ## HGS
        self.m.T_HGS_T = pyo.Var(self.T, domain=pyo.Reals)
        self.m.T_HGS_HXC_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        if self.TControlPeriodSwitch1 > 0:
            self.m.Z_HGS_HXC_V_T_1 = pyo.Var(self.V_1, self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HGS_HXC_V_T_2 = pyo.Var(self.V_2, self.T[0:-1], domain=pyo.Reals)
        self.m.T_HGS_IS_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HGS_IS_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HGS_IS_2_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.T_HGS_GS_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)  
        self.m.Z_HGS_GS_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_HGS_GS_2_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.S_T_HGS_T = pyo.Var(self.T[1:], domain=pyo.NonNegativeReals)
        ## IS
        self.m.T_IS_C_T_CR = pyo.Var(self.T,self.cr_IS, domain=pyo.Reals)
        self.m.T_IS_W_T_WR = pyo.Var(self.T,self.wr_IS, domain=pyo.Reals)
        self.m.S_T_IS_C_T_CR = pyo.Var(self.T[1:],self.cr_IS, domain=pyo.NonNegativeReals)
        self.m.S_T_IS_W_T_WR = pyo.Var(self.T[1:],self.wr_IS, domain=pyo.NonNegativeReals)
        self.m.Q_IS_C_NORTH_T_CR = pyo.Var(self.T[:-1],self.cr_IS, domain=pyo.Reals)
        self.m.Q_IS_C_SOUTH_T_CR = pyo.Var(self.T[:-1],self.cr_IS, domain=pyo.Reals)
        self.m.Q_IS_C_W_T_WR = pyo.Var(self.T[:-1],self.wr_IS, domain=pyo.Reals)
        self.m.Q_IS_W_T_WR = pyo.Var(self.T[:-1],self.wr_IS, domain=pyo.Reals)
        self.m.Q_IS_W_T_IN = pyo.Var(self.T[:-1], domain=pyo.Reals)
        self.m.Q_IS_W_C_T_WR = pyo.Var(self.T[:-1],self.wr_IS, domain=pyo.Reals)
        self.m.Z_HS_HGS_T = pyo.Var(self.T, domain=pyo.NonNegativeReals)
        # Old
        self.m.T_IS_T = pyo.Var(self.T, domain=pyo.Reals)
        self.m.T_IS_HT_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_IS_HT_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_IS_HT_2_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.T_IS_LT_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_IS_LT_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_IS_LT_2_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.E_IS_EL_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals)
        self.m.Z_IS_pump_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals)
        ## GS
        self.m.T_GS_C_T_CR_CC = pyo.Var(self.T,self.cc_GS,self.cr_GS, domain=pyo.Reals)
        self.m.T_GS_W_T_WR_WC = pyo.Var(self.T,self.wc_GS,self.wr_GS, domain=pyo.Reals)
        self.m.S_T_GS_C_T_CR_CC = pyo.Var(self.T,self.cc_GS,self.cr_GS, domain=pyo.NonNegativeReals)
        self.m.S_T_GS_W_T_WR_WC = pyo.Var(self.T,self.wc_GS,self.wr_GS, domain=pyo.NonNegativeReals)
        self.m.Q_GS_C_NORTH_T_CR_CC = pyo.Var(self.T,self.cc_GS,self.cr_GS, domain=pyo.Reals)
        self.m.Q_GS_C_EAST_T_CR_CC = pyo.Var(self.T,self.cc_GS,self.cr_GS, domain=pyo.Reals)
        self.m.Q_GS_C_SOUTH_T_CR_CC = pyo.Var(self.T,self.cc_GS,self.cr_GS, domain=pyo.Reals)
        self.m.Q_GS_C_WEST_T_CR_CC = pyo.Var(self.T,self.cc_GS,self.cr_GS, domain=pyo.Reals)
        self.m.Q_GS_C_W_T_WR_WC = pyo.Var(self.T,self.wc_GS,self.wr_GS, domain=pyo.Reals)
        self.m.Q_GS_W_EAST_T_WR_WC = pyo.Var(self.T,self.wc_GS,self.wr_GS, domain=pyo.Reals)
        self.m.Q_GS_W_WEST_T_WR_WC = pyo.Var(self.T,self.wc_GS,self.wr_GS, domain=pyo.Reals)
        self.m.Q_GS_W_C_T_WR_WC = pyo.Var(self.T,self.wc_GS,self.wr_GS, domain=pyo.Reals)
        self.m.T_GS_T = pyo.Var(self.T, domain=pyo.Reals)
        self.m.T_GS_HGS_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_T_GS_HGS_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_T_GS_HGS_2_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.T_GS_CS_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_T_GS_CS_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_T_GS_CS_2_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.E_GS_EL_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals)
        ## VP_HS_IS
        if binary == 0:
            self.m.B_HS_IS_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_HS_IS_T = pyo.Var(self.T[0:-1], domain=pyo.Binary)
        self.m.Z_HS_IS_2_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        ## V_HP_HGC_HGCHXC
        if binary == 0:
            self.m.B_HGC_HGCHXC_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_HGC_HGCHXC_T = pyo.Var(self.T[0:-1], domain=pyo.Binary)
        self.m.Z_HGC_HGCHXC_T = pyo.Var(self.T, domain=pyo.NonNegativeReals)
        self.m.Z_HP_HGC_H_T = pyo.Var(self.H, self.T[0:-1], domain=pyo.NonNegativeReals)
        self.m.Z_HP_HGCHXC_H_T = pyo.Var(self.H, self.T[0:-1], domain=pyo.NonNegativeReals)
        ## VP_IS_HGS
        if binary == 0:
            self.m.B_IS_HGS_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_IS_HGS_T = pyo.Var(self.T[0:-1], domain=pyo.Binary)
        self.m.Z_IS_HGS_2_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        ## V_GS_HGS_CS
        if binary == 0:
            self.m.B_GS_HGS_CS_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
            self.m.B_GS_HGS_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1)) 
            self.m.B_GS_CS_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_GS_HGS_CS_T = pyo.Var(self.T[0:-1], domain=pyo.Binary)
            self.m.B_GS_HGS_T = pyo.Var(self.T[0:-1], domain=pyo.Binary) 
            self.m.B_GS_CS_T = pyo.Var(self.T[0:-1], domain=pyo.Binary)
        
        self.m.Z_HGS_CS_T = pyo.Var(self.T, domain=pyo.NonNegativeReals)
        ## CS
        self.m.T_CS_T = pyo.Var(self.T, domain=pyo.Reals)
        self.m.T_CS_HXC_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        if self.TControlPeriodSwitch1 > 0:
            self.m.Z_CS_HXC_V_T_1 = pyo.Var(self.V_1, self.T[0:-1], domain=pyo.Reals)
        self.m.Z_CS_HXC_V_T_2 = pyo.Var(self.V_2, self.T[0:-1], domain=pyo.Reals)
        self.m.T_CS_GS_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_CS_GS_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.Z_CS_GS_2_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        self.m.S_T_CS_T = pyo.Var(self.T[1:], domain=pyo.NonNegativeReals)
        ## RLTS
        self.m.T_RLTS_T = pyo.Var(self.T, domain=pyo.Reals)
        self.m.T_RLTS_HXC_T = pyo.Var(self.T[0:-1], domain=pyo.Reals)
        if self.TControlPeriodSwitch1 > 0:
            self.m.Z_RLTS_HXC_V_T_1 = pyo.Var(self.V_1, self.T[0:-1], domain=pyo.Reals)
        self.m.Z_RLTS_HXC_V_T_2 = pyo.Var(self.V_2, self.T[0:-1], domain=pyo.Reals)
        self.m.S_T_RLTS_T = pyo.Var(self.T[1:], domain=pyo.NonNegativeReals)
        ## VP_HXC_HGS_CS_RLTS
        if self.TControlPeriodSwitch1 > 0:
            if binary == 0:
                self.m.B_VP_V_T_1 = pyo.Var(self.V_1, self.T[0:self.TControlPeriodSwitch1], domain=pyo.NonNegativeReals, bounds=(0,1))
            else:
                self.m.B_VP_V_T_1 = pyo.Var(self.V_1, self.T[0:self.TControlPeriodSwitch1], domain=pyo.Binary)
        
        if binary == 0:
            self.m.B_VP_V_T_2 = pyo.Var(self.V_2, self.T[self.TControlPeriodSwitch1:-1], domain=pyo.NonNegativeReals, bounds=(0,1))
        else:
            self.m.B_VP_V_T_2 = pyo.Var(self.V_2, self.T[self.TControlPeriodSwitch1:-1], domain=pyo.Binary)
        if self.TControlPeriodSwitch1 > 0:
            self.m.Z_VP_V_T_1 = pyo.Var(self.V_1, self.T, domain=pyo.NonNegativeReals)
        self.m.Z_VP_V_T_2 = pyo.Var(self.V_2, self.T, domain=pyo.NonNegativeReals)
        self.m.E_VP_EL_T = pyo.Var(self.T[0:-1], domain=pyo.NonNegativeReals)

        return self.m

    def setStartValues(self,model,T_HP_HT_start,T_HP_LT_start,T_HS_start,T_HXA_start,T_HGC_start,T_HGS_start,T_IS_w_1_start,T_IS_w_2_start,T_IS_w_3_start,T_IS_c_1_start,T_IS_c_2_start,T_IS_c_3_start,
    T_IS_c_4_start,T_IS_c_5_start,T_GS_w_1_start,T_GS_w_2_start,T_GS_w_3_start,T_GS_c_1_start,T_GS_c_2_start,T_GS_c_3_start,T_GS_c_4_start,T_GS_c_5_start,T_GS_c_6_start,T_GS_c_7_start,T_CS_start,T_RLTS_start,
    Start_Toggle_Constraints,B_HP_1_start,B_HP_2_start,B_HP_3_start,B_HP_4_start,B_HXH_HS_start,B_HGC_HGCHXC_start,B_HXA_start,B_HXH_HGC_start,B_HS_IS_start,B_IS_HGS_start,B_GS_HGS_start,B_GS_CS_start,B_GS_HGS_CS_start,B_VP_start):
        self.m = model
        self.T_HP_HT_start = T_HP_HT_start
        self.T_HP_LT_start = T_HP_LT_start
        self.T_HS_start = T_HS_start
        self.T_HXA_start = T_HXA_start
        self.T_HGC_start = T_HGC_start
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
        self.Start_Toggle_Constraints = Start_Toggle_Constraints
        if Start_Toggle_Constraints == True:
            self.B_HP_1_start = B_HP_1_start
            self.B_HP_2_start = B_HP_2_start
            self.B_HP_3_start = B_HP_3_start
            self.B_HP_4_start = B_HP_4_start
            self.B_HXH_HS_start = B_HXH_HS_start
            self.B_HGC_HGCHXC_start = B_HGC_HGCHXC_start
            self.B_HXA_start = B_HXA_start
            self.B_HXH_HGC_start = B_HXH_HGC_start
            self.B_HS_IS_start = B_HS_IS_start
            self.B_IS_HGS_start = B_IS_HGS_start
            self.B_GS_HGS_start = B_GS_HGS_start
            self.B_GS_CS_start = B_GS_CS_start
            self.B_GS_HGS_CS_start = B_GS_HGS_CS_start
            self.B_VP_start = B_VP_start
        return self.m

    def setEndValues(self,model,End_Temp_Constraints,T_HS_end,T_CS_end,T_RLTS_end,End_Toggle_Constraints,B_HP_1_end,B_HP_2_end,B_HP_3_end,B_HP_4_end,B_HXH_HS_end,B_HGC_HGCHXC_end,B_HXA_end,B_HXH_HGC_end,B_HS_IS_end,B_IS_HGS_end,B_GS_HGS_end,B_GS_CS_end,B_GS_HGS_CS_end):
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
            self.B_HXH_HS_end = B_HXH_HS_end
            self.B_HGC_HGCHXC_end = B_HGC_HGCHXC_end
            self.B_HXA_end = B_HXA_end
            self.B_HXH_HGC_end = B_HXH_HGC_end
            self.B_HS_IS_end = B_HS_IS_end
            self.B_IS_HGS_end = B_IS_HGS_end
            self.B_GS_HGS_end = B_GS_HGS_end
            self.B_GS_CS_end = B_GS_CS_end
            self.B_GS_HGS_CS_end = B_GS_HGS_CS_end
            ## All but cold side HP HXC
        return self.m

    def setConstraints(self,model):
        self.m = model
        ## General cost constraint
        self.m.Constraint_Cost_T = pyo.Constraint(expr = self.m.C_TOT_T_ == sum(self.m.C_OP_T[t] for t in self.T[0:-1]))

        ## Cost constraints 
        self.m.Constraint_Cost_time_T = pyo.ConstraintList()
        for t in self.T[0:-1]:
            self.m.Constraint_Cost_time_T.add(self.m.C_OP_T[t] == self.StepSizeInSec/self.t_hour_in_sec * ((self.m.E_HP_EL_in_T[t] + self.m.E_HXA_EL_T[t] + self.m.E_IS_EL_T[t] + self.m.E_GS_EL_T[t] + self.m.E_VP_EL_T[t]) * self.c_ELECTRICITY_buy_T[t]))

        ## General Slack constraint
        self.m.Constraint_Slack_T = pyo.Constraint(expr = self.m.S_TOT_T_ == sum(self.m.S_OP_T[t] for t in self.T[1:]))

        ## Slack constraints
        self.m.Constraint_Slack_time_T = pyo.ConstraintList()
        for t in self.T[1:]:
            self.m.Constraint_Slack_time_T.add(self.m.S_OP_T[t] == self.StepSizeInSec/self.t_hour_in_sec * (self.s_T_HP * self.m.S_T_HP_T[t] + self.s_T_HS * self.m.S_T_HS_T[t] + sum(self.s_T_IS_W * self.m.S_T_IS_W_T_WR[t,r] for r in self.wr_IS) + sum(self.s_T_IS_C * self.m.S_T_IS_C_T_CR[t,r] for r in self.cr_IS) + self.s_T_HXH * self.m.S_T_HXH_T[t] + self.s_T_HGC * self.m.S_T_HGC_T[t] + self.s_T_HXC * self.m.S_T_HXC_T[t] + self.s_T_HGS * self.m.S_T_HGS_T[t] + sum(sum(self.s_T_GS_W * self.m.S_T_GS_W_T_WR_WC[t,c,r] for r in self.wr_GS) for c in self.wc_GS) + sum(sum(self.s_T_GS_C * self.m.S_T_GS_C_T_CR_CC[t,c,r] for r in self.cr_GS) for c in self.cc_GS) + self.s_T_CS * self.m.S_T_CS_T[t] + self.s_T_RLTS * self.m.S_T_RLTS_T[t] + self.s_T_HXA * self.m.S_T_HXA_T[t]))

        ## General toggle constraint
        self.m.Constraint_Toggle_T = pyo.Constraint(expr = self.m.T_TOT_T_ == sum(self.m.T_OP_T[t] for t in self.T))

        ## Toggle constraints
        self.m.Constraint_Toggle_time_T = pyo.ConstraintList()
        if self.TControlPeriodSwitch1 > 0:
            for t in self.T:
                self.m.Constraint_Toggle_time_T.add(self.m.T_OP_T[t] == self.m.Z_HP_T_1[t] * self.c_switch_HP_1 + self.m.Z_HP_T_2[t] * self.c_switch_HP_2 + self.m.Z_HXH_HS_T[t] * self.c_switch_HXH_HS + self.m.Z_HGC_HGCHXC_T[t] * self.c_switch_HGC_HGCHXC + self.m.Z_HXH_HGC_T[t] * self.c_switch_HXH_HGC + self.m.Z_HXA_s_T[t] * self.c_switch_HXA + self.m.Z_HS_HGS_T[t] * self.c_switch_HS_HGS + self.m.Z_HGS_CS_T[t] * self.c_switch_HGS_CS + sum(self.m.Z_VP_V_T_1[v,t] * self.c_switch_VP for v in self.V_1) + sum(self.m.Z_VP_V_T_2[v,t] * self.c_switch_VP for v in self.V_2))
        else:
            for t in self.T:
                self.m.Constraint_Toggle_time_T.add(self.m.T_OP_T[t] == self.m.Z_HP_T_2[t] * self.c_switch_HP_2 + self.m.Z_HXH_HS_T[t] * self.c_switch_HXH_HS + self.m.Z_HGC_HGCHXC_T[t] * self.c_switch_HGC_HGCHXC + self.m.Z_HXH_HGC_T[t] * self.c_switch_HXH_HGC + self.m.Z_HXA_s_T[t] * self.c_switch_HXA + self.m.Z_HS_HGS_T[t] * self.c_switch_HS_HGS + self.m.Z_HGS_CS_T[t] * self.c_switch_HGS_CS + sum(self.m.Z_VP_V_T_2[v,t] * self.c_switch_VP for v in self.V_2))
        ## HP
        self.m.Constraint_HP_T = pyo.ConstraintList()
        for t in self.T[0:-1]:
            self.m.Constraint_HP_T.add(sum(self.m.B_HP_H_T[h,t] for h in self.H) == 1) ## SOS1 constraint

            self.m.Constraint_HP_T.add(self.m.Q_HP_HT_T[t] == self.a_HP_HT_0 + self.a_HP_HT_1 * self.m.T_HP_HT_in_T[t] + self.a_HP_HT_2 * self.m.T_HP_LT_in_T[t]) ## Thermal equation HT
            self.m.Constraint_HP_T.add(self.m.Q_HP_LT_T[t] == self.a_HP_LT_0 + self.a_HP_LT_1 * self.m.T_HP_HT_in_T[t] + self.a_HP_LT_2 * self.m.T_HP_LT_in_T[t]) ## Thermal equation LT
            self.m.Constraint_HP_T.add(self.m.E_HP_EL_T[t] == self.a_HP_EL_0 + self.a_HP_EL_1 * self.m.T_HP_HT_in_T[t] + self.a_HP_EL_2 * self.m.T_HP_LT_in_T[t]) ## Electrical equation 
            
            for h in self.H:
                self.m.Constraint_HP_T.add(self.m.Z_HP_Q_HT_H_T[h,t] <= self.q_HP_HT_max) ## Big M to gain self.m.Z_HP_Q_HT_H_T[0][t] as Q dot
                self.m.Constraint_HP_T.add(self.m.Z_HP_Q_HT_H_T[h,t] >= self.q_HP_HT_min) ## Big M
                self.m.Constraint_HP_T.add(self.m.Z_HP_Q_HT_H_T[h,t] <= self.m.B_HP_H_T[h,t] * self.q_HP_HT_max) ## Big M 
                self.m.Constraint_HP_T.add(self.m.Z_HP_Q_HT_H_T[h,t] >= self.m.B_HP_H_T[h,t] * self.q_HP_HT_min) ## Big M 
                self.m.Constraint_HP_T.add(self.m.Z_HP_Q_HT_H_T[h,t] <= self.m.Q_HP_HT_T[t] - (1-self.m.B_HP_H_T[h,t]) * self.q_HP_HT_min) ## Big M 
                self.m.Constraint_HP_T.add(self.m.Z_HP_Q_HT_H_T[h,t] >= self.m.Q_HP_HT_T[t] - (1-self.m.B_HP_H_T[h,t]) * self.q_HP_HT_max) ## Big M 

                self.m.Constraint_HP_T.add(self.m.Z_HP_Q_LT_H_T[h,t] <= self.q_HP_LT_max) ## Big M to gain self.m.Z_HP_Q_LT_H_T[0][t] as Q dot
                self.m.Constraint_HP_T.add(self.m.Z_HP_Q_LT_H_T[h,t] >= self.q_HP_LT_min) ## Big M 
                self.m.Constraint_HP_T.add(self.m.Z_HP_Q_LT_H_T[h,t] <= self.m.B_HP_H_T[h,t] * self.q_HP_LT_max) ## Big M 
                self.m.Constraint_HP_T.add(self.m.Z_HP_Q_LT_H_T[h,t] >= self.m.B_HP_H_T[h,t] * self.q_HP_LT_min) ## Big M 
                self.m.Constraint_HP_T.add(self.m.Z_HP_Q_LT_H_T[h,t] <= self.m.Q_HP_LT_T[t] - (1-self.m.B_HP_H_T[h,t]) * self.q_HP_LT_min) ## Big M
                self.m.Constraint_HP_T.add(self.m.Z_HP_Q_LT_H_T[h,t] >= self.m.Q_HP_LT_T[t] - (1-self.m.B_HP_H_T[h,t]) * self.q_HP_LT_max) ## Big M  

                self.m.Constraint_HP_T.add(self.m.Z_HP_E_EL_H_T[h,t] <= self.q_HP_EL_max) ## Big M to gain self.m.Z_HP_Q_H_T[0][t] as Q dot
                self.m.Constraint_HP_T.add(self.m.Z_HP_E_EL_H_T[h,t] >= self.q_HP_EL_min) ## Big M 
                self.m.Constraint_HP_T.add(self.m.Z_HP_E_EL_H_T[h,t] <= self.m.B_HP_H_T[h,t] * self.q_HP_EL_max) ## Big M 
                self.m.Constraint_HP_T.add(self.m.Z_HP_E_EL_H_T[h,t] >= self.m.B_HP_H_T[h,t] * self.q_HP_EL_min) ## Big M 
                self.m.Constraint_HP_T.add(self.m.Z_HP_E_EL_H_T[h,t] <= self.m.E_HP_EL_T[t] - (1-self.m.B_HP_H_T[h,t]) * self.q_HP_EL_min) ## Big M 
                self.m.Constraint_HP_T.add(self.m.Z_HP_E_EL_H_T[h,t] >= self.m.E_HP_EL_T[t] - (1-self.m.B_HP_H_T[h,t]) * self.q_HP_EL_max) ## Big M 

            self.m.Constraint_HP_T.add(self.m.Q_HP_HT_T[t] == sum(self.m.Z_HP_Q_HT_H_T[h,t] for h in self.H)) ## Thighten Relaxation problem
            self.m.Constraint_HP_T.add(self.m.Q_HP_LT_T[t] == sum(self.m.Z_HP_Q_LT_H_T[h,t] for h in self.H)) ## Thighten Relaxation problem
            self.m.Constraint_HP_T.add(self.m.E_HP_EL_T[t] == sum(self.m.Z_HP_E_EL_H_T[h,t] for h in self.H)) ## Thighten Relaxation problem
            
            self.m.Constraint_HP_T.add(self.m.T_HP_HT_out_T[t] == self.m.T_HP_HT_in_T[t] + 1/self.c_w * (sum(self.d_HP_power_H[h] * 1/self.mdot_HP_w_H[h] * self.m.Z_HP_Q_HT_H_T[h,t] for h in self.H[1:]))) ## HT last part for division by zero
            self.m.Constraint_HP_T.add(self.m.T_HP_LT_out_T[t] == self.m.T_HP_LT_in_T[t] - 1/self.c_b * (sum(self.d_HP_power_H[h] * 1/self.mdot_HP_b_H[h] * self.m.Z_HP_Q_LT_H_T[h,t] for h in self.H[1:]))) ## LT last part for division by zero
            self.m.Constraint_HP_T.add(self.m.E_HP_EL_in_T[t] == sum(self.d_HP_power_H[h] * self.m.Z_HP_E_EL_H_T[h,t] for h in self.H) + sum(self.e_HP_EL_pumps[h] * self.m.B_HP_H_T[h,t] for h in self.H)) ## EL

            self.m.Constraint_HP_T.add(self.m.T_HP_LT_in_T[t] >= -10) ## physical constraints
            self.m.Constraint_HP_T.add(self.m.T_HP_LT_out_T[t] >= -15) ## physical constraints
            self.m.Constraint_HP_T.add(self.m.T_HP_HT_in_T[t] <= 60) ## physical constraints

        if self.TControlPeriodSwitch1 > 0:
            for t in self.T[0:self.TControlPeriodSwitch1-(self.ControlPeriod1)+1]:
                if t%self.ControlPeriod1 == 0:
                    for i in range(1,self.ControlPeriod1):
                        for h in self.H:
                            self.m.Constraint_HP_T.add(self.m.B_HP_H_T[h,t] == self.m.B_HP_H_T[h,t+i])
        
        for t in self.T[self.TControlPeriodSwitch1:-(self.ControlPeriod2)]:
            if t%self.ControlPeriod2 == 0:
                for i in range(1,self.ControlPeriod2):
                    for h in self.H:
                        self.m.Constraint_HP_T.add(self.m.B_HP_H_T[h,t] == self.m.B_HP_H_T[h,t+i])

        if self.TControlPeriodSwitch1 > 0:
            for t in self.T[1:self.TControlPeriodSwitch1+1]:
                self.m.Constraint_HP_T.add(self.m.Z_HP_T_1[t] >= self.m.B_HP_H_T[1,t] + self.m.B_HP_H_T[0,t-1] - 1)
                self.m.Constraint_HP_T.add(self.m.Z_HP_T_1[t] >= self.m.B_HP_H_T[2,t] + self.m.B_HP_H_T[0,t-1] - 1)
                self.m.Constraint_HP_T.add(self.m.Z_HP_T_1[t] >= self.m.B_HP_H_T[3,t] + self.m.B_HP_H_T[0,t-1] - 1)
                self.m.Constraint_HP_T.add(self.m.Z_HP_T_1[t] >= self.m.B_HP_H_T[4,t] + self.m.B_HP_H_T[0,t-1] - 1)

                self.m.Constraint_HP_T.add(self.m.Z_HP_T_1[t] >= self.m.B_HP_H_T[3,t] + self.m.B_HP_H_T[1,t-1] - 1)
                self.m.Constraint_HP_T.add(self.m.Z_HP_T_1[t] >= self.m.B_HP_H_T[4,t] + self.m.B_HP_H_T[1,t-1] - 1)

                self.m.Constraint_HP_T.add(self.m.Z_HP_T_1[t] >= self.m.B_HP_H_T[3,t] + self.m.B_HP_H_T[2,t-1] - 1)
                self.m.Constraint_HP_T.add(self.m.Z_HP_T_1[t] >= self.m.B_HP_H_T[4,t] + self.m.B_HP_H_T[2,t-1] - 1)

                self.m.Constraint_HP_T.add(self.m.Z_HP_T_1[t] <= 1)

        for t in self.T[self.TControlPeriodSwitch1+1:-1]:
            self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[t] >= self.m.B_HP_H_T[1,t] + self.m.B_HP_H_T[0,t-1] - 1)
            self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[t] >= self.m.B_HP_H_T[2,t] + self.m.B_HP_H_T[0,t-1] - 1)
            self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[t] >= self.m.B_HP_H_T[3,t] + self.m.B_HP_H_T[0,t-1] - 1)
            self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[t] >= self.m.B_HP_H_T[4,t] + self.m.B_HP_H_T[0,t-1] - 1)

            self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[t] >= self.m.B_HP_H_T[3,t] + self.m.B_HP_H_T[1,t-1] - 1)
            self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[t] >= self.m.B_HP_H_T[4,t] + self.m.B_HP_H_T[1,t-1] - 1)

            self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[t] >= self.m.B_HP_H_T[3,t] + self.m.B_HP_H_T[2,t-1] - 1)
            self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[t] >= self.m.B_HP_H_T[4,t] + self.m.B_HP_H_T[2,t-1] - 1)

            self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[t] <= 1)
        
        if self.End_Toggle_Constraints == True:
            self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[self.T[-1]] >= self.B_HP_1_end + self.m.B_HP_H_T[0,self.T[-2]] - 1)
            self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[self.T[-1]] >= self.B_HP_2_end + self.m.B_HP_H_T[0,self.T[-2]] - 1)
            self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[self.T[-1]] >= self.B_HP_3_end + self.m.B_HP_H_T[0,self.T[-2]] - 1)
            self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[self.T[-1]] >= self.B_HP_4_end + self.m.B_HP_H_T[0,self.T[-2]] - 1)

            self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[self.T[-1]] >= self.B_HP_3_end + self.m.B_HP_H_T[1,self.T[-2]] - 1)
            self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[self.T[-1]] >= self.B_HP_4_end + self.m.B_HP_H_T[1,self.T[-2]] - 1)

            self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[self.T[-1]] >= self.B_HP_3_end + self.m.B_HP_H_T[2,self.T[-2]] - 1)
            self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[self.T[-1]] >= self.B_HP_4_end + self.m.B_HP_H_T[2,self.T[-2]] - 1)

            self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[self.T[-1]] <= 1)

        if self.Start_Toggle_Constraints == True:
            if self.TControlPeriodSwitch1 > 0:
                self.m.Constraint_HP_T.add(self.m.Z_HP_T_1[0] >= self.B_HP_1_start + self.m.B_HP_H_T[0,0] - 1)
                self.m.Constraint_HP_T.add(self.m.Z_HP_T_1[0] >= self.B_HP_2_start + self.m.B_HP_H_T[0,0] - 1)
                self.m.Constraint_HP_T.add(self.m.Z_HP_T_1[0] >= self.B_HP_3_start + self.m.B_HP_H_T[0,0] - 1)
                self.m.Constraint_HP_T.add(self.m.Z_HP_T_1[0] >= self.B_HP_4_start + self.m.B_HP_H_T[0,0] - 1)

                self.m.Constraint_HP_T.add(self.m.Z_HP_T_1[0] >= self.B_HP_3_start + self.m.B_HP_H_T[1,0] - 1)
                self.m.Constraint_HP_T.add(self.m.Z_HP_T_1[0] >= self.B_HP_4_start + self.m.B_HP_H_T[1,0] - 1)

                self.m.Constraint_HP_T.add(self.m.Z_HP_T_1[0] >= self.B_HP_3_start + self.m.B_HP_H_T[2,0] - 1)
                self.m.Constraint_HP_T.add(self.m.Z_HP_T_1[0] >= self.B_HP_4_start + self.m.B_HP_H_T[2,0] - 1)

                self.m.Constraint_HP_T.add(self.m.Z_HP_T_1[0] <= 1)
            if self.TControlPeriodSwitch1 == 0:
                self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[0] >= self.B_HP_1_start + self.m.B_HP_H_T[0,0] - 1)
                self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[0] >= self.B_HP_2_start + self.m.B_HP_H_T[0,0] - 1)
                self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[0] >= self.B_HP_3_start + self.m.B_HP_H_T[0,0] - 1)
                self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[0] >= self.B_HP_4_start + self.m.B_HP_H_T[0,0] - 1)

                self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[0] >= self.B_HP_3_start + self.m.B_HP_H_T[1,0] - 1)
                self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[0] >= self.B_HP_4_start + self.m.B_HP_H_T[1,0] - 1)

                self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[0] >= self.B_HP_3_start + self.m.B_HP_H_T[2,0] - 1)
                self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[0] >= self.B_HP_4_start + self.m.B_HP_H_T[2,0] - 1)

                self.m.Constraint_HP_T.add(self.m.Z_HP_T_2[0] <= 1)

        # HP HT Tank
        self.m.Constraint_HP_T.add(self.m.T_HP_HT_T[0] == self.T_HP_HT_start) ## Start temperature

        for t in self.T[0:-1]:
            self.m.Constraint_HP_T.add(self.m.T_HP_HT_T[t+1] == self.m.T_HP_HT_T[t] + self.StepSizeInSec * (1/(self.m_HP_HT_w * self.c_w) * (self.c_w * sum(self.mdot_HP_w_H[h] * self.m.Z_HP_HT_in_H_T[h,t] for h in self.H) + self.c_w * sum(self.mdot_HP_w_H[h] * self.m.Z_HP_HT_H_in_HXH_T[h,t] for h in self.H) + self.c_w * sum(self.mdot_HP_w_H[h] * self.m.Z_HP_HT_H_in_HS_T[h,t] for h in self.H))) + self.StepSizeInSec * self.alpha_HP_time * (self.t_default - self.m.T_HP_HT_T[t+1])/(self.m_HP_HT_w * self.c_w)) ## General energy flow
        
        for t in self.T[0:-1]:
            for h in self.H:
                self.m.Constraint_HP_T.add(self.m.Z_HP_HT_H_in_HXH_T[h,t] <= self.T_HP_delta_max) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_HT_H_in_HXH_T[h,t] >= self.T_HP_delta_min) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_HT_H_in_HXH_T[h,t] <= self.T_HP_delta_max * self.m.Z_HP_HXH_H_T[h,t]) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_HT_H_in_HXH_T[h,t] >= self.T_HP_delta_min * self.m.Z_HP_HXH_H_T[h,t]) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_HT_H_in_HXH_T[h,t] <= (self.m.T_HP_HT_in_HXH_T[t] - self.m.T_HP_HT_T[t+1]) - (1 - self.m.Z_HP_HXH_H_T[h,t]) * self.T_HP_delta_min) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_HT_H_in_HXH_T[h,t] >= (self.m.T_HP_HT_in_HXH_T[t] - self.m.T_HP_HT_T[t+1]) - (1 - self.m.Z_HP_HXH_H_T[h,t]) * self.T_HP_delta_max) ## Big M constraint input

                self.m.Constraint_HP_T.add(self.m.Z_HP_HT_H_in_HS_T[h,t] <= self.T_HP_delta_max) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_HT_H_in_HS_T[h,t] >= self.T_HP_delta_min) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_HT_H_in_HS_T[h,t] <= self.T_HP_delta_max * self.m.Z_HP_HS_H_T[h,t]) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_HT_H_in_HS_T[h,t] >= self.T_HP_delta_min * self.m.Z_HP_HS_H_T[h,t]) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_HT_H_in_HS_T[h,t] <= (self.m.T_HP_HT_in_HS_T[t] - self.m.T_HP_HT_T[t+1]) - (1 - self.m.Z_HP_HS_H_T[h,t]) * self.T_HP_delta_min) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_HT_H_in_HS_T[h,t] >= (self.m.T_HP_HT_in_HS_T[t] - self.m.T_HP_HT_T[t+1]) - (1 - self.m.Z_HP_HS_H_T[h,t]) * self.T_HP_delta_max) ## Big M constraint input

                self.m.Constraint_HP_T.add(self.m.Z_HP_HT_in_H_T[h,t] <= self.T_HP_delta_max) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_HT_in_H_T[h,t] >= self.T_HP_delta_min) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_HT_in_H_T[h,t] <= self.T_HP_delta_max * self.m.B_HP_H_T[h,t]) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_HT_in_H_T[h,t] >= self.T_HP_delta_min * self.m.B_HP_H_T[h,t]) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_HT_in_H_T[h,t] <= (self.m.T_HP_HT_out_T[t] - self.m.T_HP_HT_T[t+1]) - (1 - self.m.B_HP_H_T[h,t]) * self.T_HP_delta_min) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_HT_in_H_T[h,t] >= (self.m.T_HP_HT_out_T[t] - self.m.T_HP_HT_T[t+1]) - (1 - self.m.B_HP_H_T[h,t]) * self.T_HP_delta_max) ## Big M constraint input

        for t in self.T[0:-1]:
            self.m.Constraint_HP_T.add(self.m.T_HP_HT_in_T[t] == self.m.T_HP_HT_T[t+1])

        for t in self.T[1:]:
            self.m.Constraint_HP_T.add(self.m.T_HP_HT_T[t] <= self.T_HP_HT_max + self.m.S_T_HP_T[t]) ## Temperature range tank
            self.m.Constraint_HP_T.add(self.m.T_HP_HT_T[t] >= self.T_HP_HT_min - self.m.S_T_HP_T[t]) ## Temperature range tank

        # HP LT Tank
        self.m.Constraint_HP_T.add(self.m.T_HP_LT_T[0] == self.T_HP_LT_start) ## Start temperature

        for t in self.T[0:-1]:
            self.m.Constraint_HP_T.add(self.m.T_HP_LT_T[t+1] == self.m.T_HP_LT_T[t] + self.StepSizeInSec * (1/(self.m_HP_LT_b * self.c_b) * (self.c_b * sum(self.mdot_HP_b_H[h] * self.m.Z_HP_LT_in_H_T[h,t] for h in self.H) + self.c_b * sum(self.mdot_HP_b_H[h] * self.m.Z_HP_LT_H_in_HGC_T[h,t] for h in self.H))) + self.StepSizeInSec * self.alpha_HP_time * (self.t_default - self.m.T_HP_LT_T[t+1])/(self.m_HP_LT_b * self.c_b)) ## General energy flow
        
        for t in self.T[0:-1]:
            for h in self.H:
                self.m.Constraint_HP_T.add(self.m.Z_HP_LT_H_in_HGC_T[h,t] <= self.T_HP_delta_max) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_LT_H_in_HGC_T[h,t] >= self.T_HP_delta_min) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_LT_H_in_HGC_T[h,t] <= self.T_HP_delta_max * self.m.B_HP_H_T[h,t]) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_LT_H_in_HGC_T[h,t] >= self.T_HP_delta_min * self.m.B_HP_H_T[h,t]) ## Big M constraint input    ######### self.T_HP_delta_min
                self.m.Constraint_HP_T.add(self.m.Z_HP_LT_H_in_HGC_T[h,t] <= (self.m.T_HP_LT_in_HGC_T[t] - self.m.T_HP_LT_T[t+1]) - (1 - self.m.B_HP_H_T[h,t]) * self.T_HP_delta_min) ## Big M constraint input   HERE!!
                self.m.Constraint_HP_T.add(self.m.Z_HP_LT_H_in_HGC_T[h,t] >= (self.m.T_HP_LT_in_HGC_T[t] - self.m.T_HP_LT_T[t+1]) - (1 - self.m.B_HP_H_T[h,t]) * self.T_HP_delta_max) ## Big M constraint input

                self.m.Constraint_HP_T.add(self.m.Z_HP_LT_in_H_T[h,t] <= self.T_HP_delta_max) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_LT_in_H_T[h,t] >= self.T_HP_delta_min) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_LT_in_H_T[h,t] <= self.T_HP_delta_max * self.m.B_HP_H_T[h,t]) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_LT_in_H_T[h,t] >= self.T_HP_delta_min * self.m.B_HP_H_T[h,t]) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_LT_in_H_T[h,t] <= (self.m.T_HP_LT_out_T[t] - self.m.T_HP_LT_T[t+1]) - (1 - self.m.B_HP_H_T[h,t]) * self.T_HP_delta_min) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HP_LT_in_H_T[h,t] >= (self.m.T_HP_LT_out_T[t] - self.m.T_HP_LT_T[t+1]) - (1 - self.m.B_HP_H_T[h,t]) * self.T_HP_delta_max) ## Big M constraint input

        for t in self.T[0:-1]:
            self.m.Constraint_HP_T.add(self.m.T_HP_LT_in_T[t] == self.m.T_HP_LT_T[t+1])

        for t in self.T[1:]:
            self.m.Constraint_HP_T.add(self.m.T_HP_LT_T[t] <= self.T_HP_LT_max + self.m.S_T_HP_T[t]) ## Temperature range tank
            self.m.Constraint_HP_T.add(self.m.T_HP_LT_T[t] >= self.T_HP_LT_min - self.m.S_T_HP_T[t]) ## Temperature range tank
        
        ## HS
        self.m.Constraint_HS_T = pyo.ConstraintList()
        self.m.Constraint_HS_T.add(self.m.T_HS_T[0] == self.T_HS_start) ## Start temperature
        if self.End_Temp_Constraints == True:
           self.m.Constraint_HS_T.add(self.m.T_HS_T[self.T[-1]] >= self.T_HS_end - self.m.S_T_HS_T[self.T[-1]])

        for t in self.T[0:-1]:
            self.m.Constraint_HS_T.add(self.m.T_HS_T[t+1] == self.m.T_HS_T[t] + self.StepSizeInSec * (1/(self.m_HS_w * self.c_w) * (self.c_w * sum(self.mdot_HP_w_H[h] * self.m.Z_HS_HT_H_T[h,t] for h in self.H) + self.c_w * self.mdot_IS_w_2 * self.m.Z_HS_LT_T[t] + self.c_w * self.mdot_IS_w_2 * self.m.Z_HS_LT_2_T[t])) - self.StepSizeInSec * self.q_dem_HS_T[t]/(self.m_HS_w * self.c_w) + self.StepSizeInSec * self.alpha_HS_time * (self.t_default - self.m.T_HS_T[t+1])/(self.m_HS_w * self.c_w)) ## General energy flow
        
        for t in self.T[0:-1]:
            for h in self.H:
                self.m.Constraint_HS_T.add(self.m.Z_HS_HT_H_T[h,t] <= self.T_HS_delta_max) ## Big M constraint input
                self.m.Constraint_HS_T.add(self.m.Z_HS_HT_H_T[h,t] >= self.T_HS_delta_min) ## Big M constraint input
                self.m.Constraint_HS_T.add(self.m.Z_HS_HT_H_T[h,t] <= self.T_HS_delta_max * self.m.Z_HP_HS_H_T[h,t]) ## Big M constraint input
                self.m.Constraint_HS_T.add(self.m.Z_HS_HT_H_T[h,t] >= self.T_HS_delta_min * self.m.Z_HP_HS_H_T[h,t]) ## Big M constraint input
                self.m.Constraint_HS_T.add(self.m.Z_HS_HT_H_T[h,t] <= (self.m.T_HS_HT_T[t] - self.m.T_HS_T[t+1]) - (1 - self.m.Z_HP_HS_H_T[h,t]) * self.T_HS_delta_min) ## Big M constraint input
                self.m.Constraint_HS_T.add(self.m.Z_HS_HT_H_T[h,t] >= (self.m.T_HS_HT_T[t] - self.m.T_HS_T[t+1]) - (1 - self.m.Z_HP_HS_H_T[h,t]) * self.T_HS_delta_max) ## Big M constraint input

            self.m.Constraint_HS_T.add(self.m.Z_HS_LT_T[t] <= self.T_HS_delta_max) ## Big M constraint input
            self.m.Constraint_HS_T.add(self.m.Z_HS_LT_T[t] >= self.T_HS_delta_min) ## Big M constraint input
            self.m.Constraint_HS_T.add(self.m.Z_HS_LT_T[t] <= self.T_HS_delta_max * self.m.B_HS_IS_T[t]) ## Big M constraint input
            self.m.Constraint_HS_T.add(self.m.Z_HS_LT_T[t] >= self.T_HS_delta_min * self.m.B_HS_IS_T[t]) ## Big M constraint input
            self.m.Constraint_HS_T.add(self.m.Z_HS_LT_T[t] <= (self.m.T_HS_LT_T[t] - self.m.T_HS_T[t+1]) - (1 - self.m.B_HS_IS_T[t]) * self.T_HS_delta_min) ## Big M constraint input
            self.m.Constraint_HS_T.add(self.m.Z_HS_LT_T[t] >= (self.m.T_HS_LT_T[t] - self.m.T_HS_T[t+1]) - (1 - self.m.B_HS_IS_T[t]) * self.T_HS_delta_max) ## Big M constraint input

            self.m.Constraint_HS_T.add(self.m.Z_HS_LT_2_T[t] <= self.T_HS_delta_max) ## Big M constraint input
            self.m.Constraint_HS_T.add(self.m.Z_HS_LT_2_T[t] >= self.T_HS_delta_min) ## Big M constraint input
            self.m.Constraint_HS_T.add(self.m.Z_HS_LT_2_T[t] <= self.T_HS_delta_max * self.m.Z_HS_IS_2_T[t]) ## Big M constraint input
            self.m.Constraint_HS_T.add(self.m.Z_HS_LT_2_T[t] >= self.T_HS_delta_min * self.m.Z_HS_IS_2_T[t]) ## Big M constraint input
            self.m.Constraint_HS_T.add(self.m.Z_HS_LT_2_T[t] <= (self.m.T_HS_LT_T[t] - self.m.T_HS_T[t+1]) - (1 - self.m.Z_HS_IS_2_T[t]) * self.T_HS_delta_min) ## Big M constraint input
            self.m.Constraint_HS_T.add(self.m.Z_HS_LT_2_T[t] >= (self.m.T_HS_LT_T[t] - self.m.T_HS_T[t+1]) - (1 - self.m.Z_HS_IS_2_T[t]) * self.T_HS_delta_max) ## Big M constraint input

        for t in self.T[1:]:
            self.m.Constraint_HS_T.add(self.m.T_HS_T[t] <= self.T_HS_max + self.m.S_T_HS_T[t]) ## Temperature range tank
            self.m.Constraint_HS_T.add(self.m.T_HS_T[t] >= self.T_HS_min - self.m.S_T_HS_T[t]) ## Temperature range tank

        ## CS
        self.m.Constraint_CS_T = pyo.ConstraintList()
        self.m.Constraint_CS_T.add(self.m.T_CS_T[0] == self.T_CS_start) ## Start temperature
        if self.End_Temp_Constraints == True:
            self.m.Constraint_CS_T.add(self.m.T_CS_T[self.T[-1]] <= self.T_CS_end + self.m.S_T_CS_T[self.T[-1]])

        if self.TControlPeriodSwitch1 > 0:
            for t in self.T[0:self.TControlPeriodSwitch1]:
                self.m.Constraint_CS_T.add(self.m.T_CS_T[t+1] == self.m.T_CS_T[t] + self.StepSizeInSec * (1/(self.m_CS_w * self.c_w) * (self.c_w * self.mdot_GS_w * self.m.Z_CS_GS_T[t] + self.c_w * self.mdot_GS_w_2 * self.m.Z_CS_GS_2_T[t] + self.c_w * sum(self.mdot_VP_CS_V_1[v] * self.m.Z_CS_HXC_V_T_1[v,t] for v in self.V_1))) + self.StepSizeInSec * self.q_dem_CS_T[t]/(self.m_CS_w * self.c_w) + self.StepSizeInSec * self.alpha_CS_time * (self.t_default - self.m.T_CS_T[t+1])/(self.m_CS_w * self.c_w)) ## General energy flow
            
        for t in self.T[self.TControlPeriodSwitch1:-1]:
            self.m.Constraint_CS_T.add(self.m.T_CS_T[t+1] == self.m.T_CS_T[t] + self.StepSizeInSec * (1/(self.m_CS_w * self.c_w) * (self.c_w * self.mdot_GS_w * self.m.Z_CS_GS_T[t] + self.c_w * self.mdot_GS_w_2 * self.m.Z_CS_GS_2_T[t] + self.c_w * sum(self.mdot_VP_CS_V_2[v] * self.m.Z_CS_HXC_V_T_2[v,t] for v in self.V_2))) + self.StepSizeInSec * self.q_dem_CS_T[t]/(self.m_CS_w * self.c_w) + self.StepSizeInSec * self.alpha_CS_time * (self.t_default - self.m.T_CS_T[t+1])/(self.m_CS_w * self.c_w)) ## General energy flow
 
        for t in self.T[0:-1]:
            self.m.Constraint_CS_T.add(self.m.Z_CS_GS_T[t] <= self.T_CS_delta_max) ## Big M constraint input
            self.m.Constraint_CS_T.add(self.m.Z_CS_GS_T[t] >= self.T_CS_delta_min) ## Big M constraint input
            self.m.Constraint_CS_T.add(self.m.Z_CS_GS_T[t] <= self.T_CS_delta_max * self.m.B_GS_CS_T[t]) ## Big M constraint input
            self.m.Constraint_CS_T.add(self.m.Z_CS_GS_T[t] >= self.T_CS_delta_min * self.m.B_GS_CS_T[t]) ## Big M constraint input
            self.m.Constraint_CS_T.add(self.m.Z_CS_GS_T[t] <= (self.m.T_CS_GS_T[t] - self.m.T_CS_T[t+1]) - (1 - self.m.B_GS_CS_T[t]) * self.T_CS_delta_min) ## Big M constraint input
            self.m.Constraint_CS_T.add(self.m.Z_CS_GS_T[t] >= (self.m.T_CS_GS_T[t] - self.m.T_CS_T[t+1]) - (1 - self.m.B_GS_CS_T[t]) * self.T_CS_delta_max) ## Big M constraint input

            self.m.Constraint_CS_T.add(self.m.Z_CS_GS_2_T[t] <= self.T_CS_delta_max) ## Big M constraint input
            self.m.Constraint_CS_T.add(self.m.Z_CS_GS_2_T[t] >= self.T_CS_delta_min) ## Big M constraint input
            self.m.Constraint_CS_T.add(self.m.Z_CS_GS_2_T[t] <= self.T_CS_delta_max * self.m.B_GS_HGS_CS_T[t]) ## Big M constraint input
            self.m.Constraint_CS_T.add(self.m.Z_CS_GS_2_T[t] >= self.T_CS_delta_min * self.m.B_GS_HGS_CS_T[t]) ## Big M constraint input
            self.m.Constraint_CS_T.add(self.m.Z_CS_GS_2_T[t] <= (self.m.T_CS_GS_T[t] - self.m.T_CS_T[t+1]) - (1 - self.m.B_GS_HGS_CS_T[t]) * self.T_CS_delta_min) ## Big M constraint input
            self.m.Constraint_CS_T.add(self.m.Z_CS_GS_2_T[t] >= (self.m.T_CS_GS_T[t] - self.m.T_CS_T[t+1]) - (1 - self.m.B_GS_HGS_CS_T[t]) * self.T_CS_delta_max) ## Big M constraint input

        if self.TControlPeriodSwitch1 > 0:
            for t in self.T[0:self.TControlPeriodSwitch1]:
                for v in self.V_1:
                    self.m.Constraint_CS_T.add(self.m.Z_CS_HXC_V_T_1[v,t] <= self.T_CS_delta_max) ## Big M constraint input
                    self.m.Constraint_CS_T.add(self.m.Z_CS_HXC_V_T_1[v,t] >= self.T_CS_delta_min) ## Big M constraint input
                    self.m.Constraint_CS_T.add(self.m.Z_CS_HXC_V_T_1[v,t] <= self.T_CS_delta_max * self.m.B_VP_V_T_1[v,t]) ## Big M constraint input
                    self.m.Constraint_CS_T.add(self.m.Z_CS_HXC_V_T_1[v,t] >= self.T_CS_delta_min * self.m.B_VP_V_T_1[v,t]) ## Big M constraint input
                    self.m.Constraint_CS_T.add(self.m.Z_CS_HXC_V_T_1[v,t] <= (self.m.T_CS_HXC_T[t] - self.m.T_CS_T[t+1]) - (1 - self.m.B_VP_V_T_1[v,t]) * self.T_CS_delta_min) ## Big M constraint input
                    self.m.Constraint_CS_T.add(self.m.Z_CS_HXC_V_T_1[v,t] >= (self.m.T_CS_HXC_T[t] - self.m.T_CS_T[t+1]) - (1 - self.m.B_VP_V_T_1[v,t]) * self.T_CS_delta_max) ## Big M constraint input
                
                self.m.Constraint_CS_T.add((self.m.T_CS_HXC_T[t] - self.m.T_CS_T[t+1]) == sum(self.m.Z_CS_HXC_V_T_1[v,t] for v in self.V_1)) ## Thighten Relaxation problem

        for t in self.T[self.TControlPeriodSwitch1:-1]:
            for v in self.V_2:
                self.m.Constraint_CS_T.add(self.m.Z_CS_HXC_V_T_2[v,t] <= self.T_CS_delta_max) ## Big M constraint input
                self.m.Constraint_CS_T.add(self.m.Z_CS_HXC_V_T_2[v,t] >= self.T_CS_delta_min) ## Big M constraint input
                self.m.Constraint_CS_T.add(self.m.Z_CS_HXC_V_T_2[v,t] <= self.T_CS_delta_max * self.m.B_VP_V_T_2[v,t]) ## Big M constraint input
                self.m.Constraint_CS_T.add(self.m.Z_CS_HXC_V_T_2[v,t] >= self.T_CS_delta_min * self.m.B_VP_V_T_2[v,t]) ## Big M constraint input
                self.m.Constraint_CS_T.add(self.m.Z_CS_HXC_V_T_2[v,t] <= (self.m.T_CS_HXC_T[t] - self.m.T_CS_T[t+1]) - (1 - self.m.B_VP_V_T_2[v,t]) * self.T_CS_delta_min) ## Big M constraint input
                self.m.Constraint_CS_T.add(self.m.Z_CS_HXC_V_T_2[v,t] >= (self.m.T_CS_HXC_T[t] - self.m.T_CS_T[t+1]) - (1 - self.m.B_VP_V_T_2[v,t]) * self.T_CS_delta_max) ## Big M constraint input
            
            self.m.Constraint_CS_T.add((self.m.T_CS_HXC_T[t] - self.m.T_CS_T[t+1]) == sum(self.m.Z_CS_HXC_V_T_2[v,t] for v in self.V_2)) ## Thighten Relaxation problem

        for t in self.T[1:]:
            self.m.Constraint_CS_T.add(self.m.T_CS_T[t] <= self.T_CS_max + self.m.S_T_CS_T[t]) ## Temperature range tank
            self.m.Constraint_CS_T.add(self.m.T_CS_T[t] >= self.T_CS_min - self.m.S_T_CS_T[t]) ## Temperature range tank

        ## RLTS
        self.m.Constraint_RLTS_T = pyo.ConstraintList()
        self.m.Constraint_RLTS_T.add(self.m.T_RLTS_T[0] == self.T_RLTS_start) ## Start temperature
        if self.End_Temp_Constraints == True:
            self.m.Constraint_RLTS_T.add(self.m.T_RLTS_T[self.T[-1]] <= self.T_RLTS_end + self.m.S_T_RLTS_T[self.T[-1]])
        
        if self.TControlPeriodSwitch1 > 0:
            for t in self.T[0:self.TControlPeriodSwitch1]:
                self.m.Constraint_RLTS_T.add(self.m.T_RLTS_T[t+1] == self.m.T_RLTS_T[t] + self.StepSizeInSec * (1/(self.m_RLTS_w * self.c_w) * (self.c_w * sum(self.mdot_VP_RLTS_V_2[v] * self.m.Z_RLTS_HXC_V_T_1[v,t] for v in self.V_1))) + self.StepSizeInSec * self.q_dem_RLTS_T[t]/(self.m_RLTS_w * self.c_w) + self.StepSizeInSec * self.alpha_RLTS_time * (self.t_default - self.m.T_RLTS_T[t+1])/(self.m_RLTS_w * self.c_w)) ## General energy flow
        
        for t in self.T[self.TControlPeriodSwitch1:-1]:
            self.m.Constraint_RLTS_T.add(self.m.T_RLTS_T[t+1] == self.m.T_RLTS_T[t] + self.StepSizeInSec * (1/(self.m_RLTS_w * self.c_w) * (self.c_w * sum(self.mdot_VP_RLTS_V_2[v] * self.m.Z_RLTS_HXC_V_T_2[v,t] for v in self.V_2))) + self.StepSizeInSec * self.q_dem_RLTS_T[t]/(self.m_RLTS_w * self.c_w) + self.StepSizeInSec * self.alpha_RLTS_time * (self.t_default - self.m.T_RLTS_T[t+1])/(self.m_RLTS_w * self.c_w)) ## General energy flow
        
        if self.TControlPeriodSwitch1 > 0:
            for t in self.T[0:self.TControlPeriodSwitch1]:
                for v in self.V_1:
                    self.m.Constraint_RLTS_T.add(self.m.Z_RLTS_HXC_V_T_1[v,t] <= self.T_RLTS_delta_max) ## Big M constraint input
                    self.m.Constraint_RLTS_T.add(self.m.Z_RLTS_HXC_V_T_1[v,t] >= self.T_RLTS_delta_min) ## Big M constraint input
                    self.m.Constraint_RLTS_T.add(self.m.Z_RLTS_HXC_V_T_1[v,t] <= self.T_RLTS_delta_max * self.m.B_VP_V_T_1[v,t]) ## Big M constraint input
                    self.m.Constraint_RLTS_T.add(self.m.Z_RLTS_HXC_V_T_1[v,t] >= self.T_RLTS_delta_min * self.m.B_VP_V_T_1[v,t]) ## Big M constraint input
                    self.m.Constraint_RLTS_T.add(self.m.Z_RLTS_HXC_V_T_1[v,t] <= (self.m.T_RLTS_HXC_T[t] - self.m.T_RLTS_T[t+1]) - (1 - self.m.B_VP_V_T_1[v,t]) * self.T_RLTS_delta_min) ## Big M constraint input
                    self.m.Constraint_RLTS_T.add(self.m.Z_RLTS_HXC_V_T_1[v,t] >= (self.m.T_RLTS_HXC_T[t] - self.m.T_RLTS_T[t+1]) - (1 - self.m.B_VP_V_T_1[v,t]) * self.T_RLTS_delta_max) ## Big M constraint input
                
                self.m.Constraint_RLTS_T.add((self.m.T_RLTS_HXC_T[t] - self.m.T_RLTS_T[t+1]) == sum(self.m.Z_RLTS_HXC_V_T_1[v,t] for v in self.V_1)) ## Thighten Relaxation problem

        for t in self.T[self.TControlPeriodSwitch1:-1]:
            for v in self.V_2:
                self.m.Constraint_RLTS_T.add(self.m.Z_RLTS_HXC_V_T_2[v,t] <= self.T_RLTS_delta_max) ## Big M constraint input
                self.m.Constraint_RLTS_T.add(self.m.Z_RLTS_HXC_V_T_2[v,t] >= self.T_RLTS_delta_min) ## Big M constraint input
                self.m.Constraint_RLTS_T.add(self.m.Z_RLTS_HXC_V_T_2[v,t] <= self.T_RLTS_delta_max * self.m.B_VP_V_T_2[v,t]) ## Big M constraint input
                self.m.Constraint_RLTS_T.add(self.m.Z_RLTS_HXC_V_T_2[v,t] >= self.T_RLTS_delta_min * self.m.B_VP_V_T_2[v,t]) ## Big M constraint input
                self.m.Constraint_RLTS_T.add(self.m.Z_RLTS_HXC_V_T_2[v,t] <= (self.m.T_RLTS_HXC_T[t] - self.m.T_RLTS_T[t+1]) - (1 - self.m.B_VP_V_T_2[v,t]) * self.T_RLTS_delta_min) ## Big M constraint input
                self.m.Constraint_RLTS_T.add(self.m.Z_RLTS_HXC_V_T_2[v,t] >= (self.m.T_RLTS_HXC_T[t] - self.m.T_RLTS_T[t+1]) - (1 - self.m.B_VP_V_T_2[v,t]) * self.T_RLTS_delta_max) ## Big M constraint input
            
            self.m.Constraint_RLTS_T.add((self.m.T_RLTS_HXC_T[t] - self.m.T_RLTS_T[t+1]) == sum(self.m.Z_RLTS_HXC_V_T_2[v,t] for v in self.V_2)) ## Thighten Relaxation problem

        for t in self.T[1:]:
            self.m.Constraint_RLTS_T.add(self.m.T_RLTS_T[t] <= self.T_RLTS_max + self.m.S_T_RLTS_T[t]) ## Temperature range tank
            self.m.Constraint_RLTS_T.add(self.m.T_RLTS_T[t] >= self.T_RLTS_min - self.m.S_T_RLTS_T[t]) ## Temperature range tank

        ## IS
        self.m.Constraint_IS_T = pyo.ConstraintList()
        for t in self.T[:-1]:
            for r in self.cr_IS[1:]:
                self.m.Constraint_IS_T.add(self.m.Q_IS_C_NORTH_T_CR[t,r] == (self.m.T_IS_C_T_CR[t+1,r-1] - self.m.T_IS_C_T_CR[t+1,r]) * self.lambda_IS_c_c / self.height_IS * self.a_north_south_IS)
    
            for r in self.cr_IS[:-1]:
                self.m.Constraint_IS_T.add(self.m.Q_IS_C_SOUTH_T_CR[t,r] == (self.m.T_IS_C_T_CR[t+1,r+1] - self.m.T_IS_C_T_CR[t+1,r]) * self.lambda_IS_c_c / self.height_IS * self.a_north_south_IS)

            for r in self.wr_IS:
                self.m.Constraint_IS_T.add(self.m.Q_IS_C_W_T_WR[t,r] == (self.m.T_IS_W_T_WR[t+1,r] - self.m.T_IS_C_T_CR[t+1,r]) * self.alpha_IS_w_c * self.a_pipe_IS)

        ## Concrete borders
        for t in self.T[:-1]:
            self.m.Constraint_IS_T.add(self.m.Q_IS_C_NORTH_T_CR[t,0] == (self.t_IS_air - self.m.T_IS_C_T_CR[t+1,0]) * self.lambda_IS_c_a / self.height_IS * self.a_north_south_IS)
            self.m.Constraint_IS_T.add(self.m.Q_IS_C_SOUTH_T_CR[t,self.cr_IS[-1]] == (self.t_IS_air - self.m.T_IS_C_T_CR[t+1,self.cr_IS[-1]]) * self.lambda_IS_c_a / self.height_IS * self.a_north_south_IS)  

        ## Concrete temperature
            self.m.Constraint_IS_T.add(self.m.T_IS_C_T_CR[0,0] == self.T_IS_c_1_start)
            self.m.Constraint_IS_T.add(self.m.T_IS_C_T_CR[0,1] == self.T_IS_c_2_start)
            self.m.Constraint_IS_T.add(self.m.T_IS_C_T_CR[0,2] == self.T_IS_c_3_start)
            self.m.Constraint_IS_T.add(self.m.T_IS_C_T_CR[0,3] == self.T_IS_c_4_start)
            self.m.Constraint_IS_T.add(self.m.T_IS_C_T_CR[0,4] == self.T_IS_c_5_start)

        for t in self.T[:-1]:
            for r in self.cr_IS:
                if r in self.wr_IS:
                    self.m.Constraint_IS_T.add(self.m.T_IS_C_T_CR[t+1,r] == self.m.T_IS_C_T_CR[t,r] + self.StepSizeInSec * (1/(self.m_IS_c * self.c_c)) * (self.m.Q_IS_C_NORTH_T_CR[t,r] + self.m.Q_IS_C_SOUTH_T_CR[t,r] + self.m.Q_IS_C_W_T_WR[t,r]))
                else:
                    self.m.Constraint_IS_T.add(self.m.T_IS_C_T_CR[t+1,r] == self.m.T_IS_C_T_CR[t,r] + self.StepSizeInSec * (1/(self.m_IS_c * self.c_c)) * (self.m.Q_IS_C_NORTH_T_CR[t,r] + self.m.Q_IS_C_SOUTH_T_CR[t,r]))

        for t in self.T[1:]:
            for r in self.cr_IS:
                self.m.Constraint_IS_T.add(self.m.T_IS_C_T_CR[t,r] <= self.T_IS_max_c + self.m.S_T_IS_C_T_CR[t,r])
                self.m.Constraint_IS_T.add(self.m.T_IS_C_T_CR[t,r] >= self.T_IS_min_c - self.m.S_T_IS_C_T_CR[t,r])

        ## Water energy flows
        for t in self.T[:-1]:
            for r in self.wr_IS[1:]:
                self.m.Constraint_IS_T.add(self.m.Q_IS_W_T_WR[t,r] <= self.T_IS_delta_max * self.mdot_IS_w/self.n_IS_blocks * self.c_w) ## Big M constraint input
                self.m.Constraint_IS_T.add(self.m.Q_IS_W_T_WR[t,r] >= self.T_IS_delta_min * self.mdot_IS_w/self.n_IS_blocks * self.c_w) ## Big M constraint input
                self.m.Constraint_IS_T.add(self.m.Q_IS_W_T_WR[t,r] <= self.T_IS_delta_max * self.mdot_IS_w/self.n_IS_blocks * self.c_w * self.m.Z_IS_pump_T[t]) ## Big M constraint input
                self.m.Constraint_IS_T.add(self.m.Q_IS_W_T_WR[t,r] >= self.T_IS_delta_min * self.mdot_IS_w/self.n_IS_blocks * self.c_w * self.m.Z_IS_pump_T[t]) ## Big M constraint input
                self.m.Constraint_IS_T.add(self.m.Q_IS_W_T_WR[t,r] <= (self.m.T_IS_W_T_WR[t+1,r-2] - self.m.T_IS_W_T_WR[t+1,r]) * self.mdot_IS_w/self.n_IS_blocks * self.c_w - (1 - self.m.Z_IS_pump_T[t]) * self.T_IS_delta_min * self.mdot_IS_w/self.n_IS_blocks * self.c_w) ## Big M constraint input
                self.m.Constraint_IS_T.add(self.m.Q_IS_W_T_WR[t,r] >= (self.m.T_IS_W_T_WR[t+1,r-2] - self.m.T_IS_W_T_WR[t+1,r]) * self.mdot_IS_w/self.n_IS_blocks * self.c_w - (1 - self.m.Z_IS_pump_T[t]) * self.T_IS_delta_max * self.mdot_IS_w/self.n_IS_blocks * self.c_w) ## Big M constraint input
            for r in self.wr_IS[0:1]:
                self.m.Constraint_IS_T.add(self.m.Q_IS_W_T_WR[t,r] == 0)

            for r in self.wr_IS:
                self.m.Constraint_IS_T.add(self.m.Q_IS_W_C_T_WR[t,r] == (self.m.T_IS_C_T_CR[t+1,r] - self.m.T_IS_W_T_WR[t+1,r]) * self.alpha_IS_w_c * self.a_pipe_IS)      
        
        ## Water borders
        for t in self.T[:-1]:
            self.m.Constraint_IS_T.add(self.m.Q_IS_W_T_IN[t] == self.m.Z_IS_HT_T[t] * self.c_w  * self.mdot_IS_w_2/self.n_IS_blocks + self.m.Z_IS_LT_T[t] * self.c_w  * self.mdot_IS_w_2/self.n_IS_blocks + self.m.Z_IS_HT_2_T[t] * self.c_w  * self.mdot_IS_w_2/self.n_IS_blocks + self.m.Z_IS_LT_2_T[t] * self.c_w  * self.mdot_IS_w_2/self.n_IS_blocks)

        ## Water temperature
        self.m.Constraint_IS_T.add(self.m.T_IS_W_T_WR[0,0] == self.T_IS_w_1_start)
        self.m.Constraint_IS_T.add(self.m.T_IS_W_T_WR[0,2] == self.T_IS_w_2_start)
        self.m.Constraint_IS_T.add(self.m.T_IS_W_T_WR[0,4] == self.T_IS_w_3_start)

        for t in self.T[:-1]:
            for r in self.wr_IS:
                if r == 0:
                    self.m.Constraint_IS_T.add(self.m.T_IS_W_T_WR[t+1,r] == self.m.T_IS_W_T_WR[t,r] + self.StepSizeInSec * (1/(self.m_IS_w * self.c_w)) * (self.m.Q_IS_W_C_T_WR[t,r] + self.m.Q_IS_W_T_IN[t]))
                else:
                    self.m.Constraint_IS_T.add(self.m.T_IS_W_T_WR[t+1,r] == self.m.T_IS_W_T_WR[t,r] + self.StepSizeInSec * (1/(self.m_IS_w * self.c_w)) * (self.m.Q_IS_W_T_WR[t,r] + self.m.Q_IS_W_C_T_WR[t,r]))

        for t in self.T[1:]:
            for r in self.wr_IS:
                self.m.Constraint_IS_T.add(self.m.T_IS_W_T_WR[t,r] <= self.T_IS_max_w + self.m.S_T_IS_W_T_WR[t,r])
                self.m.Constraint_IS_T.add(self.m.T_IS_W_T_WR[t,r] >= self.T_IS_min_w - self.m.S_T_IS_W_T_WR[t,r])

        ## Water outflow
        for t in self.T:
                self.m.Constraint_IS_T.add(self.m.T_IS_T[t] == self.m.T_IS_W_T_WR[t,self.wr_IS[-1]])
        # Old
        for t in self.T[0:-1]:
            self.m.Constraint_IS_T.add(self.m.Z_IS_HT_T[t] <= self.T_IS_delta_max) ## Big M constraint input
            self.m.Constraint_IS_T.add(self.m.Z_IS_HT_T[t] >= self.T_IS_delta_min) ## Big M constraint input
            self.m.Constraint_IS_T.add(self.m.Z_IS_HT_T[t] <= self.T_IS_delta_max * self.m.B_HS_IS_T[t]) ## Big M constraint input
            self.m.Constraint_IS_T.add(self.m.Z_IS_HT_T[t] >= self.T_IS_delta_min * self.m.B_HS_IS_T[t]) ## Big M constraint input
            self.m.Constraint_IS_T.add(self.m.Z_IS_HT_T[t] <= (self.m.T_IS_HT_T[t] - self.m.T_IS_W_T_WR[t+1,0]) - (1 - self.m.B_HS_IS_T[t]) * self.T_IS_delta_min) ## Big M constraint input
            self.m.Constraint_IS_T.add(self.m.Z_IS_HT_T[t] >= (self.m.T_IS_HT_T[t] - self.m.T_IS_W_T_WR[t+1,0]) - (1 - self.m.B_HS_IS_T[t]) * self.T_IS_delta_max) ## Big M constraint input

            self.m.Constraint_IS_T.add(self.m.Z_IS_LT_T[t] <= self.T_IS_delta_max) ## Big M constraint input
            self.m.Constraint_IS_T.add(self.m.Z_IS_LT_T[t] >= self.T_IS_delta_min) ## Big M constraint input
            self.m.Constraint_IS_T.add(self.m.Z_IS_LT_T[t] <= self.T_IS_delta_max * self.m.B_IS_HGS_T[t]) ## Big M constraint input
            self.m.Constraint_IS_T.add(self.m.Z_IS_LT_T[t] >= self.T_IS_delta_min * self.m.B_IS_HGS_T[t]) ## Big M constraint input
            self.m.Constraint_IS_T.add(self.m.Z_IS_LT_T[t] <= (self.m.T_IS_LT_T[t] - self.m.T_IS_W_T_WR[t+1,0]) - (1 - self.m.B_IS_HGS_T[t]) * self.T_IS_delta_min) ## Big M constraint input
            self.m.Constraint_IS_T.add(self.m.Z_IS_LT_T[t] >= (self.m.T_IS_LT_T[t] - self.m.T_IS_W_T_WR[t+1,0]) - (1 - self.m.B_IS_HGS_T[t]) * self.T_IS_delta_max) ## Big M constraint input

            self.m.Constraint_IS_T.add(self.m.Z_IS_HT_2_T[t] <= self.T_IS_delta_max) ## Big M constraint input
            self.m.Constraint_IS_T.add(self.m.Z_IS_HT_2_T[t] >= self.T_IS_delta_min) ## Big M constraint input
            self.m.Constraint_IS_T.add(self.m.Z_IS_HT_2_T[t] <= self.T_IS_delta_max * self.m.Z_HS_IS_2_T[t]) ## Big M constraint input
            self.m.Constraint_IS_T.add(self.m.Z_IS_HT_2_T[t] >= self.T_IS_delta_min * self.m.Z_HS_IS_2_T[t]) ## Big M constraint input
            self.m.Constraint_IS_T.add(self.m.Z_IS_HT_2_T[t] <= (self.m.T_IS_HT_T[t] - self.m.T_IS_W_T_WR[t+1,0]) - (1 - self.m.Z_HS_IS_2_T[t]) * self.T_IS_delta_min) ## Big M constraint input
            self.m.Constraint_IS_T.add(self.m.Z_IS_HT_2_T[t] >= (self.m.T_IS_HT_T[t] - self.m.T_IS_W_T_WR[t+1,0]) - (1 - self.m.Z_HS_IS_2_T[t]) * self.T_IS_delta_max) ## Big M constraint input

            self.m.Constraint_IS_T.add(self.m.Z_IS_LT_2_T[t] <= self.T_IS_delta_max) ## Big M constraint input
            self.m.Constraint_IS_T.add(self.m.Z_IS_LT_2_T[t] >= self.T_IS_delta_min) ## Big M constraint input
            self.m.Constraint_IS_T.add(self.m.Z_IS_LT_2_T[t] <= self.T_IS_delta_max * self.m.Z_IS_HGS_2_T[t]) ## Big M constraint input
            self.m.Constraint_IS_T.add(self.m.Z_IS_LT_2_T[t] >= self.T_IS_delta_min * self.m.Z_IS_HGS_2_T[t]) ## Big M constraint input
            self.m.Constraint_IS_T.add(self.m.Z_IS_LT_2_T[t] <= (self.m.T_IS_LT_T[t] - self.m.T_IS_W_T_WR[t+1,0]) - (1 - self.m.Z_IS_HGS_2_T[t]) * self.T_IS_delta_min) ## Big M constraint input
            self.m.Constraint_IS_T.add(self.m.Z_IS_LT_2_T[t] >= (self.m.T_IS_LT_T[t] - self.m.T_IS_W_T_WR[t+1,0]) - (1 - self.m.Z_IS_HGS_2_T[t]) * self.T_IS_delta_max) ## Big M constraint input
        
        for t in self.T[0:-1]:
            self.m.Constraint_IS_T.add(self.m.Z_IS_pump_T[t] >= self.m.B_HS_IS_T[t])
            self.m.Constraint_IS_T.add(self.m.Z_IS_pump_T[t] >= self.m.B_IS_HGS_T[t])
            self.m.Constraint_IS_T.add(self.m.Z_IS_pump_T[t] <= self.m.B_HS_IS_T[t] + self.m.B_IS_HGS_T[t])
            self.m.Constraint_IS_T.add(self.m.Z_IS_pump_T[t] <= 1)
            self.m.Constraint_IS_T.add(self.m.E_IS_EL_T[t] == self.m.Z_IS_pump_T[t] * self.e_IS_EL)

        for t in self.T[1:-1]:
            self.m.Constraint_IS_T.add(self.m.Z_HS_HGS_T[t] >= self.m.B_HS_IS_T[t] - self.m.B_HS_IS_T[t-1])
            self.m.Constraint_IS_T.add(self.m.Z_HS_HGS_T[t] >= self.m.B_HS_IS_T[t-1] - self.m.B_HS_IS_T[t])
            self.m.Constraint_IS_T.add(self.m.Z_HS_HGS_T[t] >= self.m.B_IS_HGS_T[t] - self.m.B_IS_HGS_T[t-1])
            self.m.Constraint_IS_T.add(self.m.Z_HS_HGS_T[t] >= self.m.B_IS_HGS_T[t-1] - self.m.B_IS_HGS_T[t])
            self.m.Constraint_IS_T.add(self.m.Z_HS_HGS_T[t] <= 1)

        if self.End_Toggle_Constraints == True:
            self.m.Constraint_IS_T.add(self.m.Z_HS_HGS_T[self.T[-1]] >= self.m.B_HS_IS_T[self.T[-2]] - self.B_HS_IS_end)
            self.m.Constraint_IS_T.add(self.m.Z_HS_HGS_T[self.T[-1]] >= self.B_HS_IS_end - self.m.B_HS_IS_T[self.T[-2]])
            self.m.Constraint_IS_T.add(self.m.Z_HS_HGS_T[self.T[-1]] >= self.m.B_IS_HGS_T[self.T[-2]] - self.B_IS_HGS_end)
            self.m.Constraint_IS_T.add(self.m.Z_HS_HGS_T[self.T[-1]] >= self.B_IS_HGS_end - self.m.B_IS_HGS_T[self.T[-2]])
            self.m.Constraint_IS_T.add(self.m.Z_HS_HGS_T[self.T[-1]] <= 1)

        if self.Start_Toggle_Constraints == True:
            self.m.Constraint_IS_T.add(self.m.Z_HS_HGS_T[0] >= self.m.B_HS_IS_T[0] - self.B_HS_IS_start)
            self.m.Constraint_IS_T.add(self.m.Z_HS_HGS_T[0] >= self.B_HS_IS_start - self.m.B_HS_IS_T[0])
            self.m.Constraint_IS_T.add(self.m.Z_HS_HGS_T[0] >= self.m.B_IS_HGS_T[0] - self.B_IS_HGS_start)
            self.m.Constraint_IS_T.add(self.m.Z_HS_HGS_T[0] >= self.B_IS_HGS_start - self.m.B_IS_HGS_T[0])
            self.m.Constraint_IS_T.add(self.m.Z_HS_HGS_T[0] <= 1)

        for t in self.T[0:-1]:
            self.m.Constraint_IS_T.add(self.m.Z_HS_IS_2_T[t] >= (self.m.B_HS_IS_T[t] + (1-self.m.B_IS_HGS_T[t])) - 1)
            self.m.Constraint_IS_T.add(self.m.Z_HS_IS_2_T[t] <= self.m.B_HS_IS_T[t])
            self.m.Constraint_IS_T.add(self.m.Z_HS_IS_2_T[t] <= (1-self.m.B_IS_HGS_T[t]))

        for t in self.T[0:-1]:
            self.m.Constraint_IS_T.add(self.m.Z_IS_HGS_2_T[t] >= (self.m.B_IS_HGS_T[t] + (1-self.m.B_HS_IS_T[t])) - 1)
            self.m.Constraint_IS_T.add(self.m.Z_IS_HGS_2_T[t] <= self.m.B_IS_HGS_T[t])
            self.m.Constraint_IS_T.add(self.m.Z_IS_HGS_2_T[t] <= (1-self.m.B_HS_IS_T[t]))

        ## GS
        ## Concrete energy flows
        self.m.Constraint_GS_T = pyo.ConstraintList()
        for t in self.T[:-1]:
            for c in self.cc_GS:
                for r in self.cr_GS[1:]:
                    self.m.Constraint_GS_T.add(self.m.Q_GS_C_NORTH_T_CR_CC[t,c,r] == (self.m.T_GS_C_T_CR_CC[t+1,c,r-1] - self.m.T_GS_C_T_CR_CC[t+1,c,r]) * self.lambda_GS_c_c / self.height_GS * self.a_north_south_GS)
        
            for c in self.cc_GS:
                for r in self.cr_GS[:-1]:
                    self.m.Constraint_GS_T.add(self.m.Q_GS_C_SOUTH_T_CR_CC[t,c,r] == (self.m.T_GS_C_T_CR_CC[t+1,c,r+1] - self.m.T_GS_C_T_CR_CC[t+1,c,r]) * self.lambda_GS_c_c / self.height_GS * self.a_north_south_GS)

            for c in self.cc_GS[1:]:
                for r in self.cr_GS:
                    self.m.Constraint_GS_T.add(self.m.Q_GS_C_WEST_T_CR_CC[t,c,r] == (self.m.T_GS_C_T_CR_CC[t+1,c-1,r] - self.m.T_GS_C_T_CR_CC[t+1,c,r]) * self.lambda_GS_c_c / self.width_GS * self.a_east_west_GS)

            for c in self.cc_GS[:-1]:
                for r in self.cr_GS:
                    self.m.Constraint_GS_T.add(self.m.Q_GS_C_EAST_T_CR_CC[t,c,r] == (self.m.T_GS_C_T_CR_CC[t+1,c+1,r] - self.m.T_GS_C_T_CR_CC[t+1,c,r]) * self.lambda_GS_c_c / self.width_GS * self.a_east_west_GS)

            for c in self.wc_GS:
                for r in self.wr_GS:
                    self.m.Constraint_GS_T.add(self.m.Q_GS_C_W_T_WR_WC[t,c,r] == (self.m.T_GS_W_T_WR_WC[t+1,c,r] - self.m.T_GS_C_T_CR_CC[t+1,c,r]) * self.alpha_GS_w_c * self.a_pipe_GS)

        ## Concrete borders
        for t in self.T[:-1]:
            for c in self.cc_GS:
                self.m.Constraint_GS_T.add(self.m.Q_GS_C_NORTH_T_CR_CC[t,c,0] == (self.t_GS_air - self.m.T_GS_C_T_CR_CC[t+1,c,0]) * self.lambda_GS_c_a / self.height_GS * self.a_north_south_GS)

            for c in self.cc_GS:
                self.m.Constraint_GS_T.add(self.m.Q_GS_C_SOUTH_T_CR_CC[t,c,self.cr_GS[-1]] == (self.t_GS_soil - self.m.T_GS_C_T_CR_CC[t+1,c,self.cr_GS[-1]]) * self.lambda_GS_c_s / self.height_GS * self.a_north_south_GS)

            if len(self.cc_GS) > 1:
                for r in self.cr_GS:
                    self.m.Constraint_GS_T.add(self.m.Q_GS_C_WEST_T_CR_CC[t,0,r] == (self.m.T_GS_C_T_CR_CC[t+1,self.cc_GS[-1],r] - self.m.T_GS_C_T_CR_CC[t+1,0,r]) * self.lambda_GS_c_c / self.width_GS * self.a_east_west_GS)

                for r in self.cr_GS:
                    self.m.Constraint_GS_T.add(self.m.Q_GS_C_EAST_T_CR_CC[t,self.cc_GS[-1],r] == (self.m.T_GS_C_T_CR_CC[t+1,0,r] - self.m.T_GS_C_T_CR_CC[t+1,self.cc_GS[-1],r]) * self.lambda_GS_c_c / self.width_GS * self.a_east_west_GS)
            else:
                for r in self.cr_GS:
                    self.m.Constraint_GS_T.add(self.m.Q_GS_C_WEST_T_CR_CC[t,0,r] == 0)

                for r in self.cr_GS:
                    self.m.Constraint_GS_T.add(self.m.Q_GS_C_EAST_T_CR_CC[t,self.cc_GS[-1],r] == 0)    

        ## Concrete temperature
        self.m.Constraint_GS_T.add(self.m.T_GS_C_T_CR_CC[0,0,0] == self.T_GS_c_1_start)
        self.m.Constraint_GS_T.add(self.m.T_GS_C_T_CR_CC[0,0,1] == self.T_GS_c_2_start)
        self.m.Constraint_GS_T.add(self.m.T_GS_C_T_CR_CC[0,0,2] == self.T_GS_c_3_start)
        self.m.Constraint_GS_T.add(self.m.T_GS_C_T_CR_CC[0,0,3] == self.T_GS_c_4_start)
        self.m.Constraint_GS_T.add(self.m.T_GS_C_T_CR_CC[0,0,4] == self.T_GS_c_5_start)
        self.m.Constraint_GS_T.add(self.m.T_GS_C_T_CR_CC[0,0,5] == self.T_GS_c_6_start)
        self.m.Constraint_GS_T.add(self.m.T_GS_C_T_CR_CC[0,0,6] == self.T_GS_c_7_start)

        for t in self.T[:-1]:
            for c in self.cc_GS:
                for r in self.cr_GS:
                    if r in self.wr_GS:
                        self.m.Constraint_GS_T.add(self.m.T_GS_C_T_CR_CC[t+1,c,r] == self.m.T_GS_C_T_CR_CC[t,c,r] + self.StepSizeInSec * (1/(self.m_GS_c * self.c_c)) * (self.m.Q_GS_C_NORTH_T_CR_CC[t,c,r] + self.m.Q_GS_C_SOUTH_T_CR_CC[t,c,r] + self.m.Q_GS_C_WEST_T_CR_CC[t,c,r] + self.m.Q_GS_C_EAST_T_CR_CC[t,c,r] + self.m.Q_GS_C_W_T_WR_WC[t,c,r]))
                    else:
                        self.m.Constraint_GS_T.add(self.m.T_GS_C_T_CR_CC[t+1,c,r] == self.m.T_GS_C_T_CR_CC[t,c,r] + self.StepSizeInSec * (1/(self.m_GS_c * self.c_c)) * (self.m.Q_GS_C_NORTH_T_CR_CC[t,c,r] + self.m.Q_GS_C_SOUTH_T_CR_CC[t,c,r] + self.m.Q_GS_C_WEST_T_CR_CC[t,c,r] + self.m.Q_GS_C_EAST_T_CR_CC[t,c,r]))
        
        for t in self.T[1:]:
            for c in self.cc_GS:
                for r in self.cr_GS:
                    self.m.Constraint_GS_T.add(self.m.T_GS_C_T_CR_CC[t,c,r] <= self.T_GS_max_c + self.m.S_T_GS_C_T_CR_CC[t,c,r])
                    self.m.Constraint_GS_T.add(self.m.T_GS_C_T_CR_CC[t,c,r] >= self.T_GS_min_c - self.m.S_T_GS_C_T_CR_CC[t,c,r])

        ## Water energy flows
        for t in self.T[:-1]:
            for r in self.wr_GS[::2]:
                for c in self.wc_GS[1:]:
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_WEST_T_WR_WC[t,c,r] <= self.T_GS_delta_max * self.mdot_GS_w/self.n_GS_blocks * self.c_w) ## Big M constraint input
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_WEST_T_WR_WC[t,c,r] >= self.T_GS_delta_min * self.mdot_GS_w/self.n_GS_blocks * self.c_w) ## Big M constraint input
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_WEST_T_WR_WC[t,c,r] <= self.T_GS_delta_max * self.mdot_GS_w/self.n_GS_blocks * self.c_w * (self.m.B_GS_HGS_T[t] + self.m.B_GS_CS_T[t] + self.m.B_GS_HGS_CS_T[t])) ## Big M constraint input
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_WEST_T_WR_WC[t,c,r] >= self.T_GS_delta_min * self.mdot_GS_w/self.n_GS_blocks * self.c_w * (self.m.B_GS_HGS_T[t] + self.m.B_GS_CS_T[t] + self.m.B_GS_HGS_CS_T[t])) ## Big M constraint input
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_WEST_T_WR_WC[t,c,r] <= (self.m.T_GS_W_T_WR_WC[t+1,c-1,r] - self.m.T_GS_W_T_WR_WC[t+1,c,r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w - (1 - (self.m.B_GS_HGS_T[t] + self.m.B_GS_CS_T[t] + self.m.B_GS_HGS_CS_T[t])) * self.T_GS_delta_min * self.mdot_GS_w/self.n_GS_blocks * self.c_w) ## Big M constraint input
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_WEST_T_WR_WC[t,c,r] >= (self.m.T_GS_W_T_WR_WC[t+1,c-1,r] - self.m.T_GS_W_T_WR_WC[t+1,c,r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w - (1 - (self.m.B_GS_HGS_T[t] + self.m.B_GS_CS_T[t] + self.m.B_GS_HGS_CS_T[t])) * self.T_GS_delta_max * self.mdot_GS_w/self.n_GS_blocks * self.c_w) ## Big M constraint input
                for c in self.wc_GS[:]:    
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_EAST_T_WR_WC[t,c,r] == 0)

            for r in self.wr_GS[1::2]:
                for c in self.wc_GS[:-1]:
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_EAST_T_WR_WC[t,c,r] <= self.T_GS_delta_max * self.mdot_GS_w/self.n_GS_blocks * self.c_w) ## Big M constraint input
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_EAST_T_WR_WC[t,c,r] >= self.T_GS_delta_min * self.mdot_GS_w/self.n_GS_blocks * self.c_w) ## Big M constraint input
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_EAST_T_WR_WC[t,c,r] <= self.T_GS_delta_max * self.mdot_GS_w/self.n_GS_blocks * self.c_w * (self.m.B_GS_HGS_T[t] + self.m.B_GS_CS_T[t] + self.m.B_GS_HGS_CS_T[t])) ## Big M constraint input
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_EAST_T_WR_WC[t,c,r] >= self.T_GS_delta_min * self.mdot_GS_w/self.n_GS_blocks * self.c_w * (self.m.B_GS_HGS_T[t] + self.m.B_GS_CS_T[t] + self.m.B_GS_HGS_CS_T[t])) ## Big M constraint input
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_EAST_T_WR_WC[t,c,r] <= (self.m.T_GS_W_T_WR_WC[t+1,c+1,r] - self.m.T_GS_W_T_WR_WC[t+1,c,r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w - (1 - (self.m.B_GS_HGS_T[t] + self.m.B_GS_CS_T[t] + self.m.B_GS_HGS_CS_T[t])) * self.T_GS_delta_min * self.mdot_GS_w/self.n_GS_blocks * self.c_w) ## Big M constraint input
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_EAST_T_WR_WC[t,c,r] >= (self.m.T_GS_W_T_WR_WC[t+1,c+1,r] - self.m.T_GS_W_T_WR_WC[t+1,c,r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w - (1 - (self.m.B_GS_HGS_T[t] + self.m.B_GS_CS_T[t] + self.m.B_GS_HGS_CS_T[t])) * self.T_GS_delta_max * self.mdot_GS_w/self.n_GS_blocks * self.c_w) ## Big M constraint input
                for c in self.wc_GS[:]:    
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_WEST_T_WR_WC[t,c,r] == 0)

            for c in self.wc_GS:
                for r in self.wr_GS:
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_C_T_WR_WC[t,c,r] == (self.m.T_GS_C_T_CR_CC[t+1,c,r] - self.m.T_GS_W_T_WR_WC[t+1,c,r]) * self.alpha_GS_w_c * self.a_pipe_GS)      
        
        ## Water borders
        for t in self.T[:-1]:
            for r in self.wr_GS[0::2]:
                if r > 1:
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_WEST_T_WR_WC[t,0,r] <= self.T_GS_delta_max * self.mdot_GS_w/self.n_GS_blocks * self.c_w) ## Big M constraint input
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_WEST_T_WR_WC[t,0,r] >= self.T_GS_delta_min * self.mdot_GS_w/self.n_GS_blocks * self.c_w) ## Big M constraint input
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_WEST_T_WR_WC[t,0,r] <= self.T_GS_delta_max * self.mdot_GS_w/self.n_GS_blocks * self.c_w * (self.m.B_GS_HGS_T[t] + self.m.B_GS_CS_T[t] + self.m.B_GS_HGS_CS_T[t])) ## Big M constraint input
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_WEST_T_WR_WC[t,0,r] >= self.T_GS_delta_min * self.mdot_GS_w/self.n_GS_blocks * self.c_w * (self.m.B_GS_HGS_T[t] + self.m.B_GS_CS_T[t] + self.m.B_GS_HGS_CS_T[t])) ## Big M constraint input
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_WEST_T_WR_WC[t,0,r] <= (self.m.T_GS_W_T_WR_WC[t+1,0,r-2] - self.m.T_GS_W_T_WR_WC[t+1,0,r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w - (1 - (self.m.B_GS_HGS_T[t] + self.m.B_GS_CS_T[t] + self.m.B_GS_HGS_CS_T[t])) * self.T_GS_delta_min * self.mdot_GS_w/self.n_GS_blocks * self.c_w) ## Big M constraint input
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_WEST_T_WR_WC[t,0,r] >= (self.m.T_GS_W_T_WR_WC[t+1,0,r-2] - self.m.T_GS_W_T_WR_WC[t+1,0,r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w - (1 - (self.m.B_GS_HGS_T[t] + self.m.B_GS_CS_T[t] + self.m.B_GS_HGS_CS_T[t])) * self.T_GS_delta_max * self.mdot_GS_w/self.n_GS_blocks * self.c_w) ## Big M constraint input
                else: ## Start inflow
                    self.m.Constraint_GS_T.add(self.m.Q_GS_W_WEST_T_WR_WC[t,0,1] == (self.c_w * self.mdot_GS_w/self.n_GS_blocks * self.m.Z_T_GS_HGS_T[t] + self.c_w * self.mdot_GS_w/self.n_GS_blocks * self.m.Z_T_GS_CS_T[t] + self.c_w * self.mdot_GS_w_2/self.n_GS_blocks * self.m.Z_T_GS_HGS_2_T[t] + self.c_w * self.mdot_GS_w_2/self.n_GS_blocks * self.m.Z_T_GS_CS_2_T[t]))
            for r in self.wr_GS[1::2]:
                self.m.Constraint_GS_T.add(self.m.Q_GS_W_EAST_T_WR_WC[t,c,r] <= self.T_GS_delta_max * self.mdot_GS_w/self.n_GS_blocks * self.c_w) ## Big M constraint input
                self.m.Constraint_GS_T.add(self.m.Q_GS_W_EAST_T_WR_WC[t,c,r] >= self.T_GS_delta_min * self.mdot_GS_w/self.n_GS_blocks * self.c_w) ## Big M constraint input
                self.m.Constraint_GS_T.add(self.m.Q_GS_W_EAST_T_WR_WC[t,c,r] <= self.T_GS_delta_max * self.mdot_GS_w/self.n_GS_blocks * self.c_w * (self.m.B_GS_HGS_T[t] + self.m.B_GS_CS_T[t] + self.m.B_GS_HGS_CS_T[t])) ## Big M constraint input
                self.m.Constraint_GS_T.add(self.m.Q_GS_W_EAST_T_WR_WC[t,c,r] >= self.T_GS_delta_min * self.mdot_GS_w/self.n_GS_blocks * self.c_w * (self.m.B_GS_HGS_T[t] + self.m.B_GS_CS_T[t] + self.m.B_GS_HGS_CS_T[t])) ## Big M constraint input
                self.m.Constraint_GS_T.add(self.m.Q_GS_W_EAST_T_WR_WC[t,c,r] <= (self.m.T_GS_W_T_WR_WC[t+1,self.wc_GS[-1],r-2] - self.m.T_GS_W_T_WR_WC[t+1,self.wc_GS[-1],r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w - (1 - (self.m.B_GS_HGS_T[t] + self.m.B_GS_CS_T[t] + self.m.B_GS_HGS_CS_T[t])) * self.T_GS_delta_min * self.mdot_GS_w/self.n_GS_blocks * self.c_w) ## Big M constraint input
                self.m.Constraint_GS_T.add(self.m.Q_GS_W_EAST_T_WR_WC[t,c,r] >= (self.m.T_GS_W_T_WR_WC[t+1,self.wc_GS[-1],r-2] - self.m.T_GS_W_T_WR_WC[t+1,self.wc_GS[-1],r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w - (1 - (self.m.B_GS_HGS_T[t] + self.m.B_GS_CS_T[t] + self.m.B_GS_HGS_CS_T[t])) * self.T_GS_delta_max * self.mdot_GS_w/self.n_GS_blocks * self.c_w) ## Big M constraint input

        ## Water temperature
        self.m.Constraint_GS_T.add(self.m.T_GS_W_T_WR_WC[0,0,1] == self.T_GS_w_1_start)
        self.m.Constraint_GS_T.add(self.m.T_GS_W_T_WR_WC[0,0,3] == self.T_GS_w_2_start)
        self.m.Constraint_GS_T.add(self.m.T_GS_W_T_WR_WC[0,0,5] == self.T_GS_w_3_start)

        for t in self.T[:-1]:
            for c in self.wc_GS:
                for r in self.wr_GS:
                    self.m.Constraint_GS_T.add(self.m.T_GS_W_T_WR_WC[t+1,c,r] == self.m.T_GS_W_T_WR_WC[t,c,r] + self.StepSizeInSec * (1/(self.m_GS_w * self.c_w)) * (self.m.Q_GS_W_WEST_T_WR_WC[t,c,r] + self.m.Q_GS_W_EAST_T_WR_WC[t,c,r] + self.m.Q_GS_W_C_T_WR_WC[t,c,r]))

        for t in self.T[1:]:
            for c in self.wc_GS:
                for r in self.wr_GS:
                    self.m.Constraint_GS_T.add(self.m.T_GS_W_T_WR_WC[t,c,r] <= self.T_GS_max_w + self.m.S_T_GS_W_T_WR_WC[t,c,r])
                    self.m.Constraint_GS_T.add(self.m.T_GS_W_T_WR_WC[t,c,r] >= self.T_GS_min_w - self.m.S_T_GS_W_T_WR_WC[t,c,r])

        ## Water outflow
        for t in self.T:
            if self.wr_GS[::2][-1] > self.wr_GS[1::2][-1]:
                self.m.Constraint_GS_T.add(self.m.T_GS_T[t] == self.m.T_GS_W_T_WR_WC[t,self.wc_GS[-1],self.wr_GS[-1]])
            else:
                self.m.Constraint_GS_T.add(self.m.T_GS_T[t] == self.m.T_GS_W_T_WR_WC[t,self.wc_GS[0],self.wr_GS[-1]])

        ## Old
        for t in self.T[0:-1]:
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_HGS_T[t] <= self.T_GS_delta_max) ## Big M constraint input
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_HGS_T[t] >= self.T_GS_delta_min) ## Big M constraint input
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_HGS_T[t] <= self.T_GS_delta_max * self.m.B_GS_HGS_T[t]) ## Big M constraint input
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_HGS_T[t] >= self.T_GS_delta_min * self.m.B_GS_HGS_T[t]) ## Big M constraint input
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_HGS_T[t] <= (self.m.T_GS_HGS_T[t] - self.m.T_GS_W_T_WR_WC[t+1,0,1]) - (1 - self.m.B_GS_HGS_T[t]) * self.T_GS_delta_min) ## Big M constraint input
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_HGS_T[t] >= (self.m.T_GS_HGS_T[t] - self.m.T_GS_W_T_WR_WC[t+1,0,1]) - (1 - self.m.B_GS_HGS_T[t]) * self.T_GS_delta_max) ## Big M constraint input

        for t in self.T[0:-1]:
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_CS_T[t] <= self.T_GS_delta_max) ## Big M constraint input
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_CS_T[t] >= self.T_GS_delta_min) ## Big M constraint input
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_CS_T[t] <= self.T_GS_delta_max * self.m.B_GS_CS_T[t]) ## Big M constraint input
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_CS_T[t] >= self.T_GS_delta_min * self.m.B_GS_CS_T[t]) ## Big M constraint input
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_CS_T[t] <= (self.m.T_GS_CS_T[t] - self.m.T_GS_W_T_WR_WC[t+1,0,1]) - (1 - self.m.B_GS_CS_T[t]) * self.T_GS_delta_min) ## Big M constraint input
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_CS_T[t] >= (self.m.T_GS_CS_T[t] - self.m.T_GS_W_T_WR_WC[t+1,0,1]) - (1 - self.m.B_GS_CS_T[t]) * self.T_GS_delta_max) ## Big M constraint input

        for t in self.T[0:-1]:
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_HGS_2_T[t] <= self.T_GS_delta_max) ## Big M constraint input
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_HGS_2_T[t] >= self.T_GS_delta_min) ## Big M constraint input
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_HGS_2_T[t] <= self.T_GS_delta_max * self.m.B_GS_HGS_CS_T[t]) ## Big M constraint input
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_HGS_2_T[t] >= self.T_GS_delta_min * self.m.B_GS_HGS_CS_T[t]) ## Big M constraint input
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_HGS_2_T[t] <= (self.m.T_GS_HGS_T[t] - self.m.T_GS_W_T_WR_WC[t+1,0,1]) - (1 - self.m.B_GS_HGS_CS_T[t]) * self.T_GS_delta_min) ## Big M constraint input
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_HGS_2_T[t] >= (self.m.T_GS_HGS_T[t] - self.m.T_GS_W_T_WR_WC[t+1,0,1]) - (1 - self.m.B_GS_HGS_CS_T[t]) * self.T_GS_delta_max) ## Big M constraint input

        for t in self.T[0:-1]:
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_CS_2_T[t] <= self.T_GS_delta_max) ## Big M constraint input
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_CS_2_T[t] >= self.T_GS_delta_min) ## Big M constraint input
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_CS_2_T[t] <= self.T_GS_delta_max * self.m.B_GS_HGS_CS_T[t]) ## Big M constraint input
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_CS_2_T[t] >= self.T_GS_delta_min * self.m.B_GS_HGS_CS_T[t]) ## Big M constraint input
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_CS_2_T[t] <= (self.m.T_GS_CS_T[t] - self.m.T_GS_W_T_WR_WC[t+1,0,1]) - (1 - self.m.B_GS_HGS_CS_T[t]) * self.T_GS_delta_min) ## Big M constraint input
            self.m.Constraint_GS_T.add(self.m.Z_T_GS_CS_2_T[t] >= (self.m.T_GS_CS_T[t] - self.m.T_GS_W_T_WR_WC[t+1,0,1]) - (1 - self.m.B_GS_HGS_CS_T[t]) * self.T_GS_delta_max) ## Big M constraint input
    
        for t in self.T[0:-1]:
            self.m.Constraint_GS_T.add(self.m.E_GS_EL_T[t] == (self.m.B_GS_HGS_T[t] + self.m.B_GS_CS_T[t] + self.m.B_GS_HGS_CS_T[t]) * self.e_GS_EL)

        ## V_HP_HXH_HS
        self.m.Constraint_V_HP_HXH_HS_T = pyo.ConstraintList()
        for t in self.T[0:-1]:
            for h in self.H:
                self.m.Constraint_V_HP_HXH_HS_T.add(self.m.Z_HP_HXH_H_T[h,t] >= self.m.B_HP_H_T[h,t] + self.m.B_HXH_HS_T[t] - (2-1))
                self.m.Constraint_V_HP_HXH_HS_T.add(self.m.Z_HP_HXH_H_T[h,t] <= self.m.B_HP_H_T[h,t])
                self.m.Constraint_V_HP_HXH_HS_T.add(self.m.Z_HP_HXH_H_T[h,t] <= self.m.B_HXH_HS_T[t])

                self.m.Constraint_V_HP_HXH_HS_T.add(self.m.Z_HP_HS_H_T[h,t] >= self.m.B_HP_H_T[h,t] + (1-self.m.B_HXH_HS_T[t]) - (2-1))
                self.m.Constraint_V_HP_HXH_HS_T.add(self.m.Z_HP_HS_H_T[h,t] <= self.m.B_HP_H_T[h,t])
                self.m.Constraint_V_HP_HXH_HS_T.add(self.m.Z_HP_HS_H_T[h,t] <= 1-self.m.B_HXH_HS_T[t])

        for t in self.T[0:-1]:
            self.m.Constraint_V_HP_HXH_HS_T.add(sum(self.m.Z_HP_HXH_H_T[h,t] + self.m.Z_HP_HS_H_T[h,t] for h in self.H) <= 1) ## Thighten Relaxation problem

        if self.TControlPeriodSwitch1 > 0:
            for t in self.T[0:self.TControlPeriodSwitch1-(self.ControlPeriod1)+1]:
                if t%self.ControlPeriod1 == 0:
                    for i in range(1,self.ControlPeriod1):
                        self.m.Constraint_V_HP_HXH_HS_T.add(self.m.B_HXH_HS_T[t] == self.m.B_HXH_HS_T[t+i])

        for t in self.T[self.TControlPeriodSwitch1:-(self.ControlPeriod2)]:
            if t%self.ControlPeriod2 == 0:
                for i in range(1,self.ControlPeriod2):
                    self.m.Constraint_V_HP_HXH_HS_T.add(self.m.B_HXH_HS_T[t] == self.m.B_HXH_HS_T[t+i])

        for t in self.T[1:-1]:
            self.m.Constraint_V_HP_HXH_HS_T.add(self.m.Z_HXH_HS_T[t] >= self.m.B_HXH_HS_T[t] - self.m.B_HXH_HS_T[t-1])
            self.m.Constraint_V_HP_HXH_HS_T.add(self.m.Z_HXH_HS_T[t] >= self.m.B_HXH_HS_T[t-1] - self.m.B_HXH_HS_T[t])
            self.m.Constraint_V_HP_HXH_HS_T.add(self.m.Z_HXH_HS_T[t] <= 1)
        
        if self.End_Toggle_Constraints == True:
            self.m.Constraint_V_HP_HXH_HS_T.add(self.m.Z_HXH_HS_T[self.T[-1]] >= self.m.B_HXH_HS_T[self.T[-2]] - self.B_HXH_HS_end)
            self.m.Constraint_V_HP_HXH_HS_T.add(self.m.Z_HXH_HS_T[self.T[-1]] >= self.B_HXH_HS_end - self.m.B_HXH_HS_T[self.T[-2]])
            self.m.Constraint_V_HP_HXH_HS_T.add(self.m.Z_HXH_HS_T[self.T[-1]] <= 1)

        if self.Start_Toggle_Constraints == True:
            self.m.Constraint_V_HP_HXH_HS_T.add(self.m.Z_HXH_HS_T[0] >= self.m.B_HXH_HS_T[0] - self.B_HXH_HS_start)
            self.m.Constraint_V_HP_HXH_HS_T.add(self.m.Z_HXH_HS_T[0] >= self.B_HXH_HS_start - self.m.B_HXH_HS_T[0])
            self.m.Constraint_V_HP_HXH_HS_T.add(self.m.Z_HXH_HS_T[0] <= 1)

        ## V_HXA_HXH_HGC
        self.m.Constraint_V_HXA_HXH_HGC_T = pyo.ConstraintList()
        for t in self.T[0:-1]:
            self.m.Constraint_V_HXA_HXH_HGC_T.add(self.m.Z_HXA_HXH_T[t] >= self.m.Z_HXA_T[t] + self.m.B_HXH_HGC_T[t] - (2-1))
            self.m.Constraint_V_HXA_HXH_HGC_T.add(self.m.Z_HXA_HXH_T[t] <= self.m.Z_HXA_T[t])
            self.m.Constraint_V_HXA_HXH_HGC_T.add(self.m.Z_HXA_HXH_T[t] <= self.m.B_HXH_HGC_T[t])

            self.m.Constraint_V_HXA_HXH_HGC_T.add(self.m.Z_HXA_HGC_T[t] >= self.m.Z_HXA_T[t] + (1-self.m.B_HXH_HGC_T[t]) - (2-1))
            self.m.Constraint_V_HXA_HXH_HGC_T.add(self.m.Z_HXA_HGC_T[t] <= self.m.Z_HXA_T[t])
            self.m.Constraint_V_HXA_HXH_HGC_T.add(self.m.Z_HXA_HGC_T[t] <= 1-self.m.B_HXH_HGC_T[t])

        for t in self.T[0:-1]:
            self.m.Constraint_V_HXA_HXH_HGC_T.add(self.m.Z_HXA_HXH_T[t] + self.m.Z_HXA_HGC_T[t] <= 1) ## Thighten Relaxation problem

        if self.TControlPeriodSwitch1 > 0:
            for t in self.T[0:self.TControlPeriodSwitch1-(self.ControlPeriod1)+1]:
                if t%self.ControlPeriod1 == 0:
                    for i in range(1,self.ControlPeriod1):
                        self.m.Constraint_V_HXA_HXH_HGC_T.add(self.m.B_HXH_HGC_T[t] == self.m.B_HXH_HGC_T[t+i])

        for t in self.T[self.TControlPeriodSwitch1:-(self.ControlPeriod2)]:
            if t%self.ControlPeriod2 == 0:
                for i in range(1,self.ControlPeriod2):
                    self.m.Constraint_V_HXA_HXH_HGC_T.add(self.m.B_HXH_HGC_T[t] == self.m.B_HXH_HGC_T[t+i])
        
        for t in self.T[1:-1]:
            self.m.Constraint_V_HXA_HXH_HGC_T.add(self.m.Z_HXH_HGC_T[t] >= self.m.B_HXH_HGC_T[t] - self.m.B_HXH_HGC_T[t-1])
            self.m.Constraint_V_HXA_HXH_HGC_T.add(self.m.Z_HXH_HGC_T[t] >= self.m.B_HXH_HGC_T[t-1] - self.m.B_HXH_HGC_T[t])
            self.m.Constraint_V_HXA_HXH_HGC_T.add(self.m.Z_HXH_HGC_T[t] <= 1)
        
        if self.End_Toggle_Constraints == True:
            self.m.Constraint_V_HXA_HXH_HGC_T.add(self.m.Z_HXH_HGC_T[self.T[-1]] >= self.m.B_HXH_HGC_T[self.T[-2]] - self.B_HXH_HGC_end)
            self.m.Constraint_V_HXA_HXH_HGC_T.add(self.m.Z_HXH_HGC_T[self.T[-1]] >= self.B_HXH_HGC_end - self.m.B_HXH_HGC_T[self.T[-2]])
            self.m.Constraint_V_HXA_HXH_HGC_T.add(self.m.Z_HXH_HGC_T[self.T[-1]] <= 1)

        if self.Start_Toggle_Constraints == True:
            self.m.Constraint_V_HXA_HXH_HGC_T.add(self.m.Z_HXH_HGC_T[0] >= self.m.B_HXH_HGC_T[0] - self.B_HXH_HGC_start)
            self.m.Constraint_V_HXA_HXH_HGC_T.add(self.m.Z_HXH_HGC_T[0] >= self.B_HXH_HGC_start - self.m.B_HXH_HGC_T[0])
            self.m.Constraint_V_HXA_HXH_HGC_T.add(self.m.Z_HXH_HGC_T[0] <= 1)

        ## VP_HS_IS 
        if self.TControlPeriodSwitch1 > 0:
            for t in self.T[0:self.TControlPeriodSwitch1-(self.ControlPeriod1)+1]:
                if t%self.ControlPeriod1 == 0:
                    for i in range(1,self.ControlPeriod1):
                        self.m.Constraint_IS_T.add(self.m.B_HS_IS_T[t] == self.m.B_HS_IS_T[t+i])
        
        for t in self.T[self.TControlPeriodSwitch1:-(self.ControlPeriod2)]:
            if t%self.ControlPeriod2 == 0:
                for i in range(1,self.ControlPeriod2):
                    self.m.Constraint_IS_T.add(self.m.B_HS_IS_T[t] == self.m.B_HS_IS_T[t+i])

        ## VP_IS_HGS
        if self.TControlPeriodSwitch1 > 0:
            for t in self.T[0:self.TControlPeriodSwitch1-(self.ControlPeriod1)+1]:
                if t%self.ControlPeriod1 == 0:
                    for i in range(1,self.ControlPeriod1):
                        self.m.Constraint_IS_T.add(self.m.B_IS_HGS_T[t] == self.m.B_IS_HGS_T[t+i])

        for t in self.T[self.TControlPeriodSwitch1:-(self.ControlPeriod2)]:
            if t%self.ControlPeriod2 == 0:
                for i in range(1,self.ControlPeriod2):
                    self.m.Constraint_IS_T.add(self.m.B_IS_HGS_T[t] == self.m.B_IS_HGS_T[t+i])

        ## V_HP_HGC_HGCHXC
        self.m.Constraint_V_HP_HGC_HGCHXC_T = pyo.ConstraintList()
        for t in self.T[0:-1]:
            for h in self.H:
                self.m.Constraint_V_HP_HGC_HGCHXC_T.add(self.m.Z_HP_HGC_H_T[h,t] >= self.m.B_HP_H_T[h,t] + self.m.B_HGC_HGCHXC_T[t] - (2-1))
                self.m.Constraint_V_HP_HGC_HGCHXC_T.add(self.m.Z_HP_HGC_H_T[h,t] <= self.m.B_HP_H_T[h,t])
                self.m.Constraint_V_HP_HGC_HGCHXC_T.add(self.m.Z_HP_HGC_H_T[h,t] <= self.m.B_HGC_HGCHXC_T[t])

                self.m.Constraint_V_HP_HGC_HGCHXC_T.add(self.m.Z_HP_HGCHXC_H_T[h,t] >= self.m.B_HP_H_T[h,t] + (1-self.m.B_HGC_HGCHXC_T[t]) - (2-1))
                self.m.Constraint_V_HP_HGC_HGCHXC_T.add(self.m.Z_HP_HGCHXC_H_T[h,t] <= self.m.B_HP_H_T[h,t])
                self.m.Constraint_V_HP_HGC_HGCHXC_T.add(self.m.Z_HP_HGCHXC_H_T[h,t] <= 1-self.m.B_HGC_HGCHXC_T[t])

        if self.TControlPeriodSwitch1 > 0:
            for t in self.T[0:self.TControlPeriodSwitch1-(self.ControlPeriod1)+1]:
                if t%self.ControlPeriod1 == 0:
                    for i in range(1,self.ControlPeriod1):
                        self.m.Constraint_V_HP_HGC_HGCHXC_T.add(self.m.B_HGC_HGCHXC_T[t] == self.m.B_HGC_HGCHXC_T[t+i])

        for t in self.T[self.TControlPeriodSwitch1:-(self.ControlPeriod2)]:
            if t%self.ControlPeriod2 == 0:
                for i in range(1,self.ControlPeriod2):
                    self.m.Constraint_V_HP_HGC_HGCHXC_T.add(self.m.B_HGC_HGCHXC_T[t] == self.m.B_HGC_HGCHXC_T[t+i])

        for t in self.T[1:-1]:
            self.m.Constraint_V_HP_HGC_HGCHXC_T.add(self.m.Z_HGC_HGCHXC_T[t] >= self.m.B_HGC_HGCHXC_T[t] - self.m.B_HGC_HGCHXC_T[t-1])
            self.m.Constraint_V_HP_HGC_HGCHXC_T.add(self.m.Z_HGC_HGCHXC_T[t] >= self.m.B_HGC_HGCHXC_T[t-1] - self.m.B_HGC_HGCHXC_T[t])
            self.m.Constraint_V_HP_HGC_HGCHXC_T.add(self.m.Z_HGC_HGCHXC_T[t] <= 1)

        if self.End_Toggle_Constraints == True:
            self.m.Constraint_V_HP_HGC_HGCHXC_T.add(self.m.Z_HGC_HGCHXC_T[self.T[-1]] >= self.m.B_HGC_HGCHXC_T[self.T[-2]] - self.B_HGC_HGCHXC_end)
            self.m.Constraint_V_HP_HGC_HGCHXC_T.add(self.m.Z_HGC_HGCHXC_T[self.T[-1]] >= self.B_HGC_HGCHXC_end - self.m.B_HGC_HGCHXC_T[self.T[-2]])
            self.m.Constraint_V_HP_HGC_HGCHXC_T.add(self.m.Z_HGC_HGCHXC_T[self.T[-1]] <= 1)

        if self.Start_Toggle_Constraints == True:
            self.m.Constraint_V_HP_HGC_HGCHXC_T.add(self.m.Z_HGC_HGCHXC_T[0] >= self.m.B_HGC_HGCHXC_T[0] - self.B_HGC_HGCHXC_start)
            self.m.Constraint_V_HP_HGC_HGCHXC_T.add(self.m.Z_HGC_HGCHXC_T[0] >= self.B_HGC_HGCHXC_start - self.m.B_HGC_HGCHXC_T[0])
            self.m.Constraint_V_HP_HGC_HGCHXC_T.add(self.m.Z_HGC_HGCHXC_T[0] <= 1)

        ## V_GS_HGS_CS
        self.m.Constraint_V_GS_HGS_CS_T = pyo.ConstraintList()
        for t in self.T[0:-1]:
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.B_GS_HGS_T[t] + self.m.B_GS_CS_T[t] + self.m.B_GS_HGS_CS_T[t] <= 1) ## Only one can be active
        
        if self.TControlPeriodSwitch1 > 0:
            for t in self.T[0:self.TControlPeriodSwitch1-(self.ControlPeriod1)+1]:
                if t%self.ControlPeriod1 == 0:
                    for i in range(1,self.ControlPeriod1):
                        self.m.Constraint_V_GS_HGS_CS_T.add(self.m.B_GS_HGS_T[t] == self.m.B_GS_HGS_T[t+i])
                        self.m.Constraint_V_GS_HGS_CS_T.add(self.m.B_GS_CS_T[t] == self.m.B_GS_CS_T[t+i])
                        self.m.Constraint_V_GS_HGS_CS_T.add(self.m.B_GS_HGS_CS_T[t] == self.m.B_GS_HGS_CS_T[t+i])

        for t in self.T[self.TControlPeriodSwitch1:-(self.ControlPeriod2)]:
            if t%self.ControlPeriod2 == 0:
                for i in range(1,self.ControlPeriod2):
                    self.m.Constraint_V_GS_HGS_CS_T.add(self.m.B_GS_HGS_T[t] == self.m.B_GS_HGS_T[t+i])
                    self.m.Constraint_V_GS_HGS_CS_T.add(self.m.B_GS_CS_T[t] == self.m.B_GS_CS_T[t+i])
                    self.m.Constraint_V_GS_HGS_CS_T.add(self.m.B_GS_HGS_CS_T[t] == self.m.B_GS_HGS_CS_T[t+i])

        for t in self.T[1:-1]:
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.Z_HGS_CS_T[t] >= self.m.B_GS_HGS_T[t] - self.m.B_GS_HGS_T[t-1])
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.Z_HGS_CS_T[t] >= self.m.B_GS_HGS_T[t-1] - self.m.B_GS_HGS_T[t])
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.Z_HGS_CS_T[t] >= self.m.B_GS_CS_T[t] - self.m.B_GS_CS_T[t-1])
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.Z_HGS_CS_T[t] >= self.m.B_GS_CS_T[t-1] - self.m.B_GS_CS_T[t])
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.Z_HGS_CS_T[t] >= self.m.B_GS_HGS_CS_T[t] - self.m.B_GS_HGS_CS_T[t-1])
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.Z_HGS_CS_T[t] >= self.m.B_GS_HGS_CS_T[t-1] - self.m.B_GS_HGS_CS_T[t])
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.Z_HGS_CS_T[t] <= 1)
        
        if self.End_Toggle_Constraints == True:
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.Z_HGS_CS_T[self.T[-1]] >= self.m.B_GS_HGS_T[self.T[-2]] - self.B_GS_HGS_end)
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.Z_HGS_CS_T[self.T[-1]] >= self.B_GS_HGS_end - self.m.B_GS_HGS_T[self.T[-2]])
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.Z_HGS_CS_T[self.T[-1]] >= self.m.B_GS_CS_T[self.T[-2]] - self.B_GS_CS_end)
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.Z_HGS_CS_T[self.T[-1]] >= self.B_GS_CS_end - self.m.B_GS_CS_T[self.T[-2]])
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.Z_HGS_CS_T[self.T[-1]] >= self.m.B_GS_HGS_CS_T[self.T[-2]] - self.B_GS_HGS_CS_end)
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.Z_HGS_CS_T[self.T[-1]] >= self.B_GS_HGS_CS_end - self.m.B_GS_HGS_CS_T[self.T[-2]])
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.Z_HGS_CS_T[self.T[-1]] <= 1)

        if self.Start_Toggle_Constraints == True:
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.Z_HGS_CS_T[0] >= self.m.B_GS_HGS_T[0] - self.B_GS_HGS_start)
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.Z_HGS_CS_T[0] >= self.B_GS_HGS_start - self.m.B_GS_HGS_T[0])
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.Z_HGS_CS_T[0] >= self.m.B_GS_CS_T[0] - self.B_GS_CS_start)
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.Z_HGS_CS_T[0] >= self.B_GS_CS_start - self.m.B_GS_CS_T[0])
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.Z_HGS_CS_T[0] >= self.m.B_GS_HGS_CS_T[0] - self.B_GS_HGS_CS_start)
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.Z_HGS_CS_T[0] >= self.B_GS_HGS_CS_start - self.m.B_GS_HGS_CS_T[0])
            self.m.Constraint_V_GS_HGS_CS_T.add(self.m.Z_HGS_CS_T[0] <= 1)

        ## HXA
        self.m.Constraint_HXA_T = pyo.ConstraintList()
        for t in self.T[0:-1]:
            self.m.Constraint_HXA_T.add(self.m.T_HXA_in_T[t] == self.temp_amb_T[t]) ## General temperature connection
            self.m.Constraint_HXA_T.add(self.m.Z_HXA_T[t] == self.m.B_HXA_T[t] * (1-self.temp_frost_T[t]))
        
        if self.TControlPeriodSwitch1 > 0:
            for t in self.T[0:self.TControlPeriodSwitch1-(self.ControlPeriod1)+1]:
                if t%self.ControlPeriod1 == 0:
                    for i in range(1,self.ControlPeriod1):
                        self.m.Constraint_HXA_T.add(self.m.B_HXA_T[t] == self.m.B_HXA_T[t+i])

        for t in self.T[self.TControlPeriodSwitch1:-(self.ControlPeriod2)]:
            if t%self.ControlPeriod2 == 0:
                for i in range(1,self.ControlPeriod2):
                    self.m.Constraint_HXA_T.add(self.m.B_HXA_T[t] == self.m.B_HXA_T[t+i])

        for t in self.T[0:-1]:
           self.m.Constraint_HXA_T.add(self.m.E_HXA_EL_T[t] == self.m.Z_HXA_T[t] * (self.e_HXA_EL_pump + self.e_HXA_EL_device))

        for t in self.T[1:-1]:
            self.m.Constraint_HXA_T.add(self.m.Z_HXA_s_T[t] >= self.m.Z_HXA_T[t] - self.m.Z_HXA_T[t-1])
            self.m.Constraint_HXA_T.add(self.m.Z_HXA_s_T[t] >= self.m.Z_HXA_T[t-1] - self.m.Z_HXA_T[t])
            self.m.Constraint_HXA_T.add(self.m.Z_HXA_s_T[t] <= 1)
        
        if self.End_Toggle_Constraints == True:
            self.m.Constraint_HXA_T.add(self.m.Z_HXA_s_T[self.T[-1]] >= self.m.Z_HXA_T[self.T[-2]] - self.B_HXA_end)
            self.m.Constraint_HXA_T.add(self.m.Z_HXA_s_T[self.T[-1]] >= self.B_HXA_end - self.m.Z_HXA_T[self.T[-2]])
            self.m.Constraint_HXA_T.add(self.m.Z_HXA_s_T[self.T[-1]] <= 1)

        if self.Start_Toggle_Constraints == True:
            self.m.Constraint_HXA_T.add(self.m.Z_HXA_s_T[0] >= self.m.Z_HXA_T[0] - self.B_HXA_start)
            self.m.Constraint_HXA_T.add(self.m.Z_HXA_s_T[0] >= self.B_HXA_start - self.m.Z_HXA_T[0])
            self.m.Constraint_HXA_T.add(self.m.Z_HXA_s_T[0] <= 1)

        # HXA tank
        self.m.Constraint_HXA_T.add(self.m.T_HXA_T[0] == self.T_HXA_start) ## Start temperature

        for t in self.T[0:-1]:
            self.m.Constraint_HXA_T.add(self.m.T_HXA_T[t+1] == self.m.T_HXA_T[t] + self.StepSizeInSec * (1/(self.m_HXA_b * self.c_b) * (self.alpha_factor_HXA * self.c_a * self.mdot_HXA_a * self.m.Z_HXA_out_T[t] + self.c_b * self.mdot_HXA_b * self.m.Z_T_HXA_HXH_T[t] + self.c_b * self.mdot_HXA_b * self.m.Z_T_HXA_HGC_T[t])) + self.StepSizeInSec * self.alpha_HXA_time * (self.temp_amb_T[t] - self.m.T_HXA_T[t+1])/(self.m_HXA_b * self.c_b)) ## General energy flow
        
        for t in self.T[0:-1]:
            self.m.Constraint_HXA_T.add(self.m.Z_HXA_out_T[t] <= self.T_HXA_delta_max) ## Big M constraint input
            self.m.Constraint_HXA_T.add(self.m.Z_HXA_out_T[t] >= self.T_HXA_delta_min) ## Big M constraint input
            self.m.Constraint_HXA_T.add(self.m.Z_HXA_out_T[t] <= self.T_HXA_delta_max * self.m.Z_HXA_T[t]) ## Big M constraint input
            self.m.Constraint_HXA_T.add(self.m.Z_HXA_out_T[t] >= self.T_HXA_delta_min * self.m.Z_HXA_T[t]) ## Big M constraint input
            self.m.Constraint_HXA_T.add(self.m.Z_HXA_out_T[t] <= (self.m.T_HXA_in_T[t] - self.m.T_HXA_T[t+1]) - (1 - self.m.Z_HXA_T[t]) * self.T_HXA_delta_min) ## Big M constraint input
            self.m.Constraint_HXA_T.add(self.m.Z_HXA_out_T[t] >= (self.m.T_HXA_in_T[t] - self.m.T_HXA_T[t+1]) - (1 - self.m.Z_HXA_T[t]) * self.T_HXA_delta_max) ## Big M constraint input
            
            self.m.Constraint_HXA_T.add(self.m.Z_T_HXA_HXH_T[t] <= self.T_HXA_delta_max) ## Big M constraint input
            self.m.Constraint_HXA_T.add(self.m.Z_T_HXA_HXH_T[t] >= self.T_HXA_delta_min) ## Big M constraint input
            self.m.Constraint_HXA_T.add(self.m.Z_T_HXA_HXH_T[t] <= self.T_HXA_delta_max * self.m.Z_HXA_HXH_T[t]) ## Big M constraint input
            self.m.Constraint_HXA_T.add(self.m.Z_T_HXA_HXH_T[t] >= self.T_HXA_delta_min * self.m.Z_HXA_HXH_T[t]) ## Big M constraint input
            self.m.Constraint_HXA_T.add(self.m.Z_T_HXA_HXH_T[t] <= (self.m.T_HXA_HXH_T[t] - self.m.T_HXA_T[t+1]) - (1 - self.m.Z_HXA_HXH_T[t]) * self.T_HXA_delta_min) ## Big M constraint input
            self.m.Constraint_HXA_T.add(self.m.Z_T_HXA_HXH_T[t] >= (self.m.T_HXA_HXH_T[t] - self.m.T_HXA_T[t+1]) - (1 - self.m.Z_HXA_HXH_T[t]) * self.T_HXA_delta_max) ## Big M constraint input
            
            self.m.Constraint_HXA_T.add(self.m.Z_T_HXA_HGC_T[t] <= self.T_HXA_delta_max) ## Big M constraint input
            self.m.Constraint_HXA_T.add(self.m.Z_T_HXA_HGC_T[t] >= self.T_HXA_delta_min) ## Big M constraint input
            self.m.Constraint_HXA_T.add(self.m.Z_T_HXA_HGC_T[t] <= self.T_HXA_delta_max * self.m.Z_HXA_HGC_T[t]) ## Big M constraint input
            self.m.Constraint_HXA_T.add(self.m.Z_T_HXA_HGC_T[t] >= self.T_HXA_delta_min * self.m.Z_HXA_HGC_T[t]) ## Big M constraint input
            self.m.Constraint_HXA_T.add(self.m.Z_T_HXA_HGC_T[t] <= (self.m.T_HXA_HGC_T[t] - self.m.T_HXA_T[t+1]) - (1 - self.m.Z_HXA_HGC_T[t]) * self.T_HXA_delta_min) ## Big M constraint input
            self.m.Constraint_HXA_T.add(self.m.Z_T_HXA_HGC_T[t] >= (self.m.T_HXA_HGC_T[t] - self.m.T_HXA_T[t+1]) - (1 - self.m.Z_HXA_HGC_T[t]) * self.T_HXA_delta_max) ## Big M constraint input
            
        for t in self.T[1:]:
            self.m.Constraint_HXA_T.add(self.m.T_HXA_T[t] <= self.T_HXA_max + self.m.S_T_HXA_T[t]) ## Temperature range tank
            self.m.Constraint_HXA_T.add(self.m.T_HXA_T[t] >= self.T_HXA_min - self.m.S_T_HXA_T[t]) ## Temperature range tank

        ## HXH
        self.m.Constraint_HXH_T = pyo.ConstraintList()

        for t in self.T[0:-1]:
            self.m.Constraint_HXH_T.add(0 == sum(self.m.Z_HXH_w_H_T[h,t] * self.mdot_HP_w_H[h] * self.c_w for h in self.H) + self.m.Qdot_HXH_w_b_T[t])
            self.m.Constraint_HXH_T.add(0 == self.m.Z_HXH_b_T[t] * self.mdot_HXA_b * self.c_b - self.m.Qdot_HXH_w_b_T[t])
            self.m.Constraint_HXH_T.add(self.m.Qdot_HXH_w_b_T[t] == ((self.m.T_HXH_w_in_T[t] + self.m.T_HXH_w_out_T[t])/2 - (self.m.T_HXH_b_in_T[t] + self.m.T_HXH_b_out_T[t])/2 ) * self.a_HXH_w_b * self.alpha_HXH_w_b)

        for t in self.T[0:-1]:
            for h in self.H:
                self.m.Constraint_HXH_T.add(self.m.Z_HXH_w_H_T[h,t] <= self.T_HXH_delta_max) ## Big M constraint input on water side
                self.m.Constraint_HXH_T.add(self.m.Z_HXH_w_H_T[h,t] >= self.T_HXH_delta_min) ## Big M constraint input on water side
                self.m.Constraint_HXH_T.add(self.m.Z_HXH_w_H_T[h,t] <= self.T_HXH_delta_max * self.m.Z_HP_HXH_H_T[h,t]) ## Big M constraint input on water side
                self.m.Constraint_HXH_T.add(self.m.Z_HXH_w_H_T[h,t] >= self.T_HXH_delta_min * self.m.Z_HP_HXH_H_T[h,t]) ## Big M constraint input on water side
                self.m.Constraint_HXH_T.add(self.m.Z_HXH_w_H_T[h,t] <= (self.m.T_HXH_w_in_T[t] - self.m.T_HXH_w_out_T[t]) - (1 - self.m.Z_HP_HXH_H_T[h,t]) * self.T_HXH_delta_min) ## Big M constraint input on water side
                self.m.Constraint_HXH_T.add(self.m.Z_HXH_w_H_T[h,t] >= (self.m.T_HXH_w_in_T[t] - self.m.T_HXH_w_out_T[t]) - (1 - self.m.Z_HP_HXH_H_T[h,t]) * self.T_HXH_delta_max) ## Big M constraint input on water side

            self.m.Constraint_HXH_T.add(self.m.Z_HXH_b_T[t] <= self.T_HXH_delta_max) ## Big M constraint input on broil side
            self.m.Constraint_HXH_T.add(self.m.Z_HXH_b_T[t] >= self.T_HXH_delta_min) ## Big M constraint input on broil side
            self.m.Constraint_HXH_T.add(self.m.Z_HXH_b_T[t] <= self.T_HXH_delta_max * self.m.Z_HXA_HXH_T[t]) ## Big M constraint input on broil side
            self.m.Constraint_HXH_T.add(self.m.Z_HXH_b_T[t] >= self.T_HXH_delta_min * self.m.Z_HXA_HXH_T[t]) ## Big M constraint input on broil side
            self.m.Constraint_HXH_T.add(self.m.Z_HXH_b_T[t] <= (self.m.T_HXH_b_in_T[t] - self.m.T_HXH_b_out_T[t]) - (1 - self.m.Z_HXA_HXH_T[t]) * self.T_HXH_delta_min) ## Big M constraint input on broil side
            self.m.Constraint_HXH_T.add(self.m.Z_HXH_b_T[t] >= (self.m.T_HXH_b_in_T[t] - self.m.T_HXH_b_out_T[t]) - (1 - self.m.Z_HXA_HXH_T[t]) * self.T_HXH_delta_max) ## Big M constraint input on broil side

        for t in self.T[0:-1]:
            self.m.Constraint_HXH_T.add(self.m.T_HXH_T[t] == (self.m.T_HXH_w_in_T[t] + self.m.T_HXH_w_out_T[t] + self.m.T_HXH_b_in_T[t] + self.m.T_HXH_b_out_T[t])/4)
        
        for t in self.T[1:-1]:
            self.m.Constraint_HXH_T.add(self.m.T_HXH_T[t] <= self.T_HXH_max + self.m.S_T_HXH_T[t]) ## Temperature range 
            self.m.Constraint_HXH_T.add(self.m.T_HXH_T[t] >= self.T_HXH_min - self.m.S_T_HXH_T[t]) ## Temperature range 

        ## HGC
        self.m.Constraint_HGC_T = pyo.ConstraintList()
        self.m.Constraint_HGC_T.add(self.m.T_HGC_T[0] == self.T_HGC_start) ## Start temperature

        for t in self.T[0:-1]:
            self.m.Constraint_HGC_T.add(self.m.T_HGC_T[t+1] == self.m.T_HGC_T[t] + self.StepSizeInSec * (1/(self.m_HGC_b * self.c_b) * (self.c_b * sum(self.mdot_HP_b_H[h] * self.m.Z_HGC_HP_H_T[h,t] for h in self.H) + self.c_b * sum(self.mdot_HP_b_H[h] * self.m.Z_HGC_HXC_H_T[h,t] for h in self.H) + self.c_b * self.mdot_HXA_b * self.m.Z_HGC_HXA_T[t])) + self.StepSizeInSec * self.alpha_HGC_time * (self.t_default - self.m.T_HGC_T[t+1])/(self.m_HGC_b * self.c_b)) ## General energy flow

        for t in self.T[0:-1]:
            for h in self.H:
                self.m.Constraint_HGC_T.add(self.m.Z_HGC_HP_H_T[h,t] <= self.T_HGC_delta_max) ## Big M constraint input
                self.m.Constraint_HGC_T.add(self.m.Z_HGC_HP_H_T[h,t] >= self.T_HGC_delta_min) ## Big M constraint input
                self.m.Constraint_HGC_T.add(self.m.Z_HGC_HP_H_T[h,t] <= self.T_HGC_delta_max * self.m.Z_HP_HGC_H_T[h,t]) ## Big M constraint input
                self.m.Constraint_HGC_T.add(self.m.Z_HGC_HP_H_T[h,t] >= self.T_HGC_delta_min * self.m.Z_HP_HGC_H_T[h,t]) ## Big M constraint input
                self.m.Constraint_HGC_T.add(self.m.Z_HGC_HP_H_T[h,t] <= (self.m.T_HGC_HP_T[t] - self.m.T_HGC_T[t+1]) - (1 - self.m.Z_HP_HGC_H_T[h,t]) * self.T_HGC_delta_min) ## Big M constraint input
                self.m.Constraint_HGC_T.add(self.m.Z_HGC_HP_H_T[h,t] >= (self.m.T_HGC_HP_T[t] - self.m.T_HGC_T[t+1]) - (1 - self.m.Z_HP_HGC_H_T[h,t]) * self.T_HGC_delta_max) ## Big M constraint input
                                
                self.m.Constraint_HP_T.add(self.m.Z_HGC_HXC_H_T[h,t] <= self.T_HGC_delta_max) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HGC_HXC_H_T[h,t] >= self.T_HGC_delta_min) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HGC_HXC_H_T[h,t] <= self.T_HGC_delta_max * self.m.Z_HP_HGCHXC_H_T[h,t]) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HGC_HXC_H_T[h,t] >= self.T_HGC_delta_min * self.m.Z_HP_HGCHXC_H_T[h,t]) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HGC_HXC_H_T[h,t] <= (self.m.T_HGC_HXC_T[t] - self.m.T_HGC_T[t+1]) - (1 - self.m.Z_HP_HGCHXC_H_T[h,t]) * self.T_HGC_delta_min) ## Big M constraint input
                self.m.Constraint_HP_T.add(self.m.Z_HGC_HXC_H_T[h,t] >= (self.m.T_HGC_HXC_T[t] - self.m.T_HGC_T[t+1]) - (1 - self.m.Z_HP_HGCHXC_H_T[h,t]) * self.T_HGC_delta_max) ## Big M constraint input
                
            self.m.Constraint_HGC_T.add(self.m.Z_HGC_HXA_T[t] <= self.T_HGC_delta_max) ## Big M constraint input
            self.m.Constraint_HGC_T.add(self.m.Z_HGC_HXA_T[t] >= self.T_HGC_delta_min) ## Big M constraint input
            self.m.Constraint_HGC_T.add(self.m.Z_HGC_HXA_T[t] <= self.T_HGC_delta_max * self.m.Z_HXA_HGC_T[t]) ## Big M constraint input
            self.m.Constraint_HGC_T.add(self.m.Z_HGC_HXA_T[t] >= self.T_HGC_delta_min * self.m.Z_HXA_HGC_T[t]) ## Big M constraint input
            self.m.Constraint_HGC_T.add(self.m.Z_HGC_HXA_T[t] <= (self.m.T_HGC_HXA_T[t] - self.m.T_HGC_T[t+1]) - (1 - self.m.Z_HXA_HGC_T[t]) * self.T_HGC_delta_min) ## Big M constraint input
            self.m.Constraint_HGC_T.add(self.m.Z_HGC_HXA_T[t] >= (self.m.T_HGC_HXA_T[t] - self.m.T_HGC_T[t+1]) - (1 - self.m.Z_HXA_HGC_T[t]) * self.T_HGC_delta_max) ## Big M constraint input

        for t in self.T[1:]:    
            self.m.Constraint_HGC_T.add(self.m.T_HGC_T[t] <= self.T_HGC_max + self.m.S_T_HGC_T[t]) ## Temperature range tank
            self.m.Constraint_HGC_T.add(self.m.T_HGC_T[t] >= self.T_HGC_min - self.m.S_T_HGC_T[t]) ## Temperature range tank  
        
        ## HXC
        self.m.Constraint_HXC_T = pyo.ConstraintList()
        
        if self.TControlPeriodSwitch1 > 0:
            for t in self.T[0:self.TControlPeriodSwitch1]:
                self.m.Constraint_HXC_T.add(0 == sum(self.m.Z_HXC_HGS_V_T_1[v,t] * self.mdot_VP_HGS_V_1[v] * self.c_w + self.m.Z_HXC_CS_V_T_1[v,t] * self.mdot_VP_CS_V_1[v] * self.c_w + self.m.Z_HXC_RLTS_V_T_1[v,t] * self.mdot_VP_RLTS_V_1[v] * self.c_w for v in self.V_1) + self.m.Qdot_HXC_w_b_T[t])
                self.m.Constraint_HXC_T.add(0 == sum(self.m.Z_HXC_b_H_T[h,t] * self.mdot_HP_b_H[h] * self.c_b for h in self.H) - self.m.Qdot_HXC_w_b_T[t])
                self.m.Constraint_HXC_T.add(self.m.T_HXC_w_delta_in_T[t] == (sum(self.m.Z_HGS_V_T_1[v,t] * self.mdot_VP_HGS_V_1[v] + self.m.Z_CS_V_T_1[v,t] * self.mdot_VP_CS_V_1[v] + self.m.Z_RLTS_V_T_1[v,t] * self.mdot_VP_RLTS_V_1[v] for v in self.V_1) + (1/3 * self.m.Z_HGS_V_T_1[0,t] * self.mdot_VP_tot + 1/3 * self.m.Z_CS_V_T_1[0,t] * self.mdot_VP_tot + 1/3 * self.m.Z_RLTS_V_T_1[0,t] * self.mdot_VP_tot))/self.mdot_VP_tot)
                self.m.Constraint_HXC_T.add(self.m.Qdot_HXC_w_b_T[t] == ((self.m.T_HXC_w_delta_in_T[t])/2 - (self.m.T_HXC_b_in_T[t] + self.m.T_HXC_b_out_T[t])/2) * self.a_HXC_w_b * self.alpha_HXC_w_b) 

                for h in self.H:
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_b_H_T[h,t] <= self.T_HXC_delta_max) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_b_H_T[h,t] >= self.T_HXC_delta_min) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_b_H_T[h,t] <= self.T_HXC_delta_max * self.m.Z_HP_HGCHXC_H_T[h,t]) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_b_H_T[h,t] >= self.T_HXC_delta_min * self.m.Z_HP_HGCHXC_H_T[h,t]) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_b_H_T[h,t] <= (self.m.T_HXC_b_in_T[t] - self.m.T_HXC_b_out_T[t]) - (1 - self.m.Z_HP_HGCHXC_H_T[h,t]) * self.T_HXC_delta_min) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_b_H_T[h,t] >= (self.m.T_HXC_b_in_T[t] - self.m.T_HXC_b_out_T[t]) - (1 - self.m.Z_HP_HGCHXC_H_T[h,t]) * self.T_HXC_delta_max) ## Big M constraint input
                
                for v in self.V_1:
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_HGS_V_T_1[v,t] <= self.T_HXC_delta_max) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_HGS_V_T_1[v,t] >= self.T_HXC_delta_min) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_HGS_V_T_1[v,t] <= self.T_HXC_delta_max * self.m.B_VP_V_T_1[v,t]) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_HGS_V_T_1[v,t] >= self.T_HXC_delta_min * self.m.B_VP_V_T_1[v,t]) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_HGS_V_T_1[v,t] <= (self.m.T_HXC_HGS_T[t] - self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_1[v,t]) * self.T_HXC_delta_min) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_HGS_V_T_1[v,t] >= (self.m.T_HXC_HGS_T[t] - self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_1[v,t]) * self.T_HXC_delta_max) ## Big M constraint input

                for v in self.V_1:
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_CS_V_T_1[v,t] <= self.T_HXC_delta_max) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_CS_V_T_1[v,t] >= self.T_HXC_delta_min) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_CS_V_T_1[v,t] <= self.T_HXC_delta_max * self.m.B_VP_V_T_1[v,t]) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_CS_V_T_1[v,t] >= self.T_HXC_delta_min * self.m.B_VP_V_T_1[v,t]) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_CS_V_T_1[v,t] <= (self.m.T_HXC_CS_T[t] - self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_1[v,t]) * self.T_HXC_delta_min) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_CS_V_T_1[v,t] >= (self.m.T_HXC_CS_T[t] - self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_1[v,t]) * self.T_HXC_delta_max) ## Big M constraint input
        
                for v in self.V_1:
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_RLTS_V_T_1[v,t] <= self.T_HXC_delta_max) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_RLTS_V_T_1[v,t] >= self.T_HXC_delta_min) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_RLTS_V_T_1[v,t] <= self.T_HXC_delta_max * self.m.B_VP_V_T_1[v,t]) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_RLTS_V_T_1[v,t] >= self.T_HXC_delta_min * self.m.B_VP_V_T_1[v,t]) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_RLTS_V_T_1[v,t] <= (self.m.T_HXC_RLTS_T[t] - self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_1[v,t]) * self.T_HXC_delta_min) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HXC_RLTS_V_T_1[v,t] >= (self.m.T_HXC_RLTS_T[t] - self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_1[v,t]) * self.T_HXC_delta_max) ## Big M constraint input

                for v in self.V_1:
                    self.m.Constraint_HXC_T.add(self.m.Z_HGS_V_T_1[v,t] <= self.T_HXC_delta_max) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HGS_V_T_1[v,t] >= self.T_HXC_delta_min) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HGS_V_T_1[v,t] <= self.T_HXC_delta_max * self.m.B_VP_V_T_1[v,t]) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HGS_V_T_1[v,t] >= self.T_HXC_delta_min * self.m.B_VP_V_T_1[v,t]) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HGS_V_T_1[v,t] <= (self.m.T_HXC_HGS_T[t] + self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_1[v,t]) * self.T_HXC_delta_min) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_HGS_V_T_1[v,t] >= (self.m.T_HXC_HGS_T[t] + self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_1[v,t]) * self.T_HXC_delta_max) ## Big M constraint input

                for v in self.V_1:
                    self.m.Constraint_HXC_T.add(self.m.Z_CS_V_T_1[v,t] <= self.T_HXC_delta_max) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_CS_V_T_1[v,t] >= self.T_HXC_delta_min) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_CS_V_T_1[v,t] <= self.T_HXC_delta_max * self.m.B_VP_V_T_1[v,t]) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_CS_V_T_1[v,t] >= self.T_HXC_delta_min * self.m.B_VP_V_T_1[v,t]) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_CS_V_T_1[v,t] <= (self.m.T_HXC_CS_T[t] + self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_1[v,t]) * self.T_HXC_delta_min) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_CS_V_T_1[v,t] >= (self.m.T_HXC_CS_T[t] + self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_1[v,t]) * self.T_HXC_delta_max) ## Big M constraint input
        
                for v in self.V_1:
                    self.m.Constraint_HXC_T.add(self.m.Z_RLTS_V_T_1[v,t] <= self.T_HXC_delta_max) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_RLTS_V_T_1[v,t] >= self.T_HXC_delta_min) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_RLTS_V_T_1[v,t] <= self.T_HXC_delta_max * self.m.B_VP_V_T_1[v,t]) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_RLTS_V_T_1[v,t] >= self.T_HXC_delta_min * self.m.B_VP_V_T_1[v,t]) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_RLTS_V_T_1[v,t] <= (self.m.T_HXC_RLTS_T[t] + self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_1[v,t]) * self.T_HXC_delta_min) ## Big M constraint input
                    self.m.Constraint_HXC_T.add(self.m.Z_RLTS_V_T_1[v,t] >= (self.m.T_HXC_RLTS_T[t] + self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_1[v,t]) * self.T_HXC_delta_max) ## Big M constraint input

                self.m.Constraint_HXC_T.add((self.m.T_HXC_RLTS_T[t] - self.m.T_HXC_w_out_T[t]) == sum(self.m.Z_HXC_RLTS_V_T_1[v,t]  for v in self.V_1))    
                self.m.Constraint_HXC_T.add((self.m.T_HXC_CS_T[t] - self.m.T_HXC_w_out_T[t]) == sum(self.m.Z_HXC_CS_V_T_1[v,t]  for v in self.V_1))   
                self.m.Constraint_HXC_T.add((self.m.T_HXC_HGS_T[t] - self.m.T_HXC_w_out_T[t]) == sum(self.m.Z_HXC_HGS_V_T_1[v,t]  for v in self.V_1))   

        for t in self.T[self.TControlPeriodSwitch1:-1]:
            self.m.Constraint_HXC_T.add(0 == sum(self.m.Z_HXC_HGS_V_T_2[v,t] * self.mdot_VP_HGS_V_2[v] * self.c_w + self.m.Z_HXC_CS_V_T_2[v,t] * self.mdot_VP_CS_V_2[v] * self.c_w + self.m.Z_HXC_RLTS_V_T_2[v,t] * self.mdot_VP_RLTS_V_2[v] * self.c_w for v in self.V_2) + self.m.Qdot_HXC_w_b_T[t])
            self.m.Constraint_HXC_T.add(0 == sum(self.m.Z_HXC_b_H_T[h,t] * self.mdot_HP_b_H[h] * self.c_b for h in self.H) - self.m.Qdot_HXC_w_b_T[t])
            self.m.Constraint_HXC_T.add(self.m.T_HXC_w_delta_in_T[t] == (sum(self.m.Z_HGS_V_T_2[v,t] * self.mdot_VP_HGS_V_2[v] + self.m.Z_CS_V_T_2[v,t] * self.mdot_VP_CS_V_2[v] + self.m.Z_RLTS_V_T_2[v,t] * self.mdot_VP_RLTS_V_2[v] for v in self.V_2) + (1/3 * self.m.Z_HGS_V_T_2[0,t] * self.mdot_VP_tot + 1/3 * self.m.Z_CS_V_T_2[0,t] * self.mdot_VP_tot + 1/3 * self.m.Z_RLTS_V_T_2[0,t] * self.mdot_VP_tot))/self.mdot_VP_tot)
            self.m.Constraint_HXC_T.add(self.m.Qdot_HXC_w_b_T[t] == ((self.m.T_HXC_w_delta_in_T[t])/2 - (self.m.T_HXC_b_in_T[t] + self.m.T_HXC_b_out_T[t])/2) * self.a_HXC_w_b * self.alpha_HXC_w_b) 

            for h in self.H:
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_b_H_T[h,t] <= self.T_HXC_delta_max) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_b_H_T[h,t] >= self.T_HXC_delta_min) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_b_H_T[h,t] <= self.T_HXC_delta_max * self.m.Z_HP_HGCHXC_H_T[h,t]) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_b_H_T[h,t] >= self.T_HXC_delta_min * self.m.Z_HP_HGCHXC_H_T[h,t]) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_b_H_T[h,t] <= (self.m.T_HXC_b_in_T[t] - self.m.T_HXC_b_out_T[t]) - (1 - self.m.Z_HP_HGCHXC_H_T[h,t]) * self.T_HXC_delta_min) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_b_H_T[h,t] >= (self.m.T_HXC_b_in_T[t] - self.m.T_HXC_b_out_T[t]) - (1 - self.m.Z_HP_HGCHXC_H_T[h,t]) * self.T_HXC_delta_max) ## Big M constraint input
            
            for v in self.V_2:
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_HGS_V_T_2[v,t] <= self.T_HXC_delta_max) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_HGS_V_T_2[v,t] >= self.T_HXC_delta_min) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_HGS_V_T_2[v,t] <= self.T_HXC_delta_max * self.m.B_VP_V_T_2[v,t]) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_HGS_V_T_2[v,t] >= self.T_HXC_delta_min * self.m.B_VP_V_T_2[v,t]) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_HGS_V_T_2[v,t] <= (self.m.T_HXC_HGS_T[t] - self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_2[v,t]) * self.T_HXC_delta_min) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_HGS_V_T_2[v,t] >= (self.m.T_HXC_HGS_T[t] - self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_2[v,t]) * self.T_HXC_delta_max) ## Big M constraint input

            for v in self.V_2:
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_CS_V_T_2[v,t] <= self.T_HXC_delta_max) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_CS_V_T_2[v,t] >= self.T_HXC_delta_min) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_CS_V_T_2[v,t] <= self.T_HXC_delta_max * self.m.B_VP_V_T_2[v,t]) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_CS_V_T_2[v,t] >= self.T_HXC_delta_min * self.m.B_VP_V_T_2[v,t]) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_CS_V_T_2[v,t] <= (self.m.T_HXC_CS_T[t] - self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_2[v,t]) * self.T_HXC_delta_min) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_CS_V_T_2[v,t] >= (self.m.T_HXC_CS_T[t] - self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_2[v,t]) * self.T_HXC_delta_max) ## Big M constraint input
    
            for v in self.V_2:
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_RLTS_V_T_2[v,t] <= self.T_HXC_delta_max) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_RLTS_V_T_2[v,t] >= self.T_HXC_delta_min) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_RLTS_V_T_2[v,t] <= self.T_HXC_delta_max * self.m.B_VP_V_T_2[v,t]) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_RLTS_V_T_2[v,t] >= self.T_HXC_delta_min * self.m.B_VP_V_T_2[v,t]) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_RLTS_V_T_2[v,t] <= (self.m.T_HXC_RLTS_T[t] - self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_2[v,t]) * self.T_HXC_delta_min) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HXC_RLTS_V_T_2[v,t] >= (self.m.T_HXC_RLTS_T[t] - self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_2[v,t]) * self.T_HXC_delta_max) ## Big M constraint input

            for v in self.V_2:
                self.m.Constraint_HXC_T.add(self.m.Z_HGS_V_T_2[v,t] <= self.T_HXC_delta_max) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HGS_V_T_2[v,t] >= self.T_HXC_delta_min) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HGS_V_T_2[v,t] <= self.T_HXC_delta_max * self.m.B_VP_V_T_2[v,t]) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HGS_V_T_2[v,t] >= self.T_HXC_delta_min * self.m.B_VP_V_T_2[v,t]) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HGS_V_T_2[v,t] <= (self.m.T_HXC_HGS_T[t] + self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_2[v,t]) * self.T_HXC_delta_min) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_HGS_V_T_2[v,t] >= (self.m.T_HXC_HGS_T[t] + self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_2[v,t]) * self.T_HXC_delta_max) ## Big M constraint input

            for v in self.V_2:
                self.m.Constraint_HXC_T.add(self.m.Z_CS_V_T_2[v,t] <= self.T_HXC_delta_max) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_CS_V_T_2[v,t] >= self.T_HXC_delta_min) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_CS_V_T_2[v,t] <= self.T_HXC_delta_max * self.m.B_VP_V_T_2[v,t]) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_CS_V_T_2[v,t] >= self.T_HXC_delta_min * self.m.B_VP_V_T_2[v,t]) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_CS_V_T_2[v,t] <= (self.m.T_HXC_CS_T[t] + self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_2[v,t]) * self.T_HXC_delta_min) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_CS_V_T_2[v,t] >= (self.m.T_HXC_CS_T[t] + self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_2[v,t]) * self.T_HXC_delta_max) ## Big M constraint input
    
            for v in self.V_2:
                self.m.Constraint_HXC_T.add(self.m.Z_RLTS_V_T_2[v,t] <= self.T_HXC_delta_max) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_RLTS_V_T_2[v,t] >= self.T_HXC_delta_min) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_RLTS_V_T_2[v,t] <= self.T_HXC_delta_max * self.m.B_VP_V_T_2[v,t]) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_RLTS_V_T_2[v,t] >= self.T_HXC_delta_min * self.m.B_VP_V_T_2[v,t]) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_RLTS_V_T_2[v,t] <= (self.m.T_HXC_RLTS_T[t] + self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_2[v,t]) * self.T_HXC_delta_min) ## Big M constraint input
                self.m.Constraint_HXC_T.add(self.m.Z_RLTS_V_T_2[v,t] >= (self.m.T_HXC_RLTS_T[t] + self.m.T_HXC_w_out_T[t]) - (1 - self.m.B_VP_V_T_2[v,t]) * self.T_HXC_delta_max) ## Big M constraint input

            self.m.Constraint_HXC_T.add((self.m.T_HXC_RLTS_T[t] - self.m.T_HXC_w_out_T[t]) == sum(self.m.Z_HXC_RLTS_V_T_2[v,t]  for v in self.V_2))    
            self.m.Constraint_HXC_T.add((self.m.T_HXC_CS_T[t] - self.m.T_HXC_w_out_T[t]) == sum(self.m.Z_HXC_CS_V_T_2[v,t]  for v in self.V_2))   
            self.m.Constraint_HXC_T.add((self.m.T_HXC_HGS_T[t] - self.m.T_HXC_w_out_T[t]) == sum(self.m.Z_HXC_HGS_V_T_2[v,t]  for v in self.V_2))   

        for t in self.T[0:-1]:    
            self.m.Constraint_HXC_T.add(self.m.T_HXC_T[t] == (self.m.T_HXC_w_delta_in_T[t] + self.m.T_HXC_w_out_T[t] + self.m.T_HXC_w_out_T[t] + self.m.T_HXC_b_in_T[t] + self.m.T_HXC_b_out_T[t])/4)

        for t in self.T[1:]:
            self.m.Constraint_HXC_T.add(self.m.T_HXC_T[t] <= self.T_HXC_max + self.m.S_T_HXC_T[t]) ## Temperature range tank
            self.m.Constraint_HXC_T.add(self.m.T_HXC_T[t] >= self.T_HXC_min - self.m.S_T_HXC_T[t]) ## Temperature range tank  

        ## HGS
        self.m.Constraint_HGS_T = pyo.ConstraintList()
        self.m.Constraint_HGS_T.add(self.m.T_HGS_T[0] == self.T_HGS_start) ## Start temperature

        if self.TControlPeriodSwitch1 > 0:
            for t in self.T[0:self.TControlPeriodSwitch1]:
                self.m.Constraint_HGS_T.add(self.m.T_HGS_T[t+1] == self.m.T_HGS_T[t] + self.StepSizeInSec * (1/(self.m_HGS_w * self.c_w) * (self.c_w * self.mdot_IS_w_2 * self.m.Z_HGS_IS_T[t] + self.c_w * self.mdot_IS_w_2 * self.m.Z_HGS_IS_2_T[t] + self.c_w * self.mdot_GS_w * self.m.Z_HGS_GS_T[t] + self.c_w * self.mdot_GS_w_2 * self.m.Z_HGS_GS_2_T[t] + self.c_w * sum(self.mdot_VP_HGS_V_1[v] * self.m.Z_HGS_HXC_V_T_1[v,t] for v in self.V_1))) + self.StepSizeInSec * self.alpha_HGS_time * (self.t_default - self.m.T_HGS_T[t+1])/(self.m_HGS_w * self.c_w)) ## General energy flow

        for t in self.T[self.TControlPeriodSwitch1:-1]:
            self.m.Constraint_HGS_T.add(self.m.T_HGS_T[t+1] == self.m.T_HGS_T[t] + self.StepSizeInSec * (1/(self.m_HGS_w * self.c_w) * (self.c_w * self.mdot_IS_w_2 * self.m.Z_HGS_IS_T[t] + self.c_w * self.mdot_IS_w_2 * self.m.Z_HGS_IS_2_T[t] + self.c_w * self.mdot_GS_w * self.m.Z_HGS_GS_T[t] + self.c_w * self.mdot_GS_w_2 * self.m.Z_HGS_GS_2_T[t] + self.c_w * sum(self.mdot_VP_HGS_V_2[v] * self.m.Z_HGS_HXC_V_T_2[v,t] for v in self.V_2))) + self.StepSizeInSec * self.alpha_HGS_time * (self.t_default - self.m.T_HGS_T[t+1])/(self.m_HGS_w * self.c_w)) ## General energy flow

        for t in self.T[0:-1]:            
            self.m.Constraint_HGS_T.add(self.m.Z_HGS_IS_T[t] <= self.T_HGS_delta_max) ## Big M constraint input
            self.m.Constraint_HGS_T.add(self.m.Z_HGS_IS_T[t] >= self.T_HGS_delta_min) ## Big M constraint input
            self.m.Constraint_HGS_T.add(self.m.Z_HGS_IS_T[t] <= self.T_HGS_delta_max * self.m.B_IS_HGS_T[t]) ## Big M constraint input
            self.m.Constraint_HGS_T.add(self.m.Z_HGS_IS_T[t] >= self.T_HGS_delta_min * self.m.B_IS_HGS_T[t]) ## Big M constraint input
            self.m.Constraint_HGS_T.add(self.m.Z_HGS_IS_T[t] <= (self.m.T_HGS_IS_T[t] - self.m.T_HGS_T[t+1]) - (1 - self.m.B_IS_HGS_T[t]) * self.T_HGS_delta_min) ## Big M constraint input
            self.m.Constraint_HGS_T.add(self.m.Z_HGS_IS_T[t] >= (self.m.T_HGS_IS_T[t] - self.m.T_HGS_T[t+1]) - (1 - self.m.B_IS_HGS_T[t]) * self.T_HGS_delta_max) ## Big M constraint input

            self.m.Constraint_HGS_T.add(self.m.Z_HGS_IS_2_T[t] <= self.T_HGS_delta_max) ## Big M constraint input
            self.m.Constraint_HGS_T.add(self.m.Z_HGS_IS_2_T[t] >= self.T_HGS_delta_min) ## Big M constraint input
            self.m.Constraint_HGS_T.add(self.m.Z_HGS_IS_2_T[t] <= self.T_HGS_delta_max * self.m.Z_IS_HGS_2_T[t]) ## Big M constraint input
            self.m.Constraint_HGS_T.add(self.m.Z_HGS_IS_2_T[t] >= self.T_HGS_delta_min * self.m.Z_IS_HGS_2_T[t]) ## Big M constraint input
            self.m.Constraint_HGS_T.add(self.m.Z_HGS_IS_2_T[t] <= (self.m.T_HGS_IS_T[t] - self.m.T_HGS_T[t+1]) - (1 - self.m.Z_IS_HGS_2_T[t]) * self.T_HGS_delta_min) ## Big M constraint input
            self.m.Constraint_HGS_T.add(self.m.Z_HGS_IS_2_T[t] >= (self.m.T_HGS_IS_T[t] - self.m.T_HGS_T[t+1]) - (1 - self.m.Z_IS_HGS_2_T[t]) * self.T_HGS_delta_max) ## Big M constraint input

            self.m.Constraint_HGS_T.add(self.m.Z_HGS_GS_T[t] <= self.T_HGS_delta_max) ## Big M constraint input
            self.m.Constraint_HGS_T.add(self.m.Z_HGS_GS_T[t] >= self.T_HGS_delta_min) ## Big M constraint input
            self.m.Constraint_HGS_T.add(self.m.Z_HGS_GS_T[t] <= self.T_HGS_delta_max * self.m.B_GS_HGS_T[t]) ## Big M constraint input
            self.m.Constraint_HGS_T.add(self.m.Z_HGS_GS_T[t] >= self.T_HGS_delta_min * self.m.B_GS_HGS_T[t]) ## Big M constraint input
            self.m.Constraint_HGS_T.add(self.m.Z_HGS_GS_T[t] <= (self.m.T_HGS_GS_T[t] - self.m.T_HGS_T[t+1]) - (1 - self.m.B_GS_HGS_T[t]) * self.T_HGS_delta_min) ## Big M constraint input
            self.m.Constraint_HGS_T.add(self.m.Z_HGS_GS_T[t] >= (self.m.T_HGS_GS_T[t] - self.m.T_HGS_T[t+1]) - (1 - self.m.B_GS_HGS_T[t]) * self.T_HGS_delta_max) ## Big M constraint input

            self.m.Constraint_HGS_T.add(self.m.Z_HGS_GS_2_T[t] <= self.T_HGS_delta_max) ## Big M constraint input
            self.m.Constraint_HGS_T.add(self.m.Z_HGS_GS_2_T[t] >= self.T_HGS_delta_min) ## Big M constraint input
            self.m.Constraint_HGS_T.add(self.m.Z_HGS_GS_2_T[t] <= self.T_HGS_delta_max * self.m.B_GS_HGS_CS_T[t]) ## Big M constraint input
            self.m.Constraint_HGS_T.add(self.m.Z_HGS_GS_2_T[t] >= self.T_HGS_delta_min * self.m.B_GS_HGS_CS_T[t]) ## Big M constraint input
            self.m.Constraint_HGS_T.add(self.m.Z_HGS_GS_2_T[t] <= (self.m.T_HGS_GS_T[t] - self.m.T_HGS_T[t+1]) - (1 - self.m.B_GS_HGS_CS_T[t]) * self.T_HGS_delta_min) ## Big M constraint input
            self.m.Constraint_HGS_T.add(self.m.Z_HGS_GS_2_T[t] >= (self.m.T_HGS_GS_T[t] - self.m.T_HGS_T[t+1]) - (1 - self.m.B_GS_HGS_CS_T[t]) * self.T_HGS_delta_max) ## Big M constraint input

        if self.TControlPeriodSwitch1 > 0:
            for t in self.T[0:self.TControlPeriodSwitch1]:
                for v in self.V_1:
                    self.m.Constraint_HGS_T.add(self.m.Z_HGS_HXC_V_T_1[v,t] <= self.T_HGS_delta_max) ## Big M constraint input
                    self.m.Constraint_HGS_T.add(self.m.Z_HGS_HXC_V_T_1[v,t] >= self.T_HGS_delta_min) ## Big M constraint input
                    self.m.Constraint_HGS_T.add(self.m.Z_HGS_HXC_V_T_1[v,t] <= self.T_HGS_delta_max * self.m.B_VP_V_T_1[v,t]) ## Big M constraint input
                    self.m.Constraint_HGS_T.add(self.m.Z_HGS_HXC_V_T_1[v,t] >= self.T_HGS_delta_min * self.m.B_VP_V_T_1[v,t]) ## Big M constraint input
                    self.m.Constraint_HGS_T.add(self.m.Z_HGS_HXC_V_T_1[v,t] <= (self.m.T_HGS_HXC_T[t] - self.m.T_HGS_T[t+1]) - (1 - self.m.B_VP_V_T_1[v,t]) * self.T_HGS_delta_min) ## Big M constraint input
                    self.m.Constraint_HGS_T.add(self.m.Z_HGS_HXC_V_T_1[v,t] >= (self.m.T_HGS_HXC_T[t] - self.m.T_HGS_T[t+1]) - (1 - self.m.B_VP_V_T_1[v,t]) * self.T_HGS_delta_max) ## Big M constraint input
                
                self.m.Constraint_HGS_T.add((self.m.T_HGS_HXC_T[t] - self.m.T_HGS_T[t+1]) == sum(self.m.Z_HGS_HXC_V_T_1[v,t] for v in self.V_1))

        for t in self.T[self.TControlPeriodSwitch1:-1]:
            for v in self.V_2:
                self.m.Constraint_HGS_T.add(self.m.Z_HGS_HXC_V_T_2[v,t] <= self.T_HGS_delta_max) ## Big M constraint input
                self.m.Constraint_HGS_T.add(self.m.Z_HGS_HXC_V_T_2[v,t] >= self.T_HGS_delta_min) ## Big M constraint input
                self.m.Constraint_HGS_T.add(self.m.Z_HGS_HXC_V_T_2[v,t] <= self.T_HGS_delta_max * self.m.B_VP_V_T_2[v,t]) ## Big M constraint input
                self.m.Constraint_HGS_T.add(self.m.Z_HGS_HXC_V_T_2[v,t] >= self.T_HGS_delta_min * self.m.B_VP_V_T_2[v,t]) ## Big M constraint input
                self.m.Constraint_HGS_T.add(self.m.Z_HGS_HXC_V_T_2[v,t] <= (self.m.T_HGS_HXC_T[t] - self.m.T_HGS_T[t+1]) - (1 - self.m.B_VP_V_T_2[v,t]) * self.T_HGS_delta_min) ## Big M constraint input
                self.m.Constraint_HGS_T.add(self.m.Z_HGS_HXC_V_T_2[v,t] >= (self.m.T_HGS_HXC_T[t] - self.m.T_HGS_T[t+1]) - (1 - self.m.B_VP_V_T_2[v,t]) * self.T_HGS_delta_max) ## Big M constraint input
            
            self.m.Constraint_HGS_T.add((self.m.T_HGS_HXC_T[t] - self.m.T_HGS_T[t+1]) == sum(self.m.Z_HGS_HXC_V_T_2[v,t] for v in self.V_2))

        for t in self.T[1:]:    
            self.m.Constraint_HGS_T.add(self.m.T_HGS_T[t] <= self.T_HGS_max + self.m.S_T_HGS_T[t]) ## Temperature range tank
            self.m.Constraint_HGS_T.add(self.m.T_HGS_T[t] >= self.T_HGS_min - self.m.S_T_HGS_T[t]) ## Temperature range tank  

        ## VP_HXC_HGS_CS_RLTS
        self.m.Constraint_VP_T = pyo.ConstraintList()
        if self.TControlPeriodSwitch1 > 0:
            for t in self.T[0:self.TControlPeriodSwitch1]:
                self.m.Constraint_VP_T.add(sum(self.m.B_VP_V_T_1[v,t] for v in self.V_1) == 1) ## SOS1 constraint

                self.m.Constraint_VP_T.add(self.m.E_VP_EL_T[t] == sum(self.m.B_VP_V_T_1[v,t] for v in self.V_1[1:]) * self.e_VP_EL)

        for t in self.T[self.TControlPeriodSwitch1:-1]:
            self.m.Constraint_VP_T.add(sum(self.m.B_VP_V_T_2[v,t] for v in self.V_2) == 1) ## SOS1 constraint

            self.m.Constraint_VP_T.add(self.m.E_VP_EL_T[t] == sum(self.m.B_VP_V_T_2[v,t] for v in self.V_2[1:]) * self.e_VP_EL)
            
        if self.TControlPeriodSwitch1 > 0:
            for t in self.T[0:self.TControlPeriodSwitch1-(self.ControlPeriod1)+1]:
                if t%self.ControlPeriod1 == 0:
                    for i in range(1,self.ControlPeriod1):
                        for v in self.V_1:
                            self.m.Constraint_HP_T.add(self.m.B_VP_V_T_1[v,t] == self.m.B_VP_V_T_1[v,t+i])
        
        for t in self.T[self.TControlPeriodSwitch1:-(self.ControlPeriod2)]:
            if t%self.ControlPeriod2 == 0:
                for i in range(1,self.ControlPeriod2):
                    for v in self.V_2:
                        self.m.Constraint_HP_T.add(self.m.B_VP_V_T_2[v,t] == self.m.B_VP_V_T_2[v,t+i])

        if self.TControlPeriodSwitch1 > 0:
            for t in self.T[1:self.TControlPeriodSwitch1]:
                for v in self.V_1:
                    self.m.Constraint_VP_T.add(self.m.Z_VP_V_T_1[v,t] >= self.m.B_VP_V_T_1[v,t] - self.m.B_VP_V_T_1[v,t-1])
                    self.m.Constraint_VP_T.add(self.m.Z_VP_V_T_1[v,t] >= self.m.B_VP_V_T_1[v,t-1] - self.m.B_VP_V_T_1[v,t])
                    self.m.Constraint_VP_T.add(self.m.Z_VP_V_T_1[v,t] <= 1)
                for v in self.V_2:
                    self.m.Constraint_VP_T.add(self.m.Z_VP_V_T_2[v,t] == 0)
        
        if self.TControlPeriodSwitch1 > 0:
            for v in self.V_1:
                self.m.Constraint_VP_T.add(self.m.Z_VP_V_T_1[v,self.TControlPeriodSwitch1] >= self.m.B_VP_V_T_1[v,self.TControlPeriodSwitch1-1] - self.m.B_VP_V_T_2[v,self.TControlPeriodSwitch1])
                self.m.Constraint_VP_T.add(self.m.Z_VP_V_T_1[v,self.TControlPeriodSwitch1] >= self.m.B_VP_V_T_2[v,self.TControlPeriodSwitch1] - self.m.B_VP_V_T_1[v,self.TControlPeriodSwitch1-1])
                self.m.Constraint_VP_T.add(self.m.Z_VP_V_T_1[v,self.TControlPeriodSwitch1] <= 1)
            for v in self.V_2:
                self.m.Constraint_VP_T.add(self.m.Z_VP_V_T_2[v,self.TControlPeriodSwitch1] == 0)

        for t in self.T[self.TControlPeriodSwitch1+1:-1]:
            for v in self.V_2:
                self.m.Constraint_VP_T.add(self.m.Z_VP_V_T_2[v,t] >= self.m.B_VP_V_T_2[v,t] - self.m.B_VP_V_T_2[v,t-1])
                self.m.Constraint_VP_T.add(self.m.Z_VP_V_T_2[v,t] >= self.m.B_VP_V_T_2[v,t-1] - self.m.B_VP_V_T_2[v,t])
                self.m.Constraint_VP_T.add(self.m.Z_VP_V_T_2[v,t] <= 1)
            if self.TControlPeriodSwitch1 > 0:
                for v in self.V_1:
                    self.m.Constraint_VP_T.add(self.m.Z_VP_V_T_1[v,t] == 0)
        
        if self.Start_Toggle_Constraints == True:
            if self.TControlPeriodSwitch1 > 0:
                for v in self.V_1:
                    self.m.Constraint_VP_T.add(self.m.Z_VP_V_T_1[v,0] >= self.m.B_VP_V_T_1[v,0] - self.B_VP_start[v])
                    self.m.Constraint_VP_T.add(self.m.Z_VP_V_T_1[v,0] >= self.B_VP_start[v] - self.m.B_VP_V_T_1[v,0])
                    self.m.Constraint_VP_T.add(self.m.Z_VP_V_T_1[v,0] <= 1)
                for v in self.V_2:
                    self.m.Constraint_VP_T.add(self.m.Z_VP_V_T_2[v,0] == 0)
            if self.TControlPeriodSwitch1 == 0:
                for v in self.V_2:
                    self.m.Constraint_VP_T.add(self.m.Z_VP_V_T_2[v,0] == 0)
                for v in self.V_1:
                    self.m.Constraint_VP_T.add(self.m.Z_VP_V_T_1[v,0] == 0)
        
        if self.TControlPeriodSwitch1 > 0:
            for v in self.V_1:
                self.m.Constraint_VP_T.add(self.m.Z_VP_V_T_1[v,self.T[-1]] == 0)
        for v in self.V_2:
            self.m.Constraint_VP_T.add(self.m.Z_VP_V_T_2[v,self.T[-1]] == 0)

        ## Connection of the components -- only temperatures. Mass flows are connected directly since these are fixed values anyway and aren't used everywhere
        self.m.Constraint_Connection_T = pyo.ConstraintList()
        for t  in self.T[:-1]:
            self.m.Constraint_Connection_T.add(self.m.T_HS_T[t+1] == self.m.T_HP_HT_in_HS_T[t])
            self.m.Constraint_Connection_T.add(self.m.T_HS_HT_T[t] == self.m.T_HP_HT_T[t+1])

            self.m.Constraint_Connection_T.add(self.m.T_HXH_w_out_T[t] == self.m.T_HP_HT_in_HXH_T[t])
            self.m.Constraint_Connection_T.add(self.m.T_HXH_w_in_T[t] == self.m.T_HP_HT_T[t+1])

            self.m.Constraint_Connection_T.add(self.m.T_HGC_T[t+1] == self.m.T_HP_LT_in_HGC_T[t]) 
            self.m.Constraint_Connection_T.add(self.m.T_HGC_HP_T[t] == self.m.T_HP_LT_T[t+1])

            self.m.Constraint_Connection_T.add(self.m.T_HGC_HXC_T[t] == self.m.T_HXC_b_out_T[t])
            self.m.Constraint_Connection_T.add(self.m.T_HXC_b_in_T[t] == self.m.T_HP_LT_T[t+1])

            self.m.Constraint_Connection_T.add(self.m.T_HXA_T[t+1] == self.m.T_HGC_HXA_T[t]) 
            self.m.Constraint_Connection_T.add(self.m.T_HXA_HGC_T[t] == self.m.T_HGC_T[t+1]) 

            self.m.Constraint_Connection_T.add(self.m.T_HXA_T[t+1] == self.m.T_HXH_b_in_T[t]) 
            self.m.Constraint_Connection_T.add(self.m.T_HXA_HXH_T[t] == self.m.T_HXH_b_out_T[t]) 

            self.m.Constraint_Connection_T.add(self.m.T_HS_T[t+1] == self.m.T_IS_HT_T[t])
            self.m.Constraint_Connection_T.add(self.m.T_IS_T[t+1] == self.m.T_HS_LT_T[t])

            self.m.Constraint_Connection_T.add(self.m.T_HGS_T[t+1] == self.m.T_IS_LT_T[t])
            self.m.Constraint_Connection_T.add(self.m.T_IS_T[t+1] == self.m.T_HGS_IS_T[t])

            self.m.Constraint_Connection_T.add(self.m.T_GS_T[t+1] == self.m.T_HGS_GS_T[t])
            self.m.Constraint_Connection_T.add(self.m.T_HGS_T[t+1] == self.m.T_GS_HGS_T[t])

            self.m.Constraint_Connection_T.add(self.m.T_GS_T[t+1] == self.m.T_CS_GS_T[t])
            self.m.Constraint_Connection_T.add(self.m.T_CS_T[t+1] == self.m.T_GS_CS_T[t])

            self.m.Constraint_Connection_T.add(self.m.T_HXC_w_out_T[t] == self.m.T_HGS_HXC_T[t])
            self.m.Constraint_Connection_T.add(self.m.T_HGS_T[t+1] == self.m.T_HXC_HGS_T[t])

            self.m.Constraint_Connection_T.add(self.m.T_HXC_w_out_T[t] == self.m.T_CS_HXC_T[t])
            self.m.Constraint_Connection_T.add(self.m.T_CS_T[t+1] == self.m.T_HXC_CS_T[t])

            self.m.Constraint_Connection_T.add(self.m.T_HXC_w_out_T[t] == self.m.T_RLTS_HXC_T[t])
            self.m.Constraint_Connection_T.add(self.m.T_RLTS_T[t+1] == self.m.T_HXC_RLTS_T[t])

        return self.m

    def setObjective(self,model):
        self.m = model
        self.m.OBJ = pyo.Objective(expr = self.m.C_TOT_T_ + self.m.S_TOT_T_ + self.m.T_TOT_T_)
        return self.m

    def setWarmstart(self,model,available=True,file=None):
        self.m = model
        if available ==True:
            self.warmstart_available = True
            self.dFwarmStart = file
            ## HP
            for t in self.T[0:-1]:
                for h in self.H:
                    self.m.B_HP_H_T[h,t] = self.dFwarmStart[("B_HP_"+str(h)+"_T")][t]

            ## V_HP_HXH_HS
            for t in self.T[0:-1]:
                self.m.B_HXH_HS_T[t] = self.dFwarmStart["B_HXH_HS_T"][t]
            
            ## V_HP_HGC_HGCHXC
            for t in self.T[0:-1]:
                self.m.B_HGC_HGCHXC_T[t] = self.dFwarmStart["B_HGC_HGCHXC_T"][t]

            ## V_HXA_HXH_HGC
            for t in self.T[0:-1]:
                self.m.B_HXH_HGC_T[t] = self.dFwarmStart["B_HXH_HGC_T"][t]

            ## HXA
            for t in self.T[0:-1]:
                self.m.B_HXA_T[t] = self.dFwarmStart["B_HXA_T"][t]

            ## V_GS_HGS_CS
            for t in self.T[0:-1]:
                self.m.B_GS_HGS_T[t] = self.dFwarmStart["B_GS_HGS_T"][t]
                self.m.B_GS_CS_T[t] = self.dFwarmStart["B_GS_CS_T"][t]
                self.m.B_GS_HGS_CS_T[t] = self.dFwarmStart["B_GS_HGS_CS_T"][t]

            ## VP_HS_IS 
            for t in self.T[0:-1]:
                self.m.B_HS_IS_T[t] = self.dFwarmStart["B_HS_IS_T"][t]
            
            ## VP_IS_HGS
            for t in self.T[0:-1]:
                self.m.B_IS_HGS_T[t] = self.dFwarmStart["B_IS_HGS_T"][t]

            ## VP_HXC_HGS_CS_RLTS
            if self.TControlPeriodSwitch1 > 0:
                for t in self.T[0:self.TControlPeriodSwitch1]:
                    for v in self.V_1:
                        self.m.B_VP_V_T_1[v,t] = self.dFwarmStart[("B_VP_"+str(v)+"_T_1")][t]

            ## VP_HXC_HGS_CS_RLTS
            for t in self.T[self.TControlPeriodSwitch1:-1]:
                for v in self.V_2:
                    self.m.B_VP_V_T_2[v,t] = self.dFwarmStart[("B_VP_"+str(v)+"_T_2")][t]
        else:
            self.warmstart_available = False
        
        return self.m


    def setSolverAndRunOptimization(self,solver = 0, showSolverOutput = 0):
        if solver == 0:
            self.opt = pyo.SolverFactory('gurobi', solver_io="python")
            self.opt.options['TimeLimit'] = 200
            self.opt.options['threads'] = 8
            #self.opt.options['ObjBound'] = 50
            #self.opt.options['Cutoff'] = 500
            #self.opt.options['resultFile'] = 'test.ilp'
        elif solver == 1:
            self.opt = pyo.SolverFactory('cbc')
            self.opt.options['Sec'] = 200
        elif solver == 2:
            self.opt = pyo.SolverFactory('glpk')
            self.opt.options['tmlim'] = 200
            #self.opt.options['mipgap'] = 1e-6 # not needed atm
        
        self.results = self.opt.solve(self.m,warmstart=self.warmstart_available,tee=True) 
        if showSolverOutput == 1:
            print(self.results)

    def getResults(self,model,source=None,savePath="",singleFile=False):
        self.m = model
        self.safeFile = pd.DataFrame(columns = ["C_OP_T","C_HP_T","C_HXA_T","C_IS_T","C_GS_T","C_VP_T","B_HP_4_T","B_HP_3_T","B_HP_2_T","B_HP_1_T","B_HP_0_T","B_HXH_HS_T","B_HGC_HGCHXC_T","B_HXA_T","B_HXH_HGC_T",
        "B_HS_IS_T","B_IS_HGS_T","B_GS_HGS_T","B_GS_CS_T","B_GS_HGS_CS_T","B_VP_7_T_1","B_VP_6_T_1","B_VP_5_T_1","B_VP_4_T_1","B_VP_3_T_1","B_VP_2_T_1","B_VP_1_T_1","B_VP_0_T_1","B_VP_13_T_2","B_VP_12_T_2","B_VP_11_T_2",
        "B_VP_10_T_2","B_VP_9_T_2","B_VP_8_T_2","B_VP_7_T_2","B_VP_6_T_2","B_VP_5_T_2","B_VP_4_T_2","B_VP_3_T_2","B_VP_2_T_2","B_VP_1_T_2","B_VP_0_T_2","E_HP_EL_T","Q_HP_HT_T","Q_HP_LT_T","E_HP_EL_in_T",
        "T_HP_HT_T","T_HP_LT_T","T_HS_T","T_HXH_T","T_HGC_T","T_HXA_T","T_HXC_T","T_IS_T","T_ISw_T","T_ISc_T","T_IS_W_0_T","T_IS_W_1_T","T_IS_W_2_T","T_IS_C_0_T","T_IS_C_1_T","T_IS_C_2_T","T_IS_C_3_T","T_IS_C_4_T",
        "T_GS_T","T_GSw_T","T_GSc_T","T_GS_W_0_T","T_GS_W_1_T","T_GS_W_2_T","T_GS_C_0_T","T_GS_C_1_T","T_GS_C_2_T","T_GS_C_3_T","T_GS_C_4_T","T_GS_C_5_T","T_GS_C_6_T",
        "T_HGS_T","T_CS_T","T_RLTS_T","S_OP_T","S_T_HP_T","S_T_HS_T","S_T_HXH_T","S_T_HGC_T","S_T_HXA_T",
        "S_T_HXC_T","S_T_IS_T","S_T_GS_T","S_T_HGS_T","S_T_CS_T","S_T_RLTS_T","q_dem_HS_T","q_dem_CS_T","q_dem_RLTS_T","temp_amb_T"]) 
        try: # Everything but slack constraints
            if self.TControlPeriodSwitch1 > 0:
                self.safeFile = self.safeFile.append({"C_OP_T":self.m.C_OP_T[0](),"C_HP_T":self.m.E_HP_EL_in_T[0]()*self.c_ELECTRICITY_buy_T[0]*self.StepSizeInSec/self.t_hour_in_sec,
                "C_HXA_T":self.m.E_HXA_EL_T[0]()*self.c_ELECTRICITY_buy_T[0]*self.StepSizeInSec/self.t_hour_in_sec,"C_IS_T":self.m.E_IS_EL_T[0]()*self.c_ELECTRICITY_buy_T[0]*self.StepSizeInSec/self.t_hour_in_sec,
                "C_GS_T":self.m.E_GS_EL_T[0]()*self.c_ELECTRICITY_buy_T[0]*self.StepSizeInSec/self.t_hour_in_sec,"C_VP_T":self.m.E_VP_EL_T[0]()*self.c_ELECTRICITY_buy_T[0]*self.StepSizeInSec/self.t_hour_in_sec,
                "B_HP_4_T":self.m.B_HP_H_T[4,0](),"B_HP_3_T":self.m.B_HP_H_T[3,0](),"B_HP_2_T":self.m.B_HP_H_T[2,0](),"B_HP_1_T":self.m.B_HP_H_T[1,0](),"B_HP_0_T":self.m.B_HP_H_T[0,0](),
                "B_HXH_HS_T":self.m.B_HXH_HS_T[0](),"B_HGC_HGCHXC_T":self.m.B_HGC_HGCHXC_T[0](),"B_HXA_T":self.m.B_HXA_T[0](),"B_HXH_HGC_T":self.m.B_HXH_HGC_T[0](),"B_HS_IS_T":self.m.B_HS_IS_T[0](),"B_IS_HGS_T":self.m.B_IS_HGS_T[0](),
                "B_GS_HGS_T":self.m.B_GS_HGS_T[0](),"B_GS_CS_T":self.m.B_GS_CS_T[0](),"B_GS_HGS_CS_T":self.m.B_GS_HGS_CS_T[0](),"B_VP_7_T_1":self.m.B_VP_V_T_1[7,0](),"B_VP_6_T_1":self.m.B_VP_V_T_1[6,0](),
                "B_VP_5_T_1":self.m.B_VP_V_T_1[5,0](),"B_VP_4_T_1":self.m.B_VP_V_T_1[4,0](),"B_VP_3_T_1":self.m.B_VP_V_T_1[3,0](),"B_VP_2_T_1":self.m.B_VP_V_T_1[2,0](),"B_VP_1_T_1":self.m.B_VP_V_T_1[1,0](),"B_VP_0_T_1":self.m.B_VP_V_T_1[0,0](),
                "E_HP_EL_T":self.m.E_HP_EL_T[0](),"Q_HP_HT_T":self.m.Q_HP_HT_T[0](),"Q_HP_LT_T":self.m.Q_HP_LT_T[0](),"E_HP_EL_in_T":self.m.E_HP_EL_in_T[0](),"T_HP_HT_T":self.m.T_HP_HT_T[0](),
                "T_HP_LT_T":self.m.T_HP_LT_T[0](),"T_HS_T":self.m.T_HS_T[0](),"T_HXH_T":self.m.T_HXH_T[0](),"T_HGC_T":self.m.T_HGC_T[0](),
                "T_HXA_T":self.m.T_HXA_T[0](),"T_HXC_T":self.m.T_HXC_T[0](),"T_IS_T":self.m.T_IS_T[0](),"T_ISw_T":(sum(self.m.T_IS_W_T_WR[0,r]() for r in self.wr_IS)/len(self.wr_IS)),"T_ISc_T":(sum(self.m.T_IS_C_T_CR[0,r]() for r in self.cr_IS)/len(self.cr_IS)),
                "T_IS_W_0_T":self.m.T_IS_W_T_WR[0,0](),"T_IS_W_1_T":self.m.T_IS_W_T_WR[0,2](),"T_IS_W_2_T":self.m.T_IS_W_T_WR[0,4](),"T_IS_C_0_T":self.m.T_IS_C_T_CR[0,0](),"T_IS_C_1_T":self.m.T_IS_C_T_CR[0,1](),"T_IS_C_2_T":self.m.T_IS_C_T_CR[0,2](),"T_IS_C_3_T":self.m.T_IS_C_T_CR[0,3](),"T_IS_C_4_T":self.m.T_IS_C_T_CR[0,4](),
                "T_GS_T":self.m.T_GS_T[0](),"T_GSw_T":(sum(sum(self.m.T_GS_W_T_WR_WC[0,c,r]() for c in self.wc_GS) for r in self.wr_GS)/(len(self.wc_GS)*len(self.wr_GS))),"T_GSc_T":(sum(sum(self.m.T_GS_C_T_CR_CC[0,c,r]() for c in self.cc_GS) for r in self.cr_GS)/(len(self.cc_GS)*len(self.cr_GS))),
                "T_GS_W_0_T":sum(self.m.T_GS_W_T_WR_WC[0,c,1]() for c in self.wc_GS),"T_GS_W_1_T":sum(self.m.T_GS_W_T_WR_WC[0,c,3]() for c in self.wc_GS),"T_GS_W_2_T":sum(self.m.T_GS_W_T_WR_WC[0,c,5]() for c in self.wc_GS),
                "T_GS_C_0_T":sum(self.m.T_GS_C_T_CR_CC[0,c,0]() for c in self.cc_GS),"T_GS_C_1_T":sum(self.m.T_GS_C_T_CR_CC[0,c,1]() for c in self.cc_GS),"T_GS_C_2_T":sum(self.m.T_GS_C_T_CR_CC[0,c,2]() for c in self.cc_GS),"T_GS_C_3_T":sum(self.m.T_GS_C_T_CR_CC[0,c,3]() for c in self.cc_GS),"T_GS_C_4_T":sum(self.m.T_GS_C_T_CR_CC[0,c,4]() for c in self.cc_GS),"T_GS_C_5_T":sum(self.m.T_GS_C_T_CR_CC[0,c,5]() for c in self.cc_GS),"T_GS_C_6_T":sum(self.m.T_GS_C_T_CR_CC[0,c,6]() for c in self.cc_GS),
                "T_HGS_T":self.m.T_HGS_T[0](),"T_CS_T":self.m.T_CS_T[0](),"T_RLTS_T":self.m.T_RLTS_T[0](),"q_dem_HS_T":self.q_dem_HS_T[0],"q_dem_CS_T":self.q_dem_CS_T[0],"q_dem_RLTS_T":self.q_dem_RLTS_T[0],"temp_amb_T":self.temp_amb_T[0]}, ignore_index=True)
                # ALL
                for t in self.T[1:self.TControlPeriodSwitch1]:
                    self.safeFile = self.safeFile.append({"C_OP_T":self.m.C_OP_T[t](),"C_HP_T":self.m.E_HP_EL_in_T[t]()*self.c_ELECTRICITY_buy_T[t]*self.StepSizeInSec/self.t_hour_in_sec,
                    "C_HXA_T":self.m.E_HXA_EL_T[t]()*self.c_ELECTRICITY_buy_T[t]*self.StepSizeInSec/self.t_hour_in_sec,"C_IS_T":self.m.E_IS_EL_T[t]()*self.c_ELECTRICITY_buy_T[t]*self.StepSizeInSec/self.t_hour_in_sec,
                    "C_GS_T":self.m.E_GS_EL_T[t]()*self.c_ELECTRICITY_buy_T[t]*self.StepSizeInSec/self.t_hour_in_sec,"C_VP_T":self.m.E_VP_EL_T[t]()*self.c_ELECTRICITY_buy_T[t]*self.StepSizeInSec/self.t_hour_in_sec,
                    "B_HP_4_T":self.m.B_HP_H_T[4,t](),"B_HP_3_T":self.m.B_HP_H_T[3,t](),"B_HP_2_T":self.m.B_HP_H_T[2,t](),"B_HP_1_T":self.m.B_HP_H_T[1,t](),"B_HP_0_T":self.m.B_HP_H_T[0,t](),
                    "B_HXH_HS_T":self.m.B_HXH_HS_T[t](),"B_HGC_HGCHXC_T":self.m.B_HGC_HGCHXC_T[t](),"B_HXA_T":self.m.B_HXA_T[t](),"B_HXH_HGC_T":self.m.B_HXH_HGC_T[t](),"B_HS_IS_T":self.m.B_HS_IS_T[t](),"B_IS_HGS_T":self.m.B_IS_HGS_T[t](),
                    "B_GS_HGS_T":self.m.B_GS_HGS_T[t](),"B_GS_CS_T":self.m.B_GS_CS_T[t](),"B_GS_HGS_CS_T":self.m.B_GS_HGS_CS_T[t](),"B_VP_7_T_1":self.m.B_VP_V_T_1[7,t](),"B_VP_6_T_1":self.m.B_VP_V_T_1[6,t](),
                    "B_VP_5_T_1":self.m.B_VP_V_T_1[5,t](),"B_VP_4_T_1":self.m.B_VP_V_T_1[4,t](),"B_VP_3_T_1":self.m.B_VP_V_T_1[3,t](),"B_VP_2_T_1":self.m.B_VP_V_T_1[2,t](),"B_VP_1_T_1":self.m.B_VP_V_T_1[1,t](),"B_VP_0_T_1":self.m.B_VP_V_T_1[0,t](),
                    "E_HP_EL_T":self.m.E_HP_EL_T[t](),"Q_HP_HT_T":self.m.Q_HP_HT_T[t](),"Q_HP_LT_T":self.m.Q_HP_LT_T[t](),"E_HP_EL_in_T":self.m.E_HP_EL_in_T[t](),"T_HP_HT_T":self.m.T_HP_HT_T[t](),
                    "T_HP_LT_T":self.m.T_HP_LT_T[t](),"T_HS_T":self.m.T_HS_T[t](),"T_HXH_T":self.m.T_HXH_T[t](),"T_HGC_T":self.m.T_HGC_T[t](),
                    "T_HXA_T":self.m.T_HXA_T[t](),"T_HXC_T":self.m.T_HXC_T[t](),"T_IS_T":self.m.T_IS_T[t](),"T_ISw_T":(sum(self.m.T_IS_W_T_WR[t,r]() for r in self.wr_IS)/len(self.wr_IS)),"T_ISc_T":(sum(self.m.T_IS_C_T_CR[t,r]() for r in self.cr_IS)/len(self.cr_IS)),
                    "T_IS_W_0_T":self.m.T_IS_W_T_WR[t,0](),"T_IS_W_1_T":self.m.T_IS_W_T_WR[t,2](),"T_IS_W_2_T":self.m.T_IS_W_T_WR[t,4](),"T_IS_C_0_T":self.m.T_IS_C_T_CR[t,0](),"T_IS_C_1_T":self.m.T_IS_C_T_CR[t,1](),"T_IS_C_2_T":self.m.T_IS_C_T_CR[t,2](),"T_IS_C_3_T":self.m.T_IS_C_T_CR[t,3](),"T_IS_C_4_T":self.m.T_IS_C_T_CR[t,4](),
                    "T_GS_T":self.m.T_GS_T[t](),"T_GSw_T":(sum(sum(self.m.T_GS_W_T_WR_WC[t,c,r]() for c in self.wc_GS) for r in self.wr_GS)/(len(self.wc_GS)*len(self.wr_GS))),"T_GSc_T":(sum(sum(self.m.T_GS_C_T_CR_CC[t,c,r]() for c in self.cc_GS) for r in self.cr_GS)/(len(self.cc_GS)*len(self.cr_GS))),
                    "T_GS_W_0_T":sum(self.m.T_GS_W_T_WR_WC[t,c,1]() for c in self.wc_GS),"T_GS_W_1_T":sum(self.m.T_GS_W_T_WR_WC[t,c,3]() for c in self.wc_GS),"T_GS_W_2_T":sum(self.m.T_GS_W_T_WR_WC[t,c,5]() for c in self.wc_GS),
                    "T_GS_C_0_T":sum(self.m.T_GS_C_T_CR_CC[t,c,0]() for c in self.cc_GS),"T_GS_C_1_T":sum(self.m.T_GS_C_T_CR_CC[t,c,1]() for c in self.cc_GS),"T_GS_C_2_T":sum(self.m.T_GS_C_T_CR_CC[t,c,2]() for c in self.cc_GS),"T_GS_C_3_T":sum(self.m.T_GS_C_T_CR_CC[t,c,3]() for c in self.cc_GS),"T_GS_C_4_T":sum(self.m.T_GS_C_T_CR_CC[t,c,4]() for c in self.cc_GS),"T_GS_C_5_T":sum(self.m.T_GS_C_T_CR_CC[t,c,5]() for c in self.cc_GS),"T_GS_C_6_T":sum(self.m.T_GS_C_T_CR_CC[t,c,6]() for c in self.cc_GS),
                    "T_HGS_T":self.m.T_HGS_T[t](),"T_CS_T":self.m.T_CS_T[t](),"T_RLTS_T":self.m.T_RLTS_T[t](),"S_OP_T":self.m.S_OP_T[t](),"S_T_HP_T":self.m.S_T_HP_T[t](),"S_T_HS_T":self.m.S_T_HS_T[t](),"S_T_HXH_T":self.m.S_T_HXH_T[t](),"S_T_HGC_T":self.m.S_T_HGC_T[t](),
                    "S_T_HXA_T":self.m.S_T_HXA_T[t](),"S_T_HXC_T":self.m.S_T_HXC_T[t](),"S_T_IS_T":sum(self.m.S_T_IS_W_T_WR[t,r]() for r in self.wr_IS)+sum(self.m.S_T_IS_C_T_CR[t,r]() for r in self.cr_IS),"S_T_GS_T":sum(sum(self.m.S_T_GS_W_T_WR_WC[t,c,r]() for r in self.wr_GS) for c in self.wc_GS)+sum(sum(self.m.S_T_GS_C_T_CR_CC[t,c,r]() for r in self.cr_GS)for c in self.cc_GS),
                    "S_T_HGS_T":self.m.S_T_HGS_T[t](),"S_T_CS_T":self.m.S_T_CS_T[t](),"S_T_RLTS_T":self.m.S_T_RLTS_T[t](),"q_dem_HS_T":self.q_dem_HS_T[t],"q_dem_CS_T":self.q_dem_CS_T[t],"q_dem_RLTS_T":self.q_dem_RLTS_T[t],"temp_amb_T":self.temp_amb_T[t]}, ignore_index=True)
                # ALL
                for t in self.T[self.TControlPeriodSwitch1:-1]:
                    self.safeFile = self.safeFile.append({"C_OP_T":self.m.C_OP_T[t](),"C_HP_T":self.m.E_HP_EL_in_T[t]()*self.c_ELECTRICITY_buy_T[t]*self.StepSizeInSec/self.t_hour_in_sec,
                    "C_HXA_T":self.m.E_HXA_EL_T[t]()*self.c_ELECTRICITY_buy_T[t]*self.StepSizeInSec/self.t_hour_in_sec,"C_IS_T":self.m.E_IS_EL_T[t]()*self.c_ELECTRICITY_buy_T[t]*self.StepSizeInSec/self.t_hour_in_sec,
                    "C_GS_T":self.m.E_GS_EL_T[t]()*self.c_ELECTRICITY_buy_T[t]*self.StepSizeInSec/self.t_hour_in_sec,"C_VP_T":self.m.E_VP_EL_T[t]()*self.c_ELECTRICITY_buy_T[t]*self.StepSizeInSec/self.t_hour_in_sec,
                    "B_HP_4_T":self.m.B_HP_H_T[4,t](),"B_HP_3_T":self.m.B_HP_H_T[3,t](),"B_HP_2_T":self.m.B_HP_H_T[2,t](),"B_HP_1_T":self.m.B_HP_H_T[1,t](),"B_HP_0_T":self.m.B_HP_H_T[0,t](),
                    "B_HXH_HS_T":self.m.B_HXH_HS_T[t](),"B_HGC_HGCHXC_T":self.m.B_HGC_HGCHXC_T[t](),"B_HXA_T":self.m.B_HXA_T[t](),"B_HXH_HGC_T":self.m.B_HXH_HGC_T[t](),"B_HS_IS_T":self.m.B_HS_IS_T[t](),"B_IS_HGS_T":self.m.B_IS_HGS_T[t](),
                    "B_GS_HGS_T":self.m.B_GS_HGS_T[t](),"B_GS_CS_T":self.m.B_GS_CS_T[t](),"B_GS_HGS_CS_T":self.m.B_GS_HGS_CS_T[t](),"B_VP_13_T_2":self.m.B_VP_V_T_2[13,t](),"B_VP_12_T_2":self.m.B_VP_V_T_2[12,t](),"B_VP_11_T_2":self.m.B_VP_V_T_2[11,t](),
                    "B_VP_10_T_2":self.m.B_VP_V_T_2[10,t](),"B_VP_9_T_2":self.m.B_VP_V_T_2[9,t](),"B_VP_8_T_2":self.m.B_VP_V_T_2[8,t](),"B_VP_7_T_2":self.m.B_VP_V_T_2[7,t](),"B_VP_6_T_2":self.m.B_VP_V_T_2[6,t](),
                    "B_VP_5_T_2":self.m.B_VP_V_T_2[5,t](),"B_VP_4_T_2":self.m.B_VP_V_T_2[4,t](),"B_VP_3_T_2":self.m.B_VP_V_T_2[3,t](),"B_VP_2_T_2":self.m.B_VP_V_T_2[2,t](),"B_VP_1_T_2":self.m.B_VP_V_T_2[1,t](),"B_VP_0_T_2":self.m.B_VP_V_T_2[0,t](),
                    "E_HP_EL_T":self.m.E_HP_EL_T[t](),"Q_HP_HT_T":self.m.Q_HP_HT_T[t](),"Q_HP_LT_T":self.m.Q_HP_LT_T[t](),"E_HP_EL_in_T":self.m.E_HP_EL_in_T[t](),"T_HP_HT_T":self.m.T_HP_HT_T[t](),
                    "T_HP_LT_T":self.m.T_HP_LT_T[t](),"T_HS_T":self.m.T_HS_T[t](),"T_HXH_T":self.m.T_HXH_T[t](),"T_HGC_T":self.m.T_HGC_T[t](),
                    "T_HXA_T":self.m.T_HXA_T[t](),"T_HXC_T":self.m.T_HXC_T[t](),"T_IS_T":self.m.T_IS_T[t](),"T_ISw_T":(sum(self.m.T_IS_W_T_WR[t,r]() for r in self.wr_IS)/len(self.wr_IS)),"T_ISc_T":(sum(self.m.T_IS_C_T_CR[t,r]() for r in self.cr_IS)/len(self.cr_IS)),
                    "T_IS_W_0_T":self.m.T_IS_W_T_WR[t,0](),"T_IS_W_1_T":self.m.T_IS_W_T_WR[t,2](),"T_IS_W_2_T":self.m.T_IS_W_T_WR[t,4](),"T_IS_C_0_T":self.m.T_IS_C_T_CR[t,0](),"T_IS_C_1_T":self.m.T_IS_C_T_CR[t,1](),"T_IS_C_2_T":self.m.T_IS_C_T_CR[t,2](),"T_IS_C_3_T":self.m.T_IS_C_T_CR[t,3](),"T_IS_C_4_T":self.m.T_IS_C_T_CR[t,4](),   
                    "T_GS_T":self.m.T_GS_T[t](),"T_GSw_T":(sum(sum(self.m.T_GS_W_T_WR_WC[t,c,r]() for c in self.wc_GS) for r in self.wr_GS)/(len(self.wc_GS)*len(self.wr_GS))),"T_GSc_T":(sum(sum(self.m.T_GS_C_T_CR_CC[t,c,r]() for c in self.cc_GS) for r in self.cr_GS)/(len(self.cc_GS)*len(self.cr_GS))),
                    "T_GS_W_0_T":sum(self.m.T_GS_W_T_WR_WC[t,c,1]() for c in self.wc_GS),"T_GS_W_1_T":sum(self.m.T_GS_W_T_WR_WC[t,c,3]() for c in self.wc_GS),"T_GS_W_2_T":sum(self.m.T_GS_W_T_WR_WC[t,c,5]() for c in self.wc_GS),
                    "T_GS_C_0_T":sum(self.m.T_GS_C_T_CR_CC[t,c,0]() for c in self.cc_GS),"T_GS_C_1_T":sum(self.m.T_GS_C_T_CR_CC[t,c,1]() for c in self.cc_GS),"T_GS_C_2_T":sum(self.m.T_GS_C_T_CR_CC[t,c,2]() for c in self.cc_GS),"T_GS_C_3_T":sum(self.m.T_GS_C_T_CR_CC[t,c,3]() for c in self.cc_GS),"T_GS_C_4_T":sum(self.m.T_GS_C_T_CR_CC[t,c,4]() for c in self.cc_GS),"T_GS_C_5_T":sum(self.m.T_GS_C_T_CR_CC[t,c,5]() for c in self.cc_GS),"T_GS_C_6_T":sum(self.m.T_GS_C_T_CR_CC[t,c,6]() for c in self.cc_GS),
                    "T_HGS_T":self.m.T_HGS_T[t](),"T_CS_T":self.m.T_CS_T[t](),"T_RLTS_T":self.m.T_RLTS_T[t](),"S_OP_T":self.m.S_OP_T[t](),"S_T_HP_T":self.m.S_T_HP_T[t](),"S_T_HS_T":self.m.S_T_HS_T[t](),"S_T_HXH_T":self.m.S_T_HXH_T[t](),"S_T_HGC_T":self.m.S_T_HGC_T[t](),
                    "S_T_HXA_T":self.m.S_T_HXA_T[t](),"S_T_HXC_T":self.m.S_T_HXC_T[t](),"S_T_IS_T":sum(self.m.S_T_IS_W_T_WR[t,r]() for r in self.wr_IS)+sum(self.m.S_T_IS_C_T_CR[t,r]() for r in self.cr_IS),"S_T_GS_T":sum(sum(self.m.S_T_GS_W_T_WR_WC[t,c,r]() for r in self.wr_GS) for c in self.wc_GS)+sum(sum(self.m.S_T_GS_C_T_CR_CC[t,c,r]() for r in self.cr_GS)for c in self.cc_GS),
                    "S_T_HGS_T":self.m.S_T_HGS_T[t](),"S_T_CS_T":self.m.S_T_CS_T[t](),"S_T_RLTS_T":self.m.S_T_RLTS_T[t](),"q_dem_HS_T":self.q_dem_HS_T[t],"q_dem_CS_T":self.q_dem_CS_T[t],"q_dem_RLTS_T":self.q_dem_RLTS_T[t],"temp_amb_T":self.temp_amb_T[t]}, ignore_index=True)
                # Storage temperatures and slacks
                self.safeFile = self.safeFile.append({"T_HP_HT_T":self.m.T_HP_HT_T[self.T[-1]](),"T_HP_LT_T":self.m.T_HP_LT_T[self.T[-1]](),"T_HS_T":self.m.T_HS_T[self.T[-1]](),"T_HXH_T":self.m.T_HXH_T[self.T[-1]](),"T_HGC_T":self.m.T_HGC_T[self.T[-1]](),
                "T_HXA_T":self.m.T_HXA_T[self.T[-1]](),"T_HXC_T":self.m.T_HXC_T[self.T[-1]](),"T_IS_T":self.m.T_IS_T[self.T[-1]](),"T_ISw_T":(sum(self.m.T_IS_W_T_WR[self.T[-1],r]() for r in self.wr_IS)/len(self.wr_IS)),"T_ISc_T":(sum(self.m.T_IS_C_T_CR[self.T[-1],r]() for r in self.cr_IS)/len(self.cr_IS)),
                "T_IS_W_0_T":self.m.T_IS_W_T_WR[self.T[-1],0](),"T_IS_W_1_T":self.m.T_IS_W_T_WR[self.T[-1],2](),"T_IS_W_2_T":self.m.T_IS_W_T_WR[self.T[-1],4](),"T_IS_C_0_T":self.m.T_IS_C_T_CR[self.T[-1],0](),"T_IS_C_1_T":self.m.T_IS_C_T_CR[self.T[-1],1](),"T_IS_C_2_T":self.m.T_IS_C_T_CR[self.T[-1],2](),"T_IS_C_3_T":self.m.T_IS_C_T_CR[self.T[-1],3](),"T_IS_C_4_T":self.m.T_IS_C_T_CR[self.T[-1],4](),   
                "T_GS_T":self.m.T_GS_T[self.T[-1]](),"T_GSw_T":(sum(sum(self.m.T_GS_W_T_WR_WC[self.T[-1],c,r]() for c in self.wc_GS) for r in self.wr_GS)/(len(self.wc_GS)*len(self.wr_GS))),"T_GSc_T":(sum(sum(self.m.T_GS_C_T_CR_CC[self.T[-1],c,r]() for c in self.cc_GS) for r in self.cr_GS)/(len(self.cc_GS)*len(self.cr_GS))),
                "T_GS_W_0_T":sum(self.m.T_GS_W_T_WR_WC[self.T[-1],c,1]() for c in self.wc_GS),"T_GS_W_1_T":sum(self.m.T_GS_W_T_WR_WC[self.T[-1],c,3]() for c in self.wc_GS),"T_GS_W_2_T":sum(self.m.T_GS_W_T_WR_WC[self.T[-1],c,5]() for c in self.wc_GS),
                "T_GS_C_0_T":sum(self.m.T_GS_C_T_CR_CC[self.T[-1],c,0]() for c in self.cc_GS),"T_GS_C_1_T":sum(self.m.T_GS_C_T_CR_CC[self.T[-1],c,1]() for c in self.cc_GS),"T_GS_C_2_T":sum(self.m.T_GS_C_T_CR_CC[self.T[-1],c,2]() for c in self.cc_GS),"T_GS_C_3_T":sum(self.m.T_GS_C_T_CR_CC[self.T[-1],c,3]() for c in self.cc_GS),"T_GS_C_4_T":sum(self.m.T_GS_C_T_CR_CC[self.T[-1],c,4]() for c in self.cc_GS),"T_GS_C_5_T":sum(self.m.T_GS_C_T_CR_CC[self.T[-1],c,5]() for c in self.cc_GS),"T_GS_C_6_T":sum(self.m.T_GS_C_T_CR_CC[self.T[-1],c,6]() for c in self.cc_GS),
                "T_HGS_T":self.m.T_HGS_T[self.T[-1]](),"T_CS_T":self.m.T_CS_T[self.T[-1]](),
                "T_RLTS_T":self.m.T_RLTS_T[self.T[-1]](),"S_OP_T":self.m.S_OP_T[self.T[-1]](),"S_T_HP_T":self.m.S_T_HP_T[self.T[-1]](),"S_T_HS_T":self.m.S_T_HS_T[self.T[-1]](),"S_T_HGC_T":self.m.S_T_HGC_T[self.T[-1]](),
                "S_T_HXA_T":self.m.S_T_HXA_T[self.T[-1]](),"S_T_IS_T":sum(self.m.S_T_IS_W_T_WR[self.T[-1],r]() for r in self.wr_IS)+sum(self.m.S_T_IS_C_T_CR[self.T[-1],r]() for r in self.cr_IS),"S_T_GS_T":sum(sum(self.m.S_T_GS_W_T_WR_WC[self.T[-1],c,r]() for r in self.wr_GS) for c in self.wc_GS)+sum(sum(self.m.S_T_GS_C_T_CR_CC[self.T[-1],c,r]() for r in self.cr_GS)for c in self.cc_GS),
                "S_T_HGS_T":self.m.S_T_HGS_T[self.T[-1]](),"S_T_CS_T":self.m.S_T_CS_T[self.T[-1]](),"S_T_RLTS_T":self.m.S_T_RLTS_T[self.T[-1]]()}, ignore_index=True)
            else: # Everything but slack constraints
                self.safeFile = self.safeFile.append({"C_OP_T":self.m.C_OP_T[0](),"C_HP_T":self.m.E_HP_EL_in_T[0]()*self.c_ELECTRICITY_buy_T[0]*self.StepSizeInSec/self.t_hour_in_sec,
                "C_HXA_T":self.m.E_HXA_EL_T[0]()*self.c_ELECTRICITY_buy_T[0]*self.StepSizeInSec/self.t_hour_in_sec,"C_IS_T":self.m.E_IS_EL_T[0]()*self.c_ELECTRICITY_buy_T[0]*self.StepSizeInSec/self.t_hour_in_sec,
                "C_GS_T":self.m.E_GS_EL_T[0]()*self.c_ELECTRICITY_buy_T[0]*self.StepSizeInSec/self.t_hour_in_sec,"C_VP_T":self.m.E_VP_EL_T[0]()*self.c_ELECTRICITY_buy_T[0]*self.StepSizeInSec/self.t_hour_in_sec,
                "B_HP_4_T":self.m.B_HP_H_T[4,0](),"B_HP_3_T":self.m.B_HP_H_T[3,0](),"B_HP_2_T":self.m.B_HP_H_T[2,0](),"B_HP_1_T":self.m.B_HP_H_T[1,0](),"B_HP_0_T":self.m.B_HP_H_T[0,0](),
                "B_HXH_HS_T":self.m.B_HXH_HS_T[0](),"B_HGC_HGCHXC_T":self.m.B_HGC_HGCHXC_T[0](),"B_HXA_T":self.m.B_HXA_T[0](),"B_HXH_HGC_T":self.m.B_HXH_HGC_T[0](),"B_HS_IS_T":self.m.B_HS_IS_T[0](),"B_IS_HGS_T":self.m.B_IS_HGS_T[0](),
                "B_GS_HGS_T":self.m.B_GS_HGS_T[0](),"B_GS_CS_T":self.m.B_GS_CS_T[0](),"B_GS_HGS_CS_T":self.m.B_GS_HGS_CS_T[0](),"B_VP_13_T_2":self.m.B_VP_V_T_2[13,0](),"B_VP_12_T_2":self.m.B_VP_V_T_2[12,0](),"B_VP_11_T_2":self.m.B_VP_V_T_2[11,0](),
                "B_VP_10_T_2":self.m.B_VP_V_T_2[10,0](),"B_VP_9_T_2":self.m.B_VP_V_T_2[9,0](),"B_VP_8_T_2":self.m.B_VP_V_T_2[8,0](),"B_VP_7_T_2":self.m.B_VP_V_T_2[7,0](),"B_VP_6_T_2":self.m.B_VP_V_T_2[6,0](),
                "B_VP_5_T_2":self.m.B_VP_V_T_2[5,0](),"B_VP_4_T_2":self.m.B_VP_V_T_2[4,0](),"B_VP_3_T_2":self.m.B_VP_V_T_2[3,0](),"B_VP_2_T_2":self.m.B_VP_V_T_2[2,0](),"B_VP_1_T_2":self.m.B_VP_V_T_2[1,0](),"B_VP_0_T_2":self.m.B_VP_V_T_2[0,0](),
                "E_HP_EL_T":self.m.E_HP_EL_T[0](),"Q_HP_HT_T":self.m.Q_HP_HT_T[0](),"Q_HP_LT_T":self.m.Q_HP_LT_T[0](),"E_HP_EL_in_T":self.m.E_HP_EL_in_T[0](),"T_HP_HT_T":self.m.T_HP_HT_T[0](),
                "T_HP_LT_T":self.m.T_HP_LT_T[0](),"T_HS_T":self.m.T_HS_T[0](),"T_HXH_T":self.m.T_HXH_T[0](),"T_HGC_T":self.m.T_HGC_T[0](),
                "T_HXA_T":self.m.T_HXA_T[0](),"T_HXC_T":self.m.T_HXC_T[0](),"T_IS_T":self.m.T_IS_T[0](),"T_ISw_T":(sum(self.m.T_IS_W_T_WR[0,r]() for r in self.wr_IS)/len(self.wr_IS)),"T_ISc_T":(sum(self.m.T_IS_C_T_CR[0,r]() for r in self.cr_IS)/len(self.cr_IS)),
                "T_IS_W_0_T":self.m.T_IS_W_T_WR[0,0](),"T_IS_W_1_T":self.m.T_IS_W_T_WR[0,2](),"T_IS_W_2_T":self.m.T_IS_W_T_WR[0,4](),"T_IS_C_0_T":self.m.T_IS_C_T_CR[0,0](),"T_IS_C_1_T":self.m.T_IS_C_T_CR[0,1](),"T_IS_C_2_T":self.m.T_IS_C_T_CR[0,2](),"T_IS_C_3_T":self.m.T_IS_C_T_CR[0,3](),"T_IS_C_4_T":self.m.T_IS_C_T_CR[0,4](),
                "T_GS_T":self.m.T_GS_T[0](),"T_GSw_T":(sum(sum(self.m.T_GS_W_T_WR_WC[0,c,r]() for c in self.wc_GS) for r in self.wr_GS)/(len(self.wc_GS)*len(self.wr_GS))),"T_GSc_T":(sum(sum(self.m.T_GS_C_T_CR_CC[0,c,r]() for c in self.cc_GS) for r in self.cr_GS)/(len(self.cc_GS)*len(self.cr_GS))),
                "T_GS_W_0_T":sum(self.m.T_GS_W_T_WR_WC[0,c,1]() for c in self.wc_GS),"T_GS_W_1_T":sum(self.m.T_GS_W_T_WR_WC[0,c,3]() for c in self.wc_GS),"T_GS_W_2_T":sum(self.m.T_GS_W_T_WR_WC[0,c,5]() for c in self.wc_GS),
                "T_GS_C_0_T":sum(self.m.T_GS_C_T_CR_CC[0,c,0]() for c in self.cc_GS),"T_GS_C_1_T":sum(self.m.T_GS_C_T_CR_CC[0,c,1]() for c in self.cc_GS),"T_GS_C_2_T":sum(self.m.T_GS_C_T_CR_CC[0,c,2]() for c in self.cc_GS),"T_GS_C_3_T":sum(self.m.T_GS_C_T_CR_CC[0,c,3]() for c in self.cc_GS),"T_GS_C_4_T":sum(self.m.T_GS_C_T_CR_CC[0,c,4]() for c in self.cc_GS),"T_GS_C_5_T":sum(self.m.T_GS_C_T_CR_CC[0,c,5]() for c in self.cc_GS),"T_GS_C_6_T":sum(self.m.T_GS_C_T_CR_CC[0,c,6]() for c in self.cc_GS),
                "T_HGS_T":self.m.T_HGS_T[0](),"T_CS_T":self.m.T_CS_T[0](),"T_RLTS_T":self.m.T_RLTS_T[0](),"q_dem_HS_T":self.q_dem_HS_T[0],"q_dem_CS_T":self.q_dem_CS_T[0],"q_dem_RLTS_T":self.q_dem_RLTS_T[0],"temp_amb_T":self.temp_amb_T[0]}, ignore_index=True)
                # ALL
                for t in self.T[1:-1]:
                    self.safeFile = self.safeFile.append({"C_OP_T":self.m.C_OP_T[t](),"C_HP_T":self.m.E_HP_EL_in_T[t]()*self.c_ELECTRICITY_buy_T[t]*self.StepSizeInSec/self.t_hour_in_sec,
                    "C_HXA_T":self.m.E_HXA_EL_T[t]()*self.c_ELECTRICITY_buy_T[t]*self.StepSizeInSec/self.t_hour_in_sec,"C_IS_T":self.m.E_IS_EL_T[t]()*self.c_ELECTRICITY_buy_T[t]*self.StepSizeInSec/self.t_hour_in_sec,
                    "C_GS_T":self.m.E_GS_EL_T[t]()*self.c_ELECTRICITY_buy_T[t]*self.StepSizeInSec/self.t_hour_in_sec,"C_VP_T":self.m.E_VP_EL_T[t]()*self.c_ELECTRICITY_buy_T[t]*self.StepSizeInSec/self.t_hour_in_sec,
                    "B_HP_4_T":self.m.B_HP_H_T[4,t](),"B_HP_3_T":self.m.B_HP_H_T[3,t](),"B_HP_2_T":self.m.B_HP_H_T[2,t](),"B_HP_1_T":self.m.B_HP_H_T[1,t](),"B_HP_0_T":self.m.B_HP_H_T[0,t](),
                    "B_HXH_HS_T":self.m.B_HXH_HS_T[t](),"B_HGC_HGCHXC_T":self.m.B_HGC_HGCHXC_T[t](),"B_HXA_T":self.m.B_HXA_T[t](),"B_HXH_HGC_T":self.m.B_HXH_HGC_T[t](),"B_HS_IS_T":self.m.B_HS_IS_T[t](),"B_IS_HGS_T":self.m.B_IS_HGS_T[t](),
                    "B_GS_HGS_T":self.m.B_GS_HGS_T[t](),"B_GS_CS_T":self.m.B_GS_CS_T[t](),"B_GS_HGS_CS_T":self.m.B_GS_HGS_CS_T[t](),"B_VP_13_T_2":self.m.B_VP_V_T_2[13,t](),"B_VP_12_T_2":self.m.B_VP_V_T_2[12,t](),"B_VP_11_T_2":self.m.B_VP_V_T_2[11,t](),
                    "B_VP_10_T_2":self.m.B_VP_V_T_2[10,t](),"B_VP_9_T_2":self.m.B_VP_V_T_2[9,t](),"B_VP_8_T_2":self.m.B_VP_V_T_2[8,t](),"B_VP_7_T_2":self.m.B_VP_V_T_2[7,t](),"B_VP_6_T_2":self.m.B_VP_V_T_2[6,t](),
                    "B_VP_5_T_2":self.m.B_VP_V_T_2[5,t](),"B_VP_4_T_2":self.m.B_VP_V_T_2[4,t](),"B_VP_3_T_2":self.m.B_VP_V_T_2[3,t](),"B_VP_2_T_2":self.m.B_VP_V_T_2[2,t](),"B_VP_1_T_2":self.m.B_VP_V_T_2[1,t](),"B_VP_0_T_2":self.m.B_VP_V_T_2[0,t](),
                    "E_HP_EL_T":self.m.E_HP_EL_T[t](),"Q_HP_HT_T":self.m.Q_HP_HT_T[t](),"Q_HP_LT_T":self.m.Q_HP_LT_T[t](),"E_HP_EL_in_T":self.m.E_HP_EL_in_T[t](),"T_HP_HT_T":self.m.T_HP_HT_T[t](),
                    "T_HP_LT_T":self.m.T_HP_LT_T[t](),"T_HS_T":self.m.T_HS_T[t](),"T_HXH_T":self.m.T_HXH_T[t](),"T_HGC_T":self.m.T_HGC_T[t](),
                    "T_HXA_T":self.m.T_HXA_T[t](),"T_HXC_T":self.m.T_HXC_T[t](),"T_IS_T":self.m.T_IS_T[t](),"T_ISw_T":(sum(self.m.T_IS_W_T_WR[t,r]() for r in self.wr_IS)/len(self.wr_IS)),"T_ISc_T":(sum(self.m.T_IS_C_T_CR[t,r]() for r in self.cr_IS)/len(self.cr_IS)),
                    "T_IS_W_0_T":self.m.T_IS_W_T_WR[t,0](),"T_IS_W_1_T":self.m.T_IS_W_T_WR[t,2](),"T_IS_W_2_T":self.m.T_IS_W_T_WR[t,4](),"T_IS_C_0_T":self.m.T_IS_C_T_CR[t,0](),"T_IS_C_1_T":self.m.T_IS_C_T_CR[t,1](),"T_IS_C_2_T":self.m.T_IS_C_T_CR[t,2](),"T_IS_C_3_T":self.m.T_IS_C_T_CR[t,3](),"T_IS_C_4_T":self.m.T_IS_C_T_CR[t,4](),
                    "T_GS_T":self.m.T_GS_T[t](),"T_GSw_T":(sum(sum(self.m.T_GS_W_T_WR_WC[t,c,r]() for c in self.wc_GS) for r in self.wr_GS)/(len(self.wc_GS)*len(self.wr_GS))),"T_GSc_T":(sum(sum(self.m.T_GS_C_T_CR_CC[t,c,r]() for c in self.cc_GS) for r in self.cr_GS)/(len(self.cc_GS)*len(self.cr_GS))),
                    "T_GS_W_0_T":sum(self.m.T_GS_W_T_WR_WC[t,c,1]() for c in self.wc_GS),"T_GS_W_1_T":sum(self.m.T_GS_W_T_WR_WC[t,c,3]() for c in self.wc_GS),"T_GS_W_2_T":sum(self.m.T_GS_W_T_WR_WC[t,c,5]() for c in self.wc_GS),
                    "T_GS_C_0_T":sum(self.m.T_GS_C_T_CR_CC[t,c,0]() for c in self.cc_GS),"T_GS_C_1_T":sum(self.m.T_GS_C_T_CR_CC[t,c,1]() for c in self.cc_GS),"T_GS_C_2_T":sum(self.m.T_GS_C_T_CR_CC[t,c,2]() for c in self.cc_GS),"T_GS_C_3_T":sum(self.m.T_GS_C_T_CR_CC[t,c,3]() for c in self.cc_GS),"T_GS_C_4_T":sum(self.m.T_GS_C_T_CR_CC[t,c,4]() for c in self.cc_GS),"T_GS_C_5_T":sum(self.m.T_GS_C_T_CR_CC[t,c,5]() for c in self.cc_GS),"T_GS_C_6_T":sum(self.m.T_GS_C_T_CR_CC[t,c,6]() for c in self.cc_GS),
                    "T_HGS_T":self.m.T_HGS_T[t](),"T_CS_T":self.m.T_CS_T[t](),"T_RLTS_T":self.m.T_RLTS_T[t](),"S_OP_T":self.m.S_OP_T[t](),"S_T_HP_T":self.m.S_T_HP_T[t](),"S_T_HS_T":self.m.S_T_HS_T[t](),"S_T_HXH_T":self.m.S_T_HXH_T[t](),"S_T_HGC_T":self.m.S_T_HGC_T[t](),
                    "S_T_HXA_T":self.m.S_T_HXA_T[t](),"S_T_HXC_T":self.m.S_T_HXC_T[t](),"S_T_IS_T":sum(self.m.S_T_IS_W_T_WR[t,r]() for r in self.wr_IS)+sum(self.m.S_T_IS_C_T_CR[t,r]() for r in self.cr_IS),"S_T_GS_T":sum(sum(self.m.S_T_GS_W_T_WR_WC[t,c,r]() for r in self.wr_GS) for c in self.wc_GS)+sum(sum(self.m.S_T_GS_C_T_CR_CC[t,c,r]() for r in self.cr_GS)for c in self.cc_GS),
                    "S_T_HGS_T":self.m.S_T_HGS_T[t](),"S_T_CS_T":self.m.S_T_CS_T[t](),"S_T_RLTS_T":self.m.S_T_RLTS_T[t](),"q_dem_HS_T":self.q_dem_HS_T[t],"q_dem_CS_T":self.q_dem_CS_T[t],"q_dem_RLTS_T":self.q_dem_RLTS_T[t],"temp_amb_T":self.temp_amb_T[t]}, ignore_index=True)
                # Storage temperatures and slacks
                self.safeFile = self.safeFile.append({"T_HP_HT_T":self.m.T_HP_HT_T[self.T[-1]](),"T_HP_LT_T":self.m.T_HP_LT_T[self.T[-1]](),"T_HS_T":self.m.T_HS_T[self.T[-1]](),"T_HXH_T":self.m.T_HXH_T[self.T[-1]](),"T_HGC_T":self.m.T_HGC_T[self.T[-1]](),
                "T_HXA_T":self.m.T_HXA_T[self.T[-1]](),"T_HXC_T":self.m.T_HXC_T[self.T[-1]](),"T_IS_T":self.m.T_IS_T[self.T[-1]](),"T_ISw_T":(sum(self.m.T_IS_W_T_WR[self.T[-1],r]() for r in self.wr_IS)/len(self.wr_IS)),"T_ISc_T":(sum(self.m.T_IS_C_T_CR[self.T[-1],r]() for r in self.cr_IS)/len(self.cr_IS)),
                "T_IS_W_0_T":self.m.T_IS_W_T_WR[self.T[-1],0](),"T_IS_W_1_T":self.m.T_IS_W_T_WR[self.T[-1],2](),"T_IS_W_2_T":self.m.T_IS_W_T_WR[self.T[-1],4](),"T_IS_C_0_T":self.m.T_IS_C_T_CR[self.T[-1],0](),"T_IS_C_1_T":self.m.T_IS_C_T_CR[self.T[-1],1](),"T_IS_C_2_T":self.m.T_IS_C_T_CR[self.T[-1],2](),"T_IS_C_3_T":self.m.T_IS_C_T_CR[self.T[-1],3](),"T_IS_C_4_T":self.m.T_IS_C_T_CR[self.T[-1],4](),   
                "T_GS_T":self.m.T_GS_T[self.T[-1]](),"T_GSw_T":(sum(sum(self.m.T_GS_W_T_WR_WC[self.T[-1],c,r]() for c in self.wc_GS) for r in self.wr_GS)/(len(self.wc_GS)*len(self.wr_GS))),"T_GSc_T":(sum(sum(self.m.T_GS_C_T_CR_CC[self.T[-1],c,r]() for c in self.cc_GS) for r in self.cr_GS)/(len(self.cc_GS)*len(self.cr_GS))),
                "T_GS_W_0_T":sum(self.m.T_GS_W_T_WR_WC[self.T[-1],c,1]() for c in self.wc_GS),"T_GS_W_1_T":sum(self.m.T_GS_W_T_WR_WC[self.T[-1],c,3]() for c in self.wc_GS),"T_GS_W_2_T":sum(self.m.T_GS_W_T_WR_WC[self.T[-1],c,5]() for c in self.wc_GS),
                "T_GS_C_0_T":sum(self.m.T_GS_C_T_CR_CC[self.T[-1],c,0]() for c in self.cc_GS),"T_GS_C_1_T":sum(self.m.T_GS_C_T_CR_CC[self.T[-1],c,1]() for c in self.cc_GS),"T_GS_C_2_T":sum(self.m.T_GS_C_T_CR_CC[self.T[-1],c,2]() for c in self.cc_GS),"T_GS_C_3_T":sum(self.m.T_GS_C_T_CR_CC[self.T[-1],c,3]() for c in self.cc_GS),"T_GS_C_4_T":sum(self.m.T_GS_C_T_CR_CC[self.T[-1],c,4]() for c in self.cc_GS),"T_GS_C_5_T":sum(self.m.T_GS_C_T_CR_CC[self.T[-1],c,5]() for c in self.cc_GS),"T_GS_C_6_T":sum(self.m.T_GS_C_T_CR_CC[self.T[-1],c,6]() for c in self.cc_GS),
                "T_HGS_T":self.m.T_HGS_T[self.T[-1]](),"T_CS_T":self.m.T_CS_T[self.T[-1]](),
                "T_RLTS_T":self.m.T_RLTS_T[self.T[-1]](),"S_OP_T":self.m.S_OP_T[self.T[-1]](),"S_T_HP_T":self.m.S_T_HP_T[self.T[-1]](),"S_T_HS_T":self.m.S_T_HS_T[self.T[-1]](),"S_T_HGC_T":self.m.S_T_HGC_T[self.T[-1]](),
                "S_T_HXA_T":self.m.S_T_HXA_T[self.T[-1]](),"S_T_IS_T":sum(self.m.S_T_IS_W_T_WR[self.T[-1],r]() for r in self.wr_IS)+sum(self.m.S_T_IS_C_T_CR[self.T[-1],r]() for r in self.cr_IS),"S_T_GS_T":sum(sum(self.m.S_T_GS_W_T_WR_WC[self.T[-1],c,r]() for r in self.wr_GS) for c in self.wc_GS)+sum(sum(self.m.S_T_GS_C_T_CR_CC[self.T[-1],c,r]() for r in self.cr_GS)for c in self.cc_GS),
                "S_T_HGS_T":self.m.S_T_HGS_T[self.T[-1]](),"S_T_CS_T":self.m.S_T_CS_T[self.T[-1]](),"S_T_RLTS_T":self.m.S_T_RLTS_T[self.T[-1]]()}, ignore_index=True)

            self.safeFile = self.safeFile.round(4)
            if singleFile == True:
                source.setOptimizationResults(dataFrame=self.safeFile,savePath=savePath)
            return self.safeFile
        except:
            raise RuntimeError("Optimization T didn't come to a solution.")

if __name__ == "__main__":
    test = Binary_Model()
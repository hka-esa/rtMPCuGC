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

class Long_Term_Model():

    def __init__(self):
        pass

    def setProfiles(self,profileForecastHeat,profileForecastPrice,forecastFrost):
        self.q_dem_HS_J = profileForecastHeat
        self.c_ELECTRICITY_buy_J = profileForecastPrice
        self.forecast_frost = forecastFrost

    def setParams(self,timeSteps,stepSizeInSec):
        if self.forecast_frost == True:
            ## Time
            self.J = timeSteps
            self.StepSizeInSec = stepSizeInSec 
            ## General
            self.c_w = 4.18
            self.c_b = 3.56
            self.c_a = 1.01
            self.c_c = 0.879
            self.t_default = 20
            self.t_hour_in_sec = 3600
            ## Slack constants
            self.s_T_HS = 20
            self.s_T_GS_C = 20
            self.s_T_GS_W = 20
            ## HS
            self.alpha_HS_time = 0.01 
            self.T_HS_max = 40 
            self.T_HS_min = 33 
            self.m_HS_w = 6000 
            self.t_HS_out = 35
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
            self.t_GS_out = 5
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
            self.mdot_HP_w = 24.44
            self.mdot_HP_b = 39.44
            self.e_HP_EL_pumps = 3.5 * 2 + 1.3 * 4 # see datasheets 
            self.q_HP_HT = self.a_HP_HT_0 + self.a_HP_HT_1 * self.t_HS_out + self.a_HP_HT_2 * self.t_GS_out 
            self.q_HP_LT = self.a_HP_LT_0 + self.a_HP_LT_1 * self.t_HS_out + self.a_HP_LT_2 * self.t_GS_out 
            self.e_HP_EL = self.a_HP_EL_0 + self.a_HP_EL_1 * self.t_HS_out + self.a_HP_EL_2 * self.t_GS_out 
        else:
            pass

    def setVariables(self,model):
        self.m = model
        if self.forecast_frost == True:
            ## General variables
            self.m.C_TOT_J_ = pyo.Var(domain=pyo.NonNegativeReals)
            self.m.C_OP_J = pyo.Var(self.J[0:-1], domain=pyo.NonNegativeReals)
            ## Slack variables
            self.m.S_TOT_J_ = pyo.Var(domain=pyo.NonNegativeReals)
            self.m.S_OP_J = pyo.Var(self.J[1:], domain=pyo.NonNegativeReals)
            ## Toggle variables (Switching variables)
            self.m.T_TOT_J_ = pyo.Var(domain=pyo.NonNegativeReals)
            ## HP
            self.m.P_HP_J = pyo.Var(self.J[0:-1], domain=pyo.NonNegativeReals, bounds=(0,2))
            self.m.E_HP_EL_J = pyo.Var(self.J[0:-1], domain=pyo.NonNegativeReals)
            self.m.Q_HP_HT_J = pyo.Var(self.J[0:-1], domain=pyo.NonNegativeReals)
            self.m.Q_HP_LT_J = pyo.Var(self.J[0:-1], domain=pyo.NonNegativeReals)
            ## HS  
            self.m.T_HS_J = pyo.Var(self.J, domain=pyo.Reals)
            self.m.S_T_HS_J = pyo.Var(self.J, domain=pyo.NonNegativeReals)
            ## GS
            self.m.T_GS_C_J_CR_CC = pyo.Var(self.J,self.cc_GS,self.cr_GS, domain=pyo.Reals)
            self.m.T_GS_W_J_WR_WC = pyo.Var(self.J,self.wc_GS,self.wr_GS, domain=pyo.Reals)
            self.m.S_T_GS_C_J_CR_CC = pyo.Var(self.J,self.cc_GS,self.cr_GS, domain=pyo.NonNegativeReals)
            self.m.S_T_GS_W_J_WR_WC = pyo.Var(self.J,self.wc_GS,self.wr_GS, domain=pyo.NonNegativeReals)
            self.m.Q_GS_C_NORTH_J_CR_CC = pyo.Var(self.J,self.cc_GS,self.cr_GS, domain=pyo.Reals)
            self.m.Q_GS_C_EAST_J_CR_CC = pyo.Var(self.J,self.cc_GS,self.cr_GS, domain=pyo.Reals)
            self.m.Q_GS_C_SOUTH_J_CR_CC = pyo.Var(self.J,self.cc_GS,self.cr_GS, domain=pyo.Reals)
            self.m.Q_GS_C_WEST_J_CR_CC = pyo.Var(self.J,self.cc_GS,self.cr_GS, domain=pyo.Reals)
            self.m.Q_GS_C_W_J_WR_WC = pyo.Var(self.J,self.wc_GS,self.wr_GS, domain=pyo.Reals)
            self.m.Q_GS_W_EAST_J_WR_WC = pyo.Var(self.J,self.wc_GS,self.wr_GS, domain=pyo.Reals)
            self.m.Q_GS_W_WEST_J_WR_WC = pyo.Var(self.J,self.wc_GS,self.wr_GS, domain=pyo.Reals)
            self.m.Q_GS_W_C_J_WR_WC = pyo.Var(self.J,self.wc_GS,self.wr_GS, domain=pyo.Reals)
            self.m.T_GS_J = pyo.Var(self.J, domain=pyo.Reals)
            self.m.E_GS_EL_J = pyo.Var(self.J[0:-1], domain=pyo.NonNegativeReals)
        else:
            self.m.C_TOT_J_ = pyo.Var(domain=pyo.NonNegativeReals)
            self.m.S_TOT_J_ = pyo.Var(domain=pyo.NonNegativeReals)
            self.m.T_TOT_J_ = pyo.Var(domain=pyo.NonNegativeReals)
        return self.m

    def setStartValues(self,model,T_HS_start,T_GS_w_1_start,T_GS_w_2_start,T_GS_w_3_start,T_GS_c_1_start,T_GS_c_2_start,T_GS_c_3_start,T_GS_c_4_start,T_GS_c_5_start,T_GS_c_6_start,T_GS_c_7_start):
        self.m = model
        if self.forecast_frost == True:
            self.T_HS_start = T_HS_start
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
        else:
            pass
        return self.m

    def setEndValues(self,model,End_Temp_Constraints):
        self.m = model
        if End_Temp_Constraints == True:
            pass
        return self.m 

    def setConstraints(self,model):
        self.m = model
        if self.forecast_frost == True:
            ## General cost constraint
            self.m.Constraint_Cost_J = pyo.Constraint(expr = self.m.C_TOT_J_ == sum(self.m.C_OP_J[j] for j in self.J[0:-1]))

            ## Cost constraints 
            self.m.Constraint_Cost_time_J = pyo.ConstraintList()
            for j in self.J[0:-1]:
                self.m.Constraint_Cost_time_J.add(self.m.C_OP_J[j] == self.StepSizeInSec/self.t_hour_in_sec * (self.m.E_HP_EL_J[j] * self.c_ELECTRICITY_buy_J[j]))

            ## General Slack constraint
            self.m.Constraint_Slack_J = pyo.Constraint(expr = self.m.S_TOT_J_ == sum(self.m.S_OP_J[j] for j in self.J[1:]))

            ## Slack constraints
            self.m.Constraint_Slack_time_J = pyo.ConstraintList()
            for j in self.J[1:]:
                self.m.Constraint_Slack_time_J.add(self.m.S_OP_J[j] == self.StepSizeInSec/self.t_hour_in_sec * (self.s_T_HS * self.m.S_T_HS_J[j] + sum(sum(self.s_T_GS_W * self.m.S_T_GS_W_J_WR_WC[j,c,r] for r in self.wr_GS) for c in self.wc_GS) + sum(sum(self.s_T_GS_C * self.m.S_T_GS_C_J_CR_CC[j,c,r] for r in self.cr_GS) for c in self.cc_GS)))

            ## General toggle constraint
            self.m.Constraint_Toggle_J = pyo.Constraint(expr = self.m.T_TOT_J_ == 0)

            ## HP
            self.m.Constraint_HP_J = pyo.ConstraintList()
            for j in self.J[0:-1]:
                self.m.Constraint_HP_J.add(self.m.Q_HP_HT_J[j] == self.q_HP_HT * self.m.P_HP_J[j])
                self.m.Constraint_HP_J.add(self.m.Q_HP_LT_J[j] == self.q_HP_LT * self.m.P_HP_J[j])
                self.m.Constraint_HP_J.add(self.m.E_HP_EL_J[j] == (self.e_HP_EL + self.e_HP_EL_pumps + self.e_GS_EL) * self.m.P_HP_J[j]) # heat pump and ground slab, since both are needed for water flow

            ## HS
            self.m.Constraint_HS_J = pyo.ConstraintList()
            self.m.Constraint_HS_J.add(self.m.T_HS_J[0] == self.T_HS_start) ## Start temperature

            for j in self.J[0:-1]: 
                self.m.Constraint_HS_J.add(self.m.T_HS_J[j+1] == self.m.T_HS_J[j] + self.StepSizeInSec * self.m.Q_HP_HT_J[j]/(self.m_HS_w * self.c_w) - self.StepSizeInSec * self.q_dem_HS_J[j]/(self.m_HS_w * self.c_w) + self.StepSizeInSec * self.alpha_HS_time * (self.t_default - self.m.T_HS_J[j+1])/(self.m_HS_w * self.c_w)) 

            for j in self.J[1:]:
                self.m.Constraint_HS_J.add(self.m.T_HS_J[j] <= self.T_HS_max + self.m.S_T_HS_J[j]) ## Temperature range tank
                self.m.Constraint_HS_J.add(self.m.T_HS_J[j] >= self.T_HS_min - self.m.S_T_HS_J[j]) ## Temperature range tank

            ## GS
            self.m.Constraint_GS_J = pyo.ConstraintList()
            ## Concrete energy flows
            for j in self.J[:-1]:
                for c in self.cc_GS:
                    for r in self.cr_GS[1:]:
                        self.m.Constraint_GS_J.add(self.m.Q_GS_C_NORTH_J_CR_CC[j,c,r] == (self.m.T_GS_C_J_CR_CC[j+1,c,r-1] - self.m.T_GS_C_J_CR_CC[j+1,c,r]) * self.lambda_GS_c_c / self.height_GS * self.a_north_south_GS)
            
                for c in self.cc_GS:
                    for r in self.cr_GS[:-1]:
                        self.m.Constraint_GS_J.add(self.m.Q_GS_C_SOUTH_J_CR_CC[j,c,r] == (self.m.T_GS_C_J_CR_CC[j+1,c,r+1] - self.m.T_GS_C_J_CR_CC[j+1,c,r]) * self.lambda_GS_c_c / self.height_GS * self.a_north_south_GS)

                for c in self.cc_GS[1:]:
                    for r in self.cr_GS:
                        self.m.Constraint_GS_J.add(self.m.Q_GS_C_WEST_J_CR_CC[j,c,r] == (self.m.T_GS_C_J_CR_CC[j+1,c-1,r] - self.m.T_GS_C_J_CR_CC[j+1,c,r]) * self.lambda_GS_c_c / self.width_GS * self.a_east_west_GS)

                for c in self.cc_GS[:-1]:
                    for r in self.cr_GS:
                        self.m.Constraint_GS_J.add(self.m.Q_GS_C_EAST_J_CR_CC[j,c,r] == (self.m.T_GS_C_J_CR_CC[j+1,c+1,r] - self.m.T_GS_C_J_CR_CC[j+1,c,r]) * self.lambda_GS_c_c / self.width_GS * self.a_east_west_GS)

                for c in self.wc_GS:
                    for r in self.wr_GS:
                        self.m.Constraint_GS_J.add(self.m.Q_GS_C_W_J_WR_WC[j,c,r] == (self.m.T_GS_W_J_WR_WC[j+1,c,r] - self.m.T_GS_C_J_CR_CC[j+1,c,r]) * self.alpha_GS_w_c * self.a_pipe_GS)

            ## Concrete borders
            for j in self.J[:-1]:
                for c in self.cc_GS:
                    self.m.Constraint_GS_J.add(self.m.Q_GS_C_NORTH_J_CR_CC[j,c,0] == (self.t_GS_air - self.m.T_GS_C_J_CR_CC[j+1,c,0]) * self.lambda_GS_c_a / self.height_GS * self.a_north_south_GS)

                for c in self.cc_GS:
                    self.m.Constraint_GS_J.add(self.m.Q_GS_C_SOUTH_J_CR_CC[j,c,self.cr_GS[-1]] == (self.t_GS_soil - self.m.T_GS_C_J_CR_CC[j+1,c,self.cr_GS[-1]]) * self.lambda_GS_c_s / self.height_GS * self.a_north_south_GS)

                if len(self.cc_GS) > 1:
                    for r in self.cr_GS:
                        self.m.Constraint_GS_J.add(self.m.Q_GS_C_WEST_J_CR_CC[j,0,r] == (self.m.T_GS_C_J_CR_CC[j+1,self.cc_GS[-1],r] - self.m.T_GS_C_J_CR_CC[j+1,0,r]) * self.lambda_GS_c_c / self.width_GS * self.a_east_west_GS)

                    for r in self.cr_GS:
                        self.m.Constraint_GS_J.add(self.m.Q_GS_C_EAST_J_CR_CC[j,self.cc_GS[-1],r] == (self.m.T_GS_C_J_CR_CC[j+1,0,r] - self.m.T_GS_C_J_CR_CC[j+1,self.cc_GS[-1],r]) * self.lambda_GS_c_c / self.width_GS * self.a_east_west_GS)
                else:
                    for r in self.cr_GS:
                        self.m.Constraint_GS_J.add(self.m.Q_GS_C_WEST_J_CR_CC[j,0,r] == 0)

                    for r in self.cr_GS:
                        self.m.Constraint_GS_J.add(self.m.Q_GS_C_EAST_J_CR_CC[j,self.cc_GS[-1],r] == 0)    

            ## Concrete temperature
            self.m.Constraint_GS_J.add(self.m.T_GS_C_J_CR_CC[0,0,0] == self.T_GS_c_1_start)
            self.m.Constraint_GS_J.add(self.m.T_GS_C_J_CR_CC[0,0,1] == self.T_GS_c_2_start)
            self.m.Constraint_GS_J.add(self.m.T_GS_C_J_CR_CC[0,0,2] == self.T_GS_c_3_start)
            self.m.Constraint_GS_J.add(self.m.T_GS_C_J_CR_CC[0,0,3] == self.T_GS_c_4_start)
            self.m.Constraint_GS_J.add(self.m.T_GS_C_J_CR_CC[0,0,4] == self.T_GS_c_5_start)
            self.m.Constraint_GS_J.add(self.m.T_GS_C_J_CR_CC[0,0,5] == self.T_GS_c_6_start)
            self.m.Constraint_GS_J.add(self.m.T_GS_C_J_CR_CC[0,0,6] == self.T_GS_c_7_start)

            for j in self.J[:-1]:
                for c in self.cc_GS:
                    for r in self.cr_GS:
                        if r in self.wr_GS:
                            self.m.Constraint_GS_J.add(self.m.T_GS_C_J_CR_CC[j+1,c,r] == self.m.T_GS_C_J_CR_CC[j,c,r] + self.StepSizeInSec * (1/(self.m_GS_c * self.c_c)) * (self.m.Q_GS_C_NORTH_J_CR_CC[j,c,r] + self.m.Q_GS_C_SOUTH_J_CR_CC[j,c,r] + self.m.Q_GS_C_WEST_J_CR_CC[j,c,r] + self.m.Q_GS_C_EAST_J_CR_CC[j,c,r] + self.m.Q_GS_C_W_J_WR_WC[j,c,r]))
                        else:
                            self.m.Constraint_GS_J.add(self.m.T_GS_C_J_CR_CC[j+1,c,r] == self.m.T_GS_C_J_CR_CC[j,c,r] + self.StepSizeInSec * (1/(self.m_GS_c * self.c_c)) * (self.m.Q_GS_C_NORTH_J_CR_CC[j,c,r] + self.m.Q_GS_C_SOUTH_J_CR_CC[j,c,r] + self.m.Q_GS_C_WEST_J_CR_CC[j,c,r] + self.m.Q_GS_C_EAST_J_CR_CC[j,c,r]))
            
            for j in self.J[1:]:
                for c in self.cc_GS:
                    for r in self.cr_GS:
                        self.m.Constraint_GS_J.add(self.m.T_GS_C_J_CR_CC[j,c,r] <= self.T_GS_max_c + self.m.S_T_GS_C_J_CR_CC[j,c,r])
                        self.m.Constraint_GS_J.add(self.m.T_GS_C_J_CR_CC[j,c,r] >= self.T_GS_min_c - self.m.S_T_GS_C_J_CR_CC[j,c,r])

            ## Water energy flows
            for j in self.J[:-1]:
                for r in self.wr_GS[::2]:
                    for c in self.wc_GS[1:]:
                        self.m.Constraint_GS_J.add(self.m.Q_GS_W_WEST_J_WR_WC[j,c,r] == (self.m.T_GS_W_J_WR_WC[j+1,c-1,r] - self.m.T_GS_W_J_WR_WC[j+1,c,r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w)
                    for c in self.wc_GS[:-1]:    
                        self.m.Constraint_GS_J.add(self.m.Q_GS_W_EAST_J_WR_WC[j,c,r] == (self.m.T_GS_W_J_WR_WC[j+1,c+1,r] - self.m.T_GS_W_J_WR_WC[j+1,c,r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w)

                for r in self.wr_GS[1::2]:
                    for c in self.wc_GS[:-1]:
                        self.m.Constraint_GS_J.add(self.m.Q_GS_W_EAST_J_WR_WC[j,c,r] == (self.m.T_GS_W_J_WR_WC[j+1,c+1,r] - self.m.T_GS_W_J_WR_WC[j+1,c,r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w)
                    for c in self.wc_GS[1:]:    
                        self.m.Constraint_GS_J.add(self.m.Q_GS_W_WEST_J_WR_WC[j,c,r] == (self.m.T_GS_W_J_WR_WC[j+1,c-1,r] - self.m.T_GS_W_J_WR_WC[j+1,c,r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w)

                for c in self.wc_GS:
                    for r in self.wr_GS:
                        self.m.Constraint_GS_J.add(self.m.Q_GS_W_C_J_WR_WC[j,c,r] == (self.m.T_GS_C_J_CR_CC[j+1,c,r] - self.m.T_GS_W_J_WR_WC[j+1,c,r]) * self.alpha_GS_w_c * self.a_pipe_GS)      
            
            ## Water borders
            for j in self.J[:-1]:
                for r in self.wr_GS[0::2]:
                    if r > 1:
                        self.m.Constraint_GS_J.add(self.m.Q_GS_W_WEST_J_WR_WC[j,0,r] == (self.m.T_GS_W_J_WR_WC[j+1,0,r-2] - self.m.T_GS_W_J_WR_WC[j+1,0,r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w)
                        self.m.Constraint_GS_J.add(self.m.Q_GS_W_EAST_J_WR_WC[j,self.wc_GS[-1],r] == 0)
                    else: ## Start inflow
                        self.m.Constraint_GS_J.add(self.m.Q_GS_W_WEST_J_WR_WC[j,0,1] == -(self.m.Q_HP_LT_J[j]/self.n_GS_blocks))
                        self.m.Constraint_GS_J.add(self.m.Q_GS_W_EAST_J_WR_WC[j,self.wc_GS[-1],r] == (self.m.T_GS_W_J_WR_WC[j+1,self.wc_GS[-1],r+2] - self.m.T_GS_W_J_WR_WC[j+1,self.wc_GS[-1],r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w)
                for r in self.wr_GS[1::2]:
                    self.m.Constraint_GS_J.add(self.m.Q_GS_W_EAST_J_WR_WC[j,self.wc_GS[-1],r] == (self.m.T_GS_W_J_WR_WC[j+1,self.wc_GS[-1],r-2] - self.m.T_GS_W_J_WR_WC[j+1,self.wc_GS[-1],r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w)
                    self.m.Constraint_GS_J.add(self.m.Q_GS_W_WEST_J_WR_WC[j,0,r] == (self.m.T_GS_W_J_WR_WC[j+1,0,r+2] - self.m.T_GS_W_J_WR_WC[j+1,0,r]) * self.mdot_GS_w/self.n_GS_blocks * self.c_w)

            ## Water temperature
            self.m.Constraint_GS_J.add(self.m.T_GS_W_J_WR_WC[0,0,1] == self.T_GS_w_1_start)
            self.m.Constraint_GS_J.add(self.m.T_GS_W_J_WR_WC[0,0,3] == self.T_GS_w_2_start)
            self.m.Constraint_GS_J.add(self.m.T_GS_W_J_WR_WC[0,0,5] == self.T_GS_w_3_start)

            for j in self.J[:-1]:
                for c in self.wc_GS:
                    for r in self.wr_GS:
                        self.m.Constraint_GS_J.add(self.m.T_GS_W_J_WR_WC[j+1,c,r] == self.m.T_GS_W_J_WR_WC[j,c,r] + self.StepSizeInSec * (1/(self.m_GS_w * self.c_w)) * (self.m.Q_GS_W_WEST_J_WR_WC[j,c,r] + self.m.Q_GS_W_EAST_J_WR_WC[j,c,r] + self.m.Q_GS_W_C_J_WR_WC[j,c,r]))

            for j in self.J[1:]:
                for c in self.wc_GS:
                    for r in self.wr_GS:
                        self.m.Constraint_GS_J.add(self.m.T_GS_W_J_WR_WC[j,c,r] <= self.T_GS_max_w + self.m.S_T_GS_W_J_WR_WC[j,c,r])
                        self.m.Constraint_GS_J.add(self.m.T_GS_W_J_WR_WC[j,c,r] >= self.T_GS_min_w - self.m.S_T_GS_W_J_WR_WC[j,c,r])

            ## Water outflow
            for j in self.J:
                if self.wr_GS[::2][-1] > self.wr_GS[1::2][-1]:
                    self.m.Constraint_GS_J.add(self.m.T_GS_J[j] == self.m.T_GS_W_J_WR_WC[j,self.wc_GS[-1],self.wr_GS[-1]])
                else:
                    self.m.Constraint_GS_J.add(self.m.T_GS_J[j] == self.m.T_GS_W_J_WR_WC[j,self.wc_GS[0],self.wr_GS[-1]])
        else:
            self.m.Constraint_Cost_J = pyo.Constraint(expr = self.m.C_TOT_J_ == 0)
            self.m.Constraint_Slack_J = pyo.Constraint(expr = self.m.S_TOT_J_ == 0)
            self.m.Constraint_Toggle_J = pyo.Constraint(expr = self.m.T_TOT_J_ == 0)
        return self.m

    def setObjective(self,model):
        self.m = model
        if self.forecast_frost == True:
            self.m.OBJ = pyo.Objective(expr = self.m.C_TOT_J_ + self.m.S_TOT_J_ + self.m.T_TOT_J_)
        else:
            pass
        return self.m

    def setWarmstart(self,model):
        self.m = model
        if self.forecast_frost == True:
            self.warmstart_available = False
        else:
            pass
        return self.m

    def setSolverAndRunOptimization(self,model,solver = 0, showSolverOutput = 0, writeILP = 0, writeMPSfile = 0):
        self.m = model
        if self.forecast_frost == True:
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
        else:
            pass
        return self.m

    def getResults(self,model,source=None,savePath="",singleFile=False):
        self.m = model
        if self.forecast_frost == True:
            self.safeFile = pd.DataFrame(columns = ["C_OP_J","E_HP_EL_J","Q_HP_HT_J","Q_HP_LT_J","T_HS_J","T_GS_J","S_OP_J","S_T_HS_J","S_T_GS_J","q_dem_HS_J"]) 
            try: # Everything but slack constraints
                self.safeFile = self.safeFile.append({"C_OP_J":self.m.C_OP_J[0](),"E_HP_EL_J":self.m.E_HP_EL_J[0](),"Q_HP_HT_J":self.m.Q_HP_HT_J[0](),"Q_HP_LT_J":self.m.Q_HP_LT_J[0](),
                "T_HS_J":self.m.T_HS_J[0](),"T_GS_J":self.m.T_GS_J[0](),"q_dem_HS_J":self.q_dem_HS_J[0]}, ignore_index=True)
                # ALL
                for j in self.J[1:-1]:
                    self.safeFile = self.safeFile.append({"C_OP_J":self.m.C_OP_J[j](),"E_HP_EL_J":self.m.E_HP_EL_J[j](),"Q_HP_HT_J":self.m.Q_HP_HT_J[j](),"Q_HP_LT_J":self.m.Q_HP_LT_J[j](),
                    "T_HS_J":self.m.T_HS_J[j](),"T_GS_J":self.m.T_GS_J[j](),"S_OP_J":self.m.S_OP_J[self.J[-1]](),"S_T_HS_J":self.m.S_T_HS_J[self.J[-1]](),"S_T_GS_J":sum(sum(self.m.S_T_GS_W_J_WR_WC[j,c,r]() for r in self.wr_GS) for c in self.wc_GS)+sum(sum(self.m.S_T_GS_C_J_CR_CC[j,c,r]() for r in self.cr_GS)for c in self.cc_GS),"q_dem_HS_J":self.q_dem_HS_J[j]}, ignore_index=True)
                # Storage temperatures and slacks
                self.safeFile = self.safeFile.append({"T_HS_J":self.m.T_HS_J[self.J[-1]](),"T_GS_J":self.m.T_GS_J[self.J[-1]](),"S_OP_J":self.m.S_OP_J[self.J[-1]](),"S_T_HS_J":self.m.S_T_HS_J[self.J[-1]](),"S_T_GS_J":sum(sum(self.m.S_T_GS_W_J_WR_WC[self.J[-1],c,r]() for r in self.wr_GS) for c in self.wc_GS)+sum(sum(self.m.S_T_GS_C_J_CR_CC[self.J[-1],c,r]() for r in self.cr_GS)for c in self.cc_GS)}, ignore_index=True)
                print("Frost Period added")

                self.safeFile = self.safeFile.round(4)
                if singleFile == True:
                    source.setOptimizationResults(dataFrame=self.safeFile,savePath=savePath)
                return self.safeFile
            except:
                raise RuntimeError("Optimization J didn't come to a solution.")
        else:
            print("No Frost Period added")
            return (pd.DataFrame(columns = ["C_OP_J","E_HP_EL_J","Q_HP_HT_J","Q_HP_LT_J","T_HS_J","T_GS_J","S_OP_J","S_T_HS_J","S_T_GS_J","q_dem_HS_J"])) # return empty file

if __name__ == "__main__":
    test = Long_Term_Model()
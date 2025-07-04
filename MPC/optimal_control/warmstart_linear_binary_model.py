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
from datetime import timedelta
import time

from optimal_control.linear_binary_model import *

class Warmstart_Linear_Binary_Model():
    
    def __init__(self, timelimitWarmstart, warmstartPartitionLinearBinary, savingPathWarmstartSystemVals="", savingWarmstartSystemVals=False, sourceSavingSystemVals=None):
        self.timelimitWarmstart = timelimitWarmstart
        self.warmstartPartitionLinearBinary = warmstartPartitionLinearBinary
        self.savingPathWarmstartSystemVals = savingPathWarmstartSystemVals
        self.savingWarmstartSystemVals= savingWarmstartSystemVals
        self.optimization_results = pd.DataFrame()
        self.result_interface = sourceSavingSystemVals

    def setProfiles(self,profileForecastHeat,profileForecastCool,profileForecastDry,profileForecastWeather,profileForecastPrice,profileForecastFrost):
        self.profileForecastHeat = profileForecastHeat
        self.profileForecastCool = profileForecastCool
        self.profileForecastDry = profileForecastDry
        self.profileForecastWeather = profileForecastWeather
        self.profileForecastPrice = profileForecastPrice
        self.profileForecastFrost = profileForecastFrost

    def setParams(self, timestepsLinearBinary, stepSizeLinearBinary, controlPeriod,NMcCormick):
        self.timestepsLinearBinary = timestepsLinearBinary
        self.stepSizeLinearBinary = stepSizeLinearBinary
        self.controlPeriod = controlPeriod
        self.NMcCormick = NMcCormick

    def setStartValues(self,T_HP_HT_start,T_HP_LT_start,T_HS_start,T_HXA_start,T_HXH_start,T_HGC_start,T_HXC_start,T_HGS_start,T_IS_w_1_start,T_IS_w_2_start,T_IS_w_3_start,T_IS_c_1_start,T_IS_c_2_start,T_IS_c_3_start,T_IS_c_4_start,T_IS_c_5_start,T_GS_w_1_start,T_GS_w_2_start,T_GS_w_3_start,T_GS_c_1_start,T_GS_c_2_start,T_GS_c_3_start,T_GS_c_4_start,T_GS_c_5_start,T_GS_c_6_start,T_GS_c_7_start,T_CS_start,T_RLTS_start):
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
        self.T_CS_start = T_CS_start
        self.T_RLTS_start = T_RLTS_start

    def runWarmstart(self,solver = 0, showSolverOutput = 0):
        print("Warmstart linear binary model started")
        resultsFile = {} 
        j = 0

        partitionStepsTime = [int((self.timestepsLinearBinary+self.warmstartPartitionLinearBinary-1)/self.warmstartPartitionLinearBinary+0.999)] * (self.warmstartPartitionLinearBinary-1) + [self.timestepsLinearBinary-((int((self.timestepsLinearBinary+self.warmstartPartitionLinearBinary-1)/self.warmstartPartitionLinearBinary+0.999)-1)*(self.warmstartPartitionLinearBinary-1))]
        timeStepStartPartition = 0 

        if partitionStepsTime[-1] == 1:
            partitionStepsTime[-2] = partitionStepsTime[-2] - 2
            partitionStepsTime[-1] = 3
        elif partitionStepsTime[-1] == 2:
            partitionStepsTime[-2] = partitionStepsTime[-2] - 1
            partitionStepsTime[-1] = 3

        for partitionTimeSteps in partitionStepsTime:
            print("Optimizing model part " +str(j+1) + " of " + str(len(partitionStepsTime)) + ".")
            if timeStepStartPartition > 0:
                self.m_prev = self.m
            self.m = pyo.ConcreteModel()
            linear_binary_model = Linear_Binary_Model()

            linear_binary_model.setProfiles(profileForecastHeat=self.profileForecastHeat[timeStepStartPartition:timeStepStartPartition+partitionTimeSteps-1],profileForecastCool=self.profileForecastCool[timeStepStartPartition:timeStepStartPartition+partitionTimeSteps-1],profileForecastDry=self.profileForecastDry[timeStepStartPartition:timeStepStartPartition+partitionTimeSteps-1],profileForecastWeather=self.profileForecastWeather[timeStepStartPartition:timeStepStartPartition+partitionTimeSteps-1],profileForecastPrice=self.profileForecastPrice[timeStepStartPartition:timeStepStartPartition+partitionTimeSteps-1],profileForecastFrost=self.profileForecastFrost[timeStepStartPartition:timeStepStartPartition+partitionTimeSteps-1])
            linear_binary_model.setParams(timeSteps=list(range(0,partitionTimeSteps)),stepSizeInSec=self.stepSizeLinearBinary,controlPeriod=self.controlPeriod,NMcCormick=self.NMcCormick)
            self.m = linear_binary_model.setVariables(self.m)                                                          

            if timeStepStartPartition == 0:
                self.m = linear_binary_model.setStartValues(model=self.m,T_HP_HT_start=self.T_HP_HT_start,T_HP_LT_start=self.T_HP_LT_start,T_HS_start=self.T_HS_start,
                T_HXA_start=self.T_HXA_start,T_HXH_start=self.T_HXH_start,T_HGC_start=self.T_HGC_start,T_HXC_start=self.T_HXC_start,T_HGS_start=self.T_HGS_start,
                T_IS_w_1_start=self.T_IS_w_1_start,T_IS_w_2_start=self.T_IS_w_2_start,
                T_IS_w_3_start=self.T_IS_w_3_start,T_IS_c_1_start=self.T_IS_c_1_start,T_IS_c_2_start=self.T_IS_c_2_start,T_IS_c_3_start=self.T_IS_c_3_start,
                T_IS_c_4_start=self.T_IS_c_4_start,T_IS_c_5_start=self.T_IS_c_5_start,T_GS_w_1_start=self.T_GS_w_1_start,
                T_GS_w_2_start=self.T_GS_w_2_start,T_GS_w_3_start=self.T_GS_w_3_start,T_GS_c_1_start=self.T_GS_c_1_start,T_GS_c_2_start=self.T_GS_c_2_start,
                T_GS_c_3_start=self.T_GS_c_3_start,T_GS_c_4_start=self.T_GS_c_4_start,T_GS_c_5_start=self.T_GS_c_5_start,T_GS_c_6_start=self.T_GS_c_6_start,
                T_GS_c_7_start=self.T_GS_c_7_start,T_CS_start=self.T_CS_start,T_RLTS_start=self.T_RLTS_start)
            else:
                self.m = linear_binary_model.setStartValues(model=self.m,T_HP_HT_start=self.m_prev.T_HP_HT_I[partitionStepsTime[j-1]-1](),T_HP_LT_start=self.m_prev.T_HP_LT_I[partitionStepsTime[j-1]-1](),
                T_HS_start=self.m_prev.T_HS_I[partitionStepsTime[j-1]-1](),T_HXA_start=self.m_prev.T_HXA_I[partitionStepsTime[j-1]-1](),T_HXH_start=self.m_prev.T_HXH_w_I[partitionStepsTime[j-1]-1](),T_HGC_start=self.m_prev.T_HGC_I[partitionStepsTime[j-1]-1](),
                T_HXC_start=self.m_prev.T_HXC_b_I[partitionStepsTime[j-1]-1](),T_HGS_start=self.m_prev.T_HGS_I[partitionStepsTime[j-1]-1](),T_IS_w_1_start=self.m_prev.T_IS_W_I_WR[partitionStepsTime[j-1]-1,0](),T_IS_w_2_start=self.m_prev.T_IS_W_I_WR[partitionStepsTime[j-1]-1,2](),
                T_IS_w_3_start=self.m_prev.T_IS_W_I_WR[partitionStepsTime[j-1]-1,4](),T_IS_c_1_start=self.m_prev.T_IS_C_I_CR[partitionStepsTime[j-1]-1,0](),T_IS_c_2_start=self.m_prev.T_IS_C_I_CR[partitionStepsTime[j-1]-1,1](),
                T_IS_c_3_start=self.m_prev.T_IS_C_I_CR[partitionStepsTime[j-1]-1,2](),T_IS_c_4_start=self.m_prev.T_IS_C_I_CR[partitionStepsTime[j-1]-1,3](),T_IS_c_5_start=self.m_prev.T_IS_C_I_CR[partitionStepsTime[j-1]-1,4](),
                T_GS_w_1_start=self.m_prev.T_GS_W_I_WR_WC[partitionStepsTime[j-1]-1,0,1](),T_GS_w_2_start=self.m_prev.T_GS_W_I_WR_WC[partitionStepsTime[j-1]-1,0,3](),T_GS_w_3_start=self.m_prev.T_GS_W_I_WR_WC[partitionStepsTime[j-1]-1,0,5](),
                T_GS_c_1_start=self.m_prev.T_GS_C_I_CR_CC[partitionStepsTime[j-1]-1,0,0](),T_GS_c_2_start=self.m_prev.T_GS_C_I_CR_CC[partitionStepsTime[j-1]-1,0,1](),T_GS_c_3_start=self.m_prev.T_GS_C_I_CR_CC[partitionStepsTime[j-1]-1,0,2](),
                T_GS_c_4_start=self.m_prev.T_GS_C_I_CR_CC[partitionStepsTime[j-1]-1,0,3](),T_GS_c_5_start=self.m_prev.T_GS_C_I_CR_CC[partitionStepsTime[j-1]-1,0,4](),T_GS_c_6_start=self.m_prev.T_GS_C_I_CR_CC[partitionStepsTime[j-1]-1,0,5](),
                T_GS_c_7_start=self.m_prev.T_GS_C_I_CR_CC[partitionStepsTime[j-1]-1,0,6](),T_CS_start=self.m_prev.T_CS_I[partitionStepsTime[j-1]-1](),T_RLTS_start=self.m_prev.T_RLTS_I[partitionStepsTime[j-1]-1]())

            self.m = linear_binary_model.setEndValues(model=self.m,End_Temp_Constraints=False,T_HS_end=0,T_CS_end=0,T_RLTS_end=0,End_Toggle_Constraints=False,B_HP_1_end=0,B_HP_2_end=0,B_HP_3_end=0,B_HP_4_end=0,V_HP_HXH_end=0,V_HP_HS_end=0,V_HP_HGC_end=0,V_HGCHXC_end=0,V_HXA_end=0,V_HXA_HXH_end=0,V_HS_IS_end=0,V_IS_HGS_end=0,V_HXA_HGC_end=0,V_GS_HGS_end=0,V_GS_CS_end=0)
            self.m = linear_binary_model.setConstraints(model=self.m)
            self.m = linear_binary_model.setWarmstart(model=self.m,available=False,file=None)
            self.m = linear_binary_model.setObjective(model=self.m)

            if solver == 0:
                self.opt = pyo.SolverFactory('gurobi', solver_io="python")
                self.opt.options['TimeLimit'] = int(self.timelimitWarmstart/self.warmstartPartitionLinearBinary)
                self.opt.options['threads'] = 8
                #self.opt.options['MIPFocus'] = 1
                #self.opt.options['ObjBound'] = 50
                #self.opt.options['Cutoff'] = 500
            elif solver == 1:
                self.opt = pyo.SolverFactory('cbc')
                self.opt.options['Sec'] = int(self.timelimitWarmstart/self.warmstartPartitionLinearBinary)
            elif solver == 2:
                self.opt = pyo.SolverFactory('glpk')
                self.opt.options['tmlim'] = int(self.timelimitWarmstart/self.warmstartPartitionLinearBinary)
                #self.opt.options['mipgap'] = 1e-6 # not needed atm

            self.results = self.opt.solve(self.m,warmstart=False,tee=True)

            if showSolverOutput == 1:
                print(self.results)

            ## Get results ##
            resultsFile[j] = linear_binary_model.getResults(model=self.m,source=None,savePath="",singleFile=False)
            if timeStepStartPartition == 0:
                pass
            else:
                resultsFile[j].index += (timeStepStartPartition)
            self.optimization_results = pd.concat([self.optimization_results,resultsFile[j]],axis=0) 
            self.optimization_results = self.optimization_results[~self.optimization_results.index.duplicated(keep="last")]

            j = j+1
            timeStepStartPartition = timeStepStartPartition + partitionTimeSteps-1      

        if self.savingWarmstartSystemVals == True:
            try:
                self.result_interface.setOptimizationResults(dataFrame=self.optimization_results,savePath=self.savingPathWarmstartSystemVals)
            except:
                pass

    def getResults(self):
        return self.optimization_results

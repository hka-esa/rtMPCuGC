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

from optimal_control.binary_model import *

class Warmstart_Binary_Model():

    def __init__(self, timelimitWarmstart, warmstartPartitionStepBinary, savingPathWarmstartSystemVals="", savingWarmstartSystemVals=False, sourceSavingSystemVals=None):
        self.timelimitWarmstart = timelimitWarmstart
        self.warmstartPartitionStepBinary = warmstartPartitionStepBinary
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

    def setParams(self, timestepsBinary, stepSizeBinary, controlPeriod1, controlPeriod2, controlPeriodSwitch):
        self.timestepsBinary = timestepsBinary
        self.stepSizeBinary = stepSizeBinary
        self.controlPeriod1 = controlPeriod1
        self.controlPeriod2 = controlPeriod2
        self.controlPeriodSwitch = controlPeriodSwitch

    def setStartValues(self,T_HP_HT_start,T_HP_LT_start,T_HS_start,T_HXA_start,T_HGC_start,T_HGS_start,T_IS_w_1_start,T_IS_w_2_start,T_IS_w_3_start,T_IS_c_1_start,T_IS_c_2_start,T_IS_c_3_start,T_IS_c_4_start,T_IS_c_5_start,T_GS_w_1_start,T_GS_w_2_start,T_GS_w_3_start,T_GS_c_1_start,T_GS_c_2_start,T_GS_c_3_start,T_GS_c_4_start,T_GS_c_5_start,T_GS_c_6_start,T_GS_c_7_start,T_CS_start,T_RLTS_start):
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
        print("Warmstart binary model started") 
        resultsFile = {} 
        j = 0
        controlPeriodSwitchCuted = self.controlPeriodSwitch

        partitionStepsTime = [self.warmstartPartitionStepBinary,(self.timestepsBinary+1-self.warmstartPartitionStepBinary)]
        timeStepStartPartition = 0 

        for partitionTimeSteps in partitionStepsTime:
            print("Optimizing model part " +str(j+1) + " of " + str(len(partitionStepsTime)) + ".")
            if timeStepStartPartition > 0:
                self.m_prev = self.m
            self.m = pyo.ConcreteModel()
            binary_model = Binary_Model()

            if controlPeriodSwitchCuted >= partitionTimeSteps:
                controlSwitch = partitionTimeSteps-1
                controlPeriodSwitchCuted = controlPeriodSwitchCuted - partitionTimeSteps-1
            else:
                controlSwitch = controlPeriodSwitchCuted
                controlPeriodSwitchCuted = 0

            binary_model.setProfiles(profileForecastHeat=self.profileForecastHeat[timeStepStartPartition:timeStepStartPartition+partitionTimeSteps-1],profileForecastCool=self.profileForecastCool[timeStepStartPartition:timeStepStartPartition+partitionTimeSteps-1],profileForecastDry=self.profileForecastDry[timeStepStartPartition:timeStepStartPartition+partitionTimeSteps-1],profileForecastWeather=self.profileForecastWeather[timeStepStartPartition:timeStepStartPartition+partitionTimeSteps-1],profileForecastPrice=self.profileForecastPrice[timeStepStartPartition:timeStepStartPartition+partitionTimeSteps-1],profileForecastFrost=self.profileForecastFrost[timeStepStartPartition:timeStepStartPartition+partitionTimeSteps-1])
            binary_model.setParams(timeSteps=list(range(0,partitionTimeSteps)),stepSizeInSec=self.stepSizeBinary,controlPeriod1=self.controlPeriod1,controlPeriod2=self.controlPeriod2,tControlPeriodSwitch=controlSwitch)
            self.m = binary_model.setVariables(self.m)                                                          

            if timeStepStartPartition == 0:
                self.m = binary_model.setStartValues(model=self.m,T_HP_HT_start=self.T_HP_HT_start,T_HP_LT_start=self.T_HP_LT_start,T_HS_start=self.T_HS_start,
                T_HXA_start=self.T_HXA_start,T_HGC_start=self.T_HGC_start,T_HGS_start=self.T_HGS_start,T_IS_w_1_start=self.T_IS_w_1_start,T_IS_w_2_start=self.T_IS_w_2_start,
                T_IS_w_3_start=self.T_IS_w_3_start,T_IS_c_1_start=self.T_IS_c_1_start,T_IS_c_2_start=self.T_IS_c_2_start,T_IS_c_3_start=self.T_IS_c_3_start,
                T_IS_c_4_start=self.T_IS_c_4_start,T_IS_c_5_start=self.T_IS_c_5_start,T_GS_w_1_start=self.T_GS_w_1_start,
                T_GS_w_2_start=self.T_GS_w_2_start,T_GS_w_3_start=self.T_GS_w_3_start,T_GS_c_1_start=self.T_GS_c_1_start,T_GS_c_2_start=self.T_GS_c_2_start,
                T_GS_c_3_start=self.T_GS_c_3_start,T_GS_c_4_start=self.T_GS_c_4_start,T_GS_c_5_start=self.T_GS_c_5_start,T_GS_c_6_start=self.T_GS_c_6_start,
                T_GS_c_7_start=self.T_GS_c_7_start,T_CS_start=self.T_CS_start,T_RLTS_start=self.T_RLTS_start,Start_Toggle_Constraints=False,B_HP_1_start=False,B_HP_2_start=False,
                B_HP_3_start=False,B_HP_4_start=False,B_HXH_HS_start=False,B_HGC_HGCHXC_start=False,B_HXA_start=False,B_HXH_HGC_start=False,B_HS_IS_start=False,B_IS_HGS_start=False,
                B_GS_HGS_start=False,B_GS_CS_start=False,B_GS_HGS_CS_start=False,B_VP_start=False)
            else:
                self.m = binary_model.setStartValues(model=self.m,T_HP_HT_start=self.m_prev.T_HP_HT_T[partitionStepsTime[j-1]-1](),T_HP_LT_start=self.m_prev.T_HP_LT_T[partitionStepsTime[j-1]-1](),
                T_HS_start=self.m_prev.T_HS_T[partitionStepsTime[j-1]-1](),T_HXA_start=self.m_prev.T_HXA_T[partitionStepsTime[j-1]-1](),T_HGC_start=self.m_prev.T_HGC_T[partitionStepsTime[j-1]-1](),
                T_HGS_start=self.m_prev.T_HGS_T[partitionStepsTime[j-1]-1](),T_IS_w_1_start=self.m_prev.T_IS_W_T_WR[partitionStepsTime[j-1]-1,0](),T_IS_w_2_start=self.m_prev.T_IS_W_T_WR[partitionStepsTime[j-1]-1,2](),
                T_IS_w_3_start=self.m_prev.T_IS_W_T_WR[partitionStepsTime[j-1]-1,4](),T_IS_c_1_start=self.m_prev.T_IS_C_T_CR[partitionStepsTime[j-1]-1,0](),T_IS_c_2_start=self.m_prev.T_IS_C_T_CR[partitionStepsTime[j-1]-1,1](),
                T_IS_c_3_start=self.m_prev.T_IS_C_T_CR[partitionStepsTime[j-1]-1,2](),T_IS_c_4_start=self.m_prev.T_IS_C_T_CR[partitionStepsTime[j-1]-1,3](),T_IS_c_5_start=self.m_prev.T_IS_C_T_CR[partitionStepsTime[j-1]-1,4](),
                T_GS_w_1_start=self.m_prev.T_GS_W_T_WR_WC[partitionStepsTime[j-1]-1,0,1](),T_GS_w_2_start=self.m_prev.T_GS_W_T_WR_WC[partitionStepsTime[j-1]-1,0,3](),T_GS_w_3_start=self.m_prev.T_GS_W_T_WR_WC[partitionStepsTime[j-1]-1,0,5](),
                T_GS_c_1_start=self.m_prev.T_GS_C_T_CR_CC[partitionStepsTime[j-1]-1,0,0](),T_GS_c_2_start=self.m_prev.T_GS_C_T_CR_CC[partitionStepsTime[j-1]-1,0,1](),T_GS_c_3_start=self.m_prev.T_GS_C_T_CR_CC[partitionStepsTime[j-1]-1,0,2](),
                T_GS_c_4_start=self.m_prev.T_GS_C_T_CR_CC[partitionStepsTime[j-1]-1,0,3](),T_GS_c_5_start=self.m_prev.T_GS_C_T_CR_CC[partitionStepsTime[j-1]-1,0,4](),T_GS_c_6_start=self.m_prev.T_GS_C_T_CR_CC[partitionStepsTime[j-1]-1,0,5](),
                T_GS_c_7_start=self.m_prev.T_GS_C_T_CR_CC[partitionStepsTime[j-1]-1,0,6](),T_CS_start=self.m_prev.T_CS_T[partitionStepsTime[j-1]-1](),T_RLTS_start=self.m_prev.T_RLTS_T[partitionStepsTime[j-1]-1](),Start_Toggle_Constraints=False,
                B_HP_1_start=False,B_HP_2_start=False,B_HP_3_start=False,B_HP_4_start=False,B_HXH_HS_start=False,B_HGC_HGCHXC_start=False,B_HXA_start=False,B_HXH_HGC_start=False,B_HS_IS_start=False,B_IS_HGS_start=False,
                B_GS_HGS_start=False,B_GS_CS_start=False,B_GS_HGS_CS_start=False,B_VP_start=False)

            self.m = binary_model.setEndValues(model=self.m,End_Temp_Constraints=False,T_HS_end=0,T_CS_end=0,T_RLTS_end=0,End_Toggle_Constraints=False,B_HP_1_end=False,B_HP_2_end=False,B_HP_3_end=False,B_HP_4_end=False,B_HXH_HS_end=False,B_HGC_HGCHXC_end=False,B_HXA_end=False,B_HXH_HGC_end=False,B_HS_IS_end=False,B_IS_HGS_end=False,B_GS_HGS_end=False,B_GS_CS_end=False,B_GS_HGS_CS_end=False)
            self.m = binary_model.setConstraints(model=self.m)
            self.m = binary_model.setWarmstart(model=self.m,available=False,file=None)
            self.m = binary_model.setObjective(model=self.m)

            if solver == 0:
                self.opt = pyo.SolverFactory('gurobi', solver_io="python")
                self.opt.options['TimeLimit'] = int(self.timelimitWarmstart/2)
                self.opt.options['threads'] = 8
                #self.opt.options['MIPFocus'] = 1
                #self.opt.options['ObjBound'] = 50
                #self.opt.options['Cutoff'] = 500
            elif solver == 1:
                self.opt = pyo.SolverFactory('cbc')
                self.opt.options['Sec'] = int(self.timelimitWarmstart/2)
            elif solver == 2:
                self.opt = pyo.SolverFactory('glpk')
                self.opt.options['tmlim'] = int(self.timelimitWarmstart/2)
                #self.opt.options['mipgap'] = 1e-6 # not needed atm

            self.results = self.opt.solve(self.m,warmstart=False,tee=True)

            if showSolverOutput == 1:
                print(self.results)

            ## Get results ##
            resultsFile[j] = binary_model.getResults(model=self.m,source=None,savePath="",singleFile=False)
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
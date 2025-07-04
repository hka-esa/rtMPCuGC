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

############################ IMPORTS #############################
import os
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
import pyomo.environ as pyo
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import time
##################################################################

#################### OPTIMAL CONTROL IMPORTS #####################
from optimal_control.optimal_control import *
from optimal_control.binary_model import *
from optimal_control.linear_binary_model import *
from optimal_control.long_term_model import *
from optimal_control.forecast_interface import *
from optimal_control.market_interface import *
from optimal_control.measurements_interface import *
from optimal_control.optimization_results_interface import *
from optimal_control.warmstart_binary_model import *
from optimal_control.warmstart_linear_binary_model import *
#from optimal_control.modelica_interface import * !! activate, if modelica model connected
##################################################################

############################ SETTINGS ############################
TIMESTEPS_BINARY = 7 ## +1 for Endtime and always 10 min steps, no matter whitch control period
TIMESTEPS_LINEAR_BINARY = 23   
TIMESTEPS_LONG_TERM = 25

CONTROL_PERIOD_1 = 1
CONTROL_PERIOD_2 = 2
CONTROL_PERIOD_3 = 1
CONTROL_PERIOD_SWITCH = 2

SAVEPATH_MPC= FILE_PATH + "\\optimal_control\\optimization_results\\"
LOADPATH_FORECAST_DEMAND= FILE_PATH + "\\optimal_control\\forecast_values\\all.csv"
LOADPATH_FORECAST_WEATHER= FILE_PATH + "\\optimal_control\\forecast_values\\weather.csv"
LOADPATH_FORECAST_PRICE = FILE_PATH + "\\optimal_control\\forecast_values\\dayaheadprices_2022.csv"
SAVELOADPATH_MEASUREMENTS= FILE_PATH + "\\optimal_control\\optimization_results\\"
SAVEPATH_WARMSTART= FILE_PATH + "\\optimal_control\\warmstart_values\\"
#PACKAGEPATH_MODELICA= FILE_PATH + "XXX\\package.mo" !! activate, if modelica model connected
#MODEL_NAME_MODELICA= FILE_PATH + "XXX.essystem.control" !! activate, if modelica model connected
#OUTPUTPATH_MODELICA= FILE_PATH + "XXX\\results" !! activate, if modelica model connected
#LOADPATH_MODELICA= FILE_PATH + "XXX\\input\\" !! activate, if modelica model connected

#DYM_STARTTIME = "2022-06-14 23:50:00" ## dymola start time is always 10 minutes earlier !! activate, if modelica model connected
SIM_STARTTIME = "2022-06-15 00:00:00" ## This here for MPC, since first iteration (10 minutes) need to be done by dymola/modelica
SIM_ENDTIME = "2022-06-16 00:00:00"
SIM_ENDTIME_PLUS_A_WEEK = "2022-06-30 00:00:00" # for long term model (end time + one week horizon)
SIM_INTERVAL = 600 ## in seconds

PRICE_TYPE = "flat" ## flat or variable

MARKET_ACTIVE = False
MARKET_SIGNAL_STARTTIME = "2022-06-15 00:00:00"
MARKET_SIGNAL_STOPPTIME = "2022-06-15 00:50:00"  ## Always put last interval time here, e.g., 11:50:00 for ending on 12:00:00
FACTOR_MARKET = 3.0
DIRECTION_MARKET = "pos"
TYPE_MARKET = "demandResponse"

TIMELIMIT_SOLVER = 200 ## in seconds
CYCLETIME_LOOP = 240 ## in seconds

WARMSTART = True
TIMELIMIT_WARMSTART = 100 ## in seconds
WARMSTART_PARTITION_STEP_BINARY = 5
WARMSTART_PARTITION_LINEAR_BINARY = 11  

TEN_MINUTES = 600
ONE_HOUR = 3600
SIX_HOURS = 21600
ONE_WEEK_IN_HOURS = 168
##################################################################

############################## CODE ##############################
def setup():
    print("Folder: " + str(FILE_PATH))
    print("")

def loop():
    started = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    optimization_results_interface = Optimization_Results_Interface(source="csv",time="extern",timestamp=started)
    optimization_results_interface_warmstart = Optimization_Results_Interface(source="csv",time="extern",timestamp=started)
    #sim_results_interface = Optimization_Results_Interface(source="csv",time="extern",timestamp=started) !! activate, if modelica model connected
    forecast_interface = Forecast_Interface(source="random",priceType=PRICE_TYPE,loadPathDemand=LOADPATH_FORECAST_DEMAND,loadPathWeather=LOADPATH_FORECAST_WEATHER,loadPathPrice=LOADPATH_FORECAST_PRICE)
    measurements_interface = Measurements_Interface(source="sim",loadPathMeasurements=SAVELOADPATH_MEASUREMENTS,time="extern",timestamp=started)
    market_interface = Market_Interface(type=TYPE_MARKET, directionSignal=DIRECTION_MARKET, timestampSignalStart=MARKET_SIGNAL_STARTTIME, timestampSignalStop=MARKET_SIGNAL_STOPPTIME, factorSignal=FACTOR_MARKET, simTimeStart=SIM_STARTTIME, simTimeStop=SIM_ENDTIME_PLUS_A_WEEK, intervalInSec=SIM_INTERVAL)

    #modelica_interface = Modelica_Interface(simTimeStart=DYM_STARTTIME,simTimeStop=SIM_ENDTIME,packagePath=PACKAGEPATH_MODELICA, modelName=MODEL_NAME_MODELICA,simOutputPath=OUTPUTPATH_MODELICA,loadPathDemandsWeatherSIM=LOADPATH_MODELICA,loadPathDemandsMPC=LOADPATH_FORECAST_DEMAND,loadPathWeatherMPC=LOADPATH_FORECAST_WEATHER) !! activate, if modelica model connected
    #modelica_interface.setParams(stepSizeInSec=SIM_INTERVAL) !! activate, if modelica model connected
    #modelica_interface.runInitialSimulation() !! activate, if modelica model connected
    #modelica_results = modelica_interface.getResults() !! activate, if modelica model connected
    #sim_results_interface.setOptimizationResults(dataFrame=modelica_results,savePath=SAVELOADPATH_MEASUREMENTS) !! activate, if modelica model connected

    timestampSimEndtime = datetime.strptime(SIM_ENDTIME,"%Y-%m-%d %H:%M:%S")
    timestampSim = datetime.strptime(SIM_STARTTIME,"%Y-%m-%d %H:%M:%S")

    i_loop = 0

    while timestampSim < timestampSimEndtime:
        timestampStartLoop = datetime.now()

        forecast_data = forecast_interface.getProfilesAll(timestampStart=timestampSim, intervals=[TEN_MINUTES] * (TIMESTEPS_BINARY-1) + [ONE_HOUR] * (TIMESTEPS_LINEAR_BINARY-1) + [SIX_HOURS] * (TIMESTEPS_LONG_TERM-1), periodFrostInHours=ONE_WEEK_IN_HOURS)
        measurements_data = measurements_interface.getMeasurementsAll()
        if MARKET_ACTIVE == True:
            market_data = market_interface.getProfileForecastMarket(timestampStart=timestampSim, intervals=[TEN_MINUTES] * (TIMESTEPS_BINARY-1) + [ONE_HOUR] * (TIMESTEPS_LINEAR_BINARY-1) + [SIX_HOURS] * (TIMESTEPS_LONG_TERM-1))
            profile_forecast_price = np.array(forecast_data["profileForecastPrice"]) + (np.array(market_data) * np.array(forecast_data["profileForecastPrice"]))
            profile_forecast_price = profile_forecast_price.tolist()
        else:
            profile_forecast_price = forecast_data["profileForecastPrice"]

        optimal_control = Optimal_Control()
        binary_model = Binary_Model()
        linear_binary_model = Linear_Binary_Model()
        long_term_model = Long_Term_Model()
        warmstart_binary_model = Warmstart_Binary_Model(timelimitWarmstart=TIMELIMIT_WARMSTART, warmstartPartitionStepBinary=WARMSTART_PARTITION_STEP_BINARY, savingPathWarmstartSystemVals=SAVEPATH_WARMSTART, savingWarmstartSystemVals=True, sourceSavingSystemVals=optimization_results_interface_warmstart)
        warmstart_linear_binary_model = Warmstart_Linear_Binary_Model(timelimitWarmstart=TIMELIMIT_WARMSTART, warmstartPartitionLinearBinary=WARMSTART_PARTITION_LINEAR_BINARY, savingPathWarmstartSystemVals=SAVEPATH_WARMSTART, savingWarmstartSystemVals=True, sourceSavingSystemVals=optimization_results_interface_warmstart)

        binary_model.setProfiles(profileForecastHeat=forecast_data["profileForecastHeat"][:TIMESTEPS_BINARY-1],profileForecastCool=forecast_data["profileForecastCool"][:TIMESTEPS_BINARY-1],profileForecastDry=forecast_data["profileForecastDry"][:TIMESTEPS_BINARY-1],profileForecastWeather=forecast_data["profileForecastWeather"][:TIMESTEPS_BINARY-1],profileForecastPrice=profile_forecast_price[:TIMESTEPS_BINARY-1],profileForecastFrost=forecast_data["profileForecastFrost"][:TIMESTEPS_BINARY-1])
        linear_binary_model.setProfiles(profileForecastHeat=forecast_data["profileForecastHeat"][TIMESTEPS_BINARY-1:TIMESTEPS_BINARY+TIMESTEPS_LINEAR_BINARY-2],profileForecastCool=forecast_data["profileForecastCool"][TIMESTEPS_BINARY-1:TIMESTEPS_BINARY+TIMESTEPS_LINEAR_BINARY-2],profileForecastDry=forecast_data["profileForecastDry"][TIMESTEPS_BINARY-1:TIMESTEPS_BINARY+TIMESTEPS_LINEAR_BINARY-2],profileForecastWeather=forecast_data["profileForecastWeather"][TIMESTEPS_BINARY-1:TIMESTEPS_BINARY+TIMESTEPS_LINEAR_BINARY-2],profileForecastPrice=profile_forecast_price[TIMESTEPS_BINARY-1:TIMESTEPS_BINARY+TIMESTEPS_LINEAR_BINARY-2],profileForecastFrost=forecast_data["profileForecastFrost"][TIMESTEPS_BINARY-1:TIMESTEPS_BINARY+TIMESTEPS_LINEAR_BINARY-2])
        long_term_model.setProfiles(profileForecastHeat=forecast_data["profileForecastHeat"][TIMESTEPS_BINARY+TIMESTEPS_LINEAR_BINARY-2:TIMESTEPS_BINARY+TIMESTEPS_LINEAR_BINARY+TIMESTEPS_LONG_TERM-3],profileForecastPrice=profile_forecast_price[TIMESTEPS_BINARY+TIMESTEPS_LINEAR_BINARY-2:TIMESTEPS_BINARY+TIMESTEPS_LINEAR_BINARY+TIMESTEPS_LONG_TERM-3],forecastFrost=forecast_data["forecastFrost"])

        binary_model.setParams(timeSteps=list(range(0,TIMESTEPS_BINARY)),stepSizeInSec=TEN_MINUTES,controlPeriod1=CONTROL_PERIOD_1,controlPeriod2=CONTROL_PERIOD_2,tControlPeriodSwitch=CONTROL_PERIOD_SWITCH)
        linear_binary_model.setParams(timeSteps=list(range(0,TIMESTEPS_LINEAR_BINARY)),stepSizeInSec=ONE_HOUR,controlPeriod=CONTROL_PERIOD_3,NMcCormick=list(range(0,2)))
        long_term_model.setParams(timeSteps=list(range(0,TIMESTEPS_LONG_TERM)),stepSizeInSec=SIX_HOURS)

        optimal_control.addModelParts(model = binary_model.setVariables(optimal_control.getModel()))
        optimal_control.addModelParts(model = linear_binary_model.setVariables(optimal_control.getModel()))
        optimal_control.addModelParts(model = long_term_model.setVariables(optimal_control.getModel()))

        optimal_control.addModelParts(binary_model.setStartValues(model=optimal_control.getModel(),T_HP_HT_start=measurements_data["measurementHP_HT"],T_HP_LT_start=measurements_data["measurementHP_LT"],T_HS_start=measurements_data["measurementHS"],
        T_HXA_start=measurements_data["measurementHXA"],T_HGC_start=measurements_data["measurementHGC"],T_HGS_start=measurements_data["measurementHGS"],T_IS_w_1_start=measurements_data["measurementISw"],T_IS_w_2_start=measurements_data["measurementISw"],
        T_IS_w_3_start=measurements_data["measurementISw"],T_IS_c_1_start=measurements_data["measurementISwc"],T_IS_c_2_start=measurements_data["measurementISc"],T_IS_c_3_start=measurements_data["measurementISwc"],
        T_IS_c_4_start=measurements_data["measurementISc"],T_IS_c_5_start=measurements_data["measurementISwc"],T_GS_w_1_start=measurements_data["measurementGSw"],
        T_GS_w_2_start=measurements_data["measurementGSw"],T_GS_w_3_start=measurements_data["measurementGSw"],T_GS_c_1_start=measurements_data["measurementGSc"],T_GS_c_2_start=measurements_data["measurementGSwc"],
        T_GS_c_3_start=measurements_data["measurementGSc"],T_GS_c_4_start=measurements_data["measurementGSwc"],T_GS_c_5_start=measurements_data["measurementGSc"],T_GS_c_6_start=measurements_data["measurementGSwc"],
        T_GS_c_7_start=measurements_data["measurementGSc"],T_CS_start=measurements_data["measurementCS"],T_RLTS_start=measurements_data["measurementRLTS"],
        Start_Toggle_Constraints=True,B_HP_1_start=measurements_data["measurementHP"][0],B_HP_2_start=measurements_data["measurementHP"][1],B_HP_3_start=measurements_data["measurementHP"][2],
        B_HP_4_start=measurements_data["measurementHP"][3],B_HXH_HS_start=measurements_data["measurementHXH_HS"],B_HGC_HGCHXC_start=measurements_data["measurementHGC_HGCHXC"],B_HXA_start=measurements_data["measurementHXAb"],
        B_HXH_HGC_start=measurements_data["measurementHXH_HGC"],B_HS_IS_start=measurements_data["measurementHS_IS"],B_IS_HGS_start=measurements_data["measurementIS_HGS"],B_GS_HGS_start=measurements_data["measurementGS_HGS"],
        B_GS_CS_start=measurements_data["measurementGS_CS"],B_GS_HGS_CS_start=measurements_data["measurementGS_HGS_CS"],B_VP_start=measurements_data["measurementVP"]))
        optimal_control.addModelParts(linear_binary_model.setStartValues(model=optimal_control.getModel(),T_HP_HT_start=optimal_control.m.T_HP_HT_T[TIMESTEPS_BINARY-1],T_HP_LT_start=optimal_control.m.T_HP_LT_T[TIMESTEPS_BINARY-1],
        T_HS_start=optimal_control.m.T_HS_T[TIMESTEPS_BINARY-1],T_HXA_start=optimal_control.m.T_HXA_T[TIMESTEPS_BINARY-1],T_HXH_start=optimal_control.m.T_HP_HT_T[TIMESTEPS_BINARY-1],T_HGC_start=optimal_control.m.T_HGC_T[TIMESTEPS_BINARY-1],
        T_HXC_start=optimal_control.m.T_HP_LT_T[TIMESTEPS_BINARY-1],T_HGS_start=optimal_control.m.T_HGS_T[TIMESTEPS_BINARY-1],T_IS_w_1_start=optimal_control.m.T_IS_W_T_WR[TIMESTEPS_BINARY-1,0],T_IS_w_2_start=optimal_control.m.T_IS_W_T_WR[TIMESTEPS_BINARY-1,2],
        T_IS_w_3_start=optimal_control.m.T_IS_W_T_WR[TIMESTEPS_BINARY-1,4],T_IS_c_1_start=optimal_control.m.T_IS_C_T_CR[TIMESTEPS_BINARY-1,0],T_IS_c_2_start=optimal_control.m.T_IS_C_T_CR[TIMESTEPS_BINARY-1,1],
        T_IS_c_3_start=optimal_control.m.T_IS_C_T_CR[TIMESTEPS_BINARY-1,2],T_IS_c_4_start=optimal_control.m.T_IS_C_T_CR[TIMESTEPS_BINARY-1,3],T_IS_c_5_start=optimal_control.m.T_IS_C_T_CR[TIMESTEPS_BINARY-1,4],
        T_GS_w_1_start=optimal_control.m.T_GS_W_T_WR_WC[TIMESTEPS_BINARY-1,0,1],T_GS_w_2_start=optimal_control.m.T_GS_W_T_WR_WC[TIMESTEPS_BINARY-1,0,3],T_GS_w_3_start=optimal_control.m.T_GS_W_T_WR_WC[TIMESTEPS_BINARY-1,0,5],
        T_GS_c_1_start=optimal_control.m.T_GS_C_T_CR_CC[TIMESTEPS_BINARY-1,0,0],T_GS_c_2_start=optimal_control.m.T_GS_C_T_CR_CC[TIMESTEPS_BINARY-1,0,1],T_GS_c_3_start=optimal_control.m.T_GS_C_T_CR_CC[TIMESTEPS_BINARY-1,0,2],
        T_GS_c_4_start=optimal_control.m.T_GS_C_T_CR_CC[TIMESTEPS_BINARY-1,0,3],T_GS_c_5_start=optimal_control.m.T_GS_C_T_CR_CC[TIMESTEPS_BINARY-1,0,4],T_GS_c_6_start=optimal_control.m.T_GS_C_T_CR_CC[TIMESTEPS_BINARY-1,0,5],
        T_GS_c_7_start=optimal_control.m.T_GS_C_T_CR_CC[TIMESTEPS_BINARY-1,0,6],T_CS_start=optimal_control.m.T_CS_T[TIMESTEPS_BINARY-1],T_RLTS_start=optimal_control.m.T_RLTS_T[TIMESTEPS_BINARY-1]))
        optimal_control.addModelParts(long_term_model.setStartValues(model=optimal_control.getModel(),T_HS_start=optimal_control.m.T_HS_I[TIMESTEPS_LINEAR_BINARY-1],T_GS_w_1_start=optimal_control.m.T_GS_W_I_WR_WC[TIMESTEPS_LINEAR_BINARY-1,0,1],
        T_GS_w_2_start=optimal_control.m.T_GS_W_I_WR_WC[TIMESTEPS_LINEAR_BINARY-1,0,3],T_GS_w_3_start=optimal_control.m.T_GS_W_I_WR_WC[TIMESTEPS_LINEAR_BINARY-1,0,5],T_GS_c_1_start=optimal_control.m.T_GS_C_I_CR_CC[TIMESTEPS_LINEAR_BINARY-1,0,0],
        T_GS_c_2_start=optimal_control.m.T_GS_C_I_CR_CC[TIMESTEPS_LINEAR_BINARY-1,0,1],T_GS_c_3_start=optimal_control.m.T_GS_C_I_CR_CC[TIMESTEPS_LINEAR_BINARY-1,0,2],T_GS_c_4_start=optimal_control.m.T_GS_C_I_CR_CC[TIMESTEPS_LINEAR_BINARY-1,0,3],
        T_GS_c_5_start=optimal_control.m.T_GS_C_I_CR_CC[TIMESTEPS_LINEAR_BINARY-1,0,4],T_GS_c_6_start=optimal_control.m.T_GS_C_I_CR_CC[TIMESTEPS_LINEAR_BINARY-1,0,5],T_GS_c_7_start=optimal_control.m.T_GS_C_I_CR_CC[TIMESTEPS_LINEAR_BINARY-1,0,6]))

        optimal_control.addModelParts(binary_model.setEndValues(model=optimal_control.getModel(),End_Temp_Constraints=False,T_HS_end=0,T_CS_end=0,T_RLTS_end=0,End_Toggle_Constraints=True,B_HP_1_end=optimal_control.m.B_HP_H_I[1,0],B_HP_2_end=optimal_control.m.B_HP_H_I[2,0],B_HP_3_end=optimal_control.m.B_HP_H_I[3,0],B_HP_4_end=optimal_control.m.B_HP_H_I[4,0],B_HXH_HS_end=optimal_control.m.V_HP_HXH_I[0],
        B_HGC_HGCHXC_end=optimal_control.m.V_HP_HGCHXC_I[0],B_HXA_end=optimal_control.m.P_HXA_I[0],B_HXH_HGC_end=optimal_control.m.V_HXA_HXH_I[0],B_HS_IS_end=optimal_control.m.V_HS_IS_I[0],B_IS_HGS_end=optimal_control.m.V_IS_HGS_I[0],B_GS_HGS_end=optimal_control.m.V_GS_HGS_I[0],B_GS_CS_end=optimal_control.m.V_GS_CS_I[0],B_GS_HGS_CS_end=optimal_control.m.V_GS_HGS_I[0]+optimal_control.m.V_GS_CS_I[0]))
        optimal_control.addModelParts(linear_binary_model.setEndValues(model=optimal_control.getModel(),End_Temp_Constraints=False,T_HS_end=(40+33)/2,T_CS_end=(18+10)/2,T_RLTS_end=(18+6)/2,End_Toggle_Constraints=False,B_HP_1_end=0,B_HP_2_end=0,B_HP_3_end=0,B_HP_4_end=0,V_HP_HXH_end=0,V_HP_HS_end=0,V_HP_HGC_end=0,V_HGCHXC_end=0,V_HXA_end=0,V_HXA_HXH_end=0,V_HS_IS_end=0,V_IS_HGS_end=0,V_HXA_HGC_end=0,V_GS_HGS_end=0,V_GS_CS_end=0))
        optimal_control.addModelParts(long_term_model.setEndValues(model=optimal_control.getModel(),End_Temp_Constraints=False))

        optimal_control.addModelParts(binary_model.setConstraints(model=optimal_control.getModel()))
        optimal_control.addModelParts(linear_binary_model.setConstraints(model=optimal_control.getModel()))
        optimal_control.addModelParts(long_term_model.setConstraints(model=optimal_control.getModel()))
         
        if WARMSTART == True:
            if i_loop > 0:
                old_warmstart_binary_model_results = warmstart_binary_model_results
            try:
                warmstart_binary_model.setProfiles(profileForecastHeat=forecast_data["profileForecastHeat"][:TIMESTEPS_BINARY-1],profileForecastCool=forecast_data["profileForecastCool"][:TIMESTEPS_BINARY-1],profileForecastDry=forecast_data["profileForecastDry"][:TIMESTEPS_BINARY-1],profileForecastWeather=forecast_data["profileForecastWeather"][:TIMESTEPS_BINARY-1],profileForecastPrice=profile_forecast_price[:TIMESTEPS_BINARY-1],profileForecastFrost=forecast_data["profileForecastFrost"][:TIMESTEPS_BINARY-1])
                warmstart_binary_model.setParams(timestepsBinary=TIMESTEPS_BINARY,stepSizeBinary=TEN_MINUTES,controlPeriod1=CONTROL_PERIOD_1,controlPeriod2=CONTROL_PERIOD_2,controlPeriodSwitch=CONTROL_PERIOD_SWITCH)
                warmstart_binary_model.setStartValues(T_HP_HT_start=measurements_data["measurementHP_HT"],T_HP_LT_start=measurements_data["measurementHP_LT"],T_HS_start=measurements_data["measurementHS"],
                T_HXA_start=measurements_data["measurementHXA"],T_HGC_start=measurements_data["measurementHGC"],T_HGS_start=measurements_data["measurementHGS"],T_IS_w_1_start=measurements_data["measurementISw"],T_IS_w_2_start=measurements_data["measurementISw"],
                T_IS_w_3_start=measurements_data["measurementISw"],T_IS_c_1_start=measurements_data["measurementISwc"],T_IS_c_2_start=measurements_data["measurementISc"],T_IS_c_3_start=measurements_data["measurementISwc"],
                T_IS_c_4_start=measurements_data["measurementISc"],T_IS_c_5_start=measurements_data["measurementISwc"],T_GS_w_1_start=measurements_data["measurementGSw"],
                T_GS_w_2_start=measurements_data["measurementGSw"],T_GS_w_3_start=measurements_data["measurementGSw"],T_GS_c_1_start=measurements_data["measurementGSc"],T_GS_c_2_start=measurements_data["measurementGSwc"],
                T_GS_c_3_start=measurements_data["measurementGSc"],T_GS_c_4_start=measurements_data["measurementGSwc"],T_GS_c_5_start=measurements_data["measurementGSc"],T_GS_c_6_start=measurements_data["measurementGSwc"],
                T_GS_c_7_start=measurements_data["measurementGSc"],T_CS_start=measurements_data["measurementCS"],T_RLTS_start=measurements_data["measurementRLTS"])
                warmstart_binary_model.runWarmstart()
                warmstart_binary_model_results = warmstart_binary_model.getResults()
            except:
                warmstart_binary_model_results = old_warmstart_binary_model_results
        else:
            warmstart_binary_model_results = None
        
        if WARMSTART == True:
            if i_loop > 0:
                old_warmstart_linear_binary_model_results = warmstart_linear_binary_model_results
            try:
                warmstart_linear_binary_model.setProfiles(profileForecastHeat=forecast_data["profileForecastHeat"][TIMESTEPS_BINARY-1:TIMESTEPS_BINARY+TIMESTEPS_LINEAR_BINARY-2],profileForecastCool=forecast_data["profileForecastCool"][TIMESTEPS_BINARY-1:TIMESTEPS_BINARY+TIMESTEPS_LINEAR_BINARY-2],profileForecastDry=forecast_data["profileForecastDry"][TIMESTEPS_BINARY-1:TIMESTEPS_BINARY+TIMESTEPS_LINEAR_BINARY-2],profileForecastWeather=forecast_data["profileForecastWeather"][TIMESTEPS_BINARY-1:TIMESTEPS_BINARY+TIMESTEPS_LINEAR_BINARY-2],profileForecastPrice=profile_forecast_price[TIMESTEPS_BINARY-1:TIMESTEPS_BINARY+TIMESTEPS_LINEAR_BINARY-2],profileForecastFrost=forecast_data["profileForecastFrost"][TIMESTEPS_BINARY-1:TIMESTEPS_BINARY+TIMESTEPS_LINEAR_BINARY-2])
                warmstart_linear_binary_model.setParams(timestepsLinearBinary=TIMESTEPS_LINEAR_BINARY,stepSizeLinearBinary=ONE_HOUR,controlPeriod=CONTROL_PERIOD_3,NMcCormick=list(range(0,2)))
                warmstart_linear_binary_model.setStartValues(T_HP_HT_start=warmstart_binary_model_results["T_HP_HT_T"].iloc[-1],T_HP_LT_start=warmstart_binary_model_results["T_HP_LT_T"].iloc[-1],
                T_HS_start=warmstart_binary_model_results["T_HS_T"].iloc[-1],T_HXA_start=warmstart_binary_model_results["T_HXA_T"].iloc[-1],T_HXH_start=warmstart_binary_model_results["T_HP_HT_T"].iloc[-1],T_HGC_start=warmstart_binary_model_results["T_HGC_T"].iloc[-1],
                T_HXC_start=warmstart_binary_model_results["T_HP_LT_T"].iloc[-1],T_HGS_start=warmstart_binary_model_results["T_HGS_T"].iloc[-1],T_IS_w_1_start=warmstart_binary_model_results["T_IS_W_0_T"].iloc[-1],T_IS_w_2_start=warmstart_binary_model_results["T_IS_W_1_T"].iloc[-1],
                T_IS_w_3_start=warmstart_binary_model_results["T_IS_W_2_T"].iloc[-1],T_IS_c_1_start=warmstart_binary_model_results["T_IS_C_0_T"].iloc[-1],T_IS_c_2_start=warmstart_binary_model_results["T_IS_C_1_T"].iloc[-1],
                T_IS_c_3_start=warmstart_binary_model_results["T_IS_C_2_T"].iloc[-1],T_IS_c_4_start=warmstart_binary_model_results["T_IS_C_3_T"].iloc[-1],T_IS_c_5_start=warmstart_binary_model_results["T_IS_C_4_T"].iloc[-1],
                T_GS_w_1_start=warmstart_binary_model_results["T_GS_W_0_T"].iloc[-1],T_GS_w_2_start=warmstart_binary_model_results["T_GS_W_1_T"].iloc[-1],T_GS_w_3_start=warmstart_binary_model_results["T_GS_W_2_T"].iloc[-1],
                T_GS_c_1_start=warmstart_binary_model_results["T_GS_C_0_T"].iloc[-1],T_GS_c_2_start=warmstart_binary_model_results["T_GS_C_1_T"].iloc[-1],T_GS_c_3_start=warmstart_binary_model_results["T_GS_C_2_T"].iloc[-1],
                T_GS_c_4_start=warmstart_binary_model_results["T_GS_C_3_T"].iloc[-1],T_GS_c_5_start=warmstart_binary_model_results["T_GS_C_4_T"].iloc[-1],T_GS_c_6_start=warmstart_binary_model_results["T_GS_C_5_T"].iloc[-1],
                T_GS_c_7_start=warmstart_binary_model_results["T_GS_C_6_T"].iloc[-1],T_CS_start=warmstart_binary_model_results["T_CS_T"].iloc[-1],T_RLTS_start=warmstart_binary_model_results["T_RLTS_T"].iloc[-1])
                warmstart_linear_binary_model.runWarmstart()
                warmstart_linear_binary_model_results = warmstart_linear_binary_model.getResults()
            except:
                warmstart_linear_binary_model_results = old_warmstart_linear_binary_model_results
        else:
            warmstart_linear_binary_model_results = None

        optimal_control.addModelParts(binary_model.setWarmstart(model=optimal_control.getModel(),available=WARMSTART,file=warmstart_binary_model_results))
        optimal_control.addModelParts(linear_binary_model.setWarmstart(model=optimal_control.getModel(),available=WARMSTART,file=warmstart_linear_binary_model_results))
        optimal_control.addModelParts(long_term_model.setWarmstart(model=optimal_control.getModel()))

        optimal_control.addModelObject(object=binary_model,position=0,symbol="T")
        optimal_control.addModelObject(object=linear_binary_model,position=1,symbol="I")
        optimal_control.addModelObject(object=long_term_model,position=2,symbol="J")

        optimal_control.setObjective()

        optimal_control.setSolverAndRunOptimization(solver=0,warmstart=WARMSTART, timeLimit=TIMELIMIT_SOLVER, showSolverOutput=0,writeILP=0,writeMPSfile=0)

        if i_loop > 0: 
            old_results_optimal_control = results_optimal_control

        try:
            if forecast_data["forecastFrost"] == False:
                results_optimal_control = optimal_control.getResults(source=optimization_results_interface,savePath=SAVEPATH_MPC,combinedFile=True,singleFile=False,timestampStart=timestampSim,intervals=[TEN_MINUTES] * TIMESTEPS_BINARY + [ONE_HOUR] * (TIMESTEPS_LINEAR_BINARY-1))
            else:
                results_optimal_control = optimal_control.getResults(source=optimization_results_interface,savePath=SAVEPATH_MPC,combinedFile=True,singleFile=False,timestampStart=timestampSim,intervals=[TEN_MINUTES] * TIMESTEPS_BINARY + [ONE_HOUR] * (TIMESTEPS_LINEAR_BINARY-1) + [SIX_HOURS] * (TIMESTEPS_LONG_TERM-1))

            #modelica_interface.runSimulation(B_HP_0=results_optimal_control["B_HP_0_T"].iloc[0],B_HP_1=results_optimal_control["B_HP_1_T"].iloc[0],B_HP_2=results_optimal_control["B_HP_2_T"].iloc[0],B_HP_3=results_optimal_control["B_HP_3_T"].iloc[0],B_HP_4=results_optimal_control["B_HP_4_T"].iloc[0],B_HXH_HS=results_optimal_control["B_HXH_HS_T"].iloc[0],
            #B_HGC_HGCHXC=results_optimal_control["B_HGC_HGCHXC_T"].iloc[0],B_HXA=results_optimal_control["B_HXA_T"].iloc[0],B_HXH_HGC=results_optimal_control["B_HXH_HGC_T"].iloc[0],B_HS_IS=results_optimal_control["B_HS_IS_T"].iloc[0],B_IS_HGS=results_optimal_control["B_IS_HGS_T"].iloc[0],B_GS_HGS=results_optimal_control["B_GS_HGS_T"].iloc[0],
            #B_GS_CS=results_optimal_control["B_GS_CS_T"].iloc[0],B_GS_HGS_CS=results_optimal_control["B_GS_HGS_CS_T"].iloc[0],B_VP_0=results_optimal_control["B_VP_0_T_1"].iloc[0],B_VP_1=results_optimal_control["B_VP_1_T_1"].iloc[0],B_VP_2=results_optimal_control["B_VP_2_T_1"].iloc[0],B_VP_3=results_optimal_control["B_VP_3_T_1"].iloc[0],
            #B_VP_4=results_optimal_control["B_VP_4_T_1"].iloc[0],B_VP_5=results_optimal_control["B_VP_5_T_1"].iloc[0],B_VP_6=results_optimal_control["B_VP_6_T_1"].iloc[0],B_VP_7=results_optimal_control["B_VP_7_T_1"].iloc[0]) !! activate, if modelica model connected
        except:
            pass
            #results_optimal_control = old_results_optimal_control
            #modelica_interface.runSimulation(B_HP_0=results_optimal_control["B_HP_0_T"].iloc[1],B_HP_1=results_optimal_control["B_HP_1_T"].iloc[1],B_HP_2=results_optimal_control["B_HP_2_T"].iloc[1],B_HP_3=results_optimal_control["B_HP_3_T"].iloc[1],B_HP_4=results_optimal_control["B_HP_4_T"].iloc[1],B_HXH_HS=results_optimal_control["B_HXH_HS_T"].iloc[1],
            #B_HGC_HGCHXC=results_optimal_control["B_HGC_HGCHXC_T"].iloc[1],B_HXA=results_optimal_control["B_HXA_T"].iloc[1],B_HXH_HGC=results_optimal_control["B_HXH_HGC_T"].iloc[1],B_HS_IS=results_optimal_control["B_HS_IS_T"].iloc[1],B_IS_HGS=results_optimal_control["B_IS_HGS_T"].iloc[1],B_GS_HGS=results_optimal_control["B_GS_HGS_T"].iloc[1],
            #B_GS_CS=results_optimal_control["B_GS_CS_T"].iloc[1],B_GS_HGS_CS=results_optimal_control["B_GS_HGS_CS_T"].iloc[1],B_VP_0=results_optimal_control["B_VP_0_T_1"].iloc[1],B_VP_1=results_optimal_control["B_VP_1_T_1"].iloc[1],B_VP_2=results_optimal_control["B_VP_2_T_1"].iloc[1],B_VP_3=results_optimal_control["B_VP_3_T_1"].iloc[1],
            #B_VP_4=results_optimal_control["B_VP_4_T_1"].iloc[1],B_VP_5=results_optimal_control["B_VP_5_T_1"].iloc[1],B_VP_6=results_optimal_control["B_VP_6_T_1"].iloc[1],B_VP_7=results_optimal_control["B_VP_7_T_1"].iloc[1]) !! activate, if modelica model connected
            #optimization_results_interface.setOptimizationResults(dataFrame=pd.DataFrame(),savePath=SAVEPATH_MPC) !! activate, if modelica model connected

        #modelica_results = modelica_interface.getResults() !! activate, if modelica model connected
        #sim_results_interface.setOptimizationResults(dataFrame=modelica_results,savePath=SAVELOADPATH_MEASUREMENTS) !! activate, if modelica model connected
        timestampSim = timestampSim + timedelta(seconds=SIM_INTERVAL)
        i_loop=i_loop+1
        timestampStopLoop = datetime.now()
        timeDeltaLoop = timestampStopLoop - timestampStartLoop
        if timeDeltaLoop.total_seconds() > CYCLETIME_LOOP:
            timeDeltaLoop = timedelta(seconds=CYCLETIME_LOOP)
        print("### Done ! ###")
        print("Sleeping for " +str(round(CYCLETIME_LOOP-timeDeltaLoop.total_seconds(),2)) +" seconds. Good night!")
        time.sleep(CYCLETIME_LOOP-timeDeltaLoop.total_seconds())
##################################################################
if __name__ == "__main__":
    setup()
    loop()

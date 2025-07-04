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

from dymola.dymola_interface import DymolaInterface
from dymola.dymola_exception import DymolaException
from datetime import datetime
import pandas as pd
import os
import re

class Modelica_Interface():

    def __init__(self, simTimeStart="", simTimeStop="", packagePath="", modelName="",simOutputPath="",loadPathDemandsWeatherSIM="",loadPathDemandsMPC="",loadPathWeatherMPC=""):
        self.started = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.packagePath = packagePath
        self.modelName = modelName
        self.simOutputPath = simOutputPath

        # Make demand and weather files for dymola
        demands = pd.read_csv(loadPathDemandsMPC,index_col=0)
        weather = pd.read_csv(loadPathWeatherMPC,index_col=0)
        demands = demands.loc[simTimeStart:simTimeStop,]
        weather = weather.loc[simTimeStart:simTimeStop,]
        demands.index = range(0,len(demands.index)*120,120)
        weather.index = range(0,len(weather.index)*600,600)

        Qdot_load_cold = demands["Q_HP_Last_Kältespeicher_NEW"]
        Qdot_load_cold.rename("Qdot_load_cold",inplace=True)
        Qdot_load_dehumid = demands["Q_HP_Last_Pufferspeicher_NEW"]
        Qdot_load_dehumid.rename("Qdot_load_dehumid",inplace=True)
        Qdot_load_hot = demands["Q_HP_Last_Waerme_NEW"]
        Qdot_load_hot.rename("Qdot_load_hot",inplace=True)
        T_amb = weather["TT_10"]
        T_amb.rename("T_amb",inplace=True)

        Qdot_load_cold_txt = "#1\ndouble Qdot_load_cold(" +str(len(Qdot_load_cold)) +",2)\n" + Qdot_load_cold.to_string()
        Qdot_load_cold_txt = re.sub(' +',' ',Qdot_load_cold_txt)

        Qdot_load_dehumid_txt = "#1\ndouble Qdot_load_dehumid(" +str(len(Qdot_load_dehumid)) +",2)\n" + Qdot_load_dehumid.to_string()
        Qdot_load_dehumid_txt = re.sub(' +',' ',Qdot_load_dehumid_txt)

        Qdot_load_hot_txt = "#1\ndouble Qdot_load_hot(" +str(len(Qdot_load_hot)) +",2)\n" + Qdot_load_hot.to_string()
        Qdot_load_hot_txt = re.sub(' +',' ',Qdot_load_hot_txt)

        T_amb_txt = "#1\ndouble T_amb(" +str(len(T_amb)) +",2)\n" + T_amb.to_string()
        T_amb_txt = re.sub(' +',' ',T_amb_txt)

        f = open(loadPathDemandsWeatherSIM + "Qdot_load_cold.txt","w")
        f.write(Qdot_load_cold_txt)
        f.close()

        f = open(loadPathDemandsWeatherSIM + "Qdot_load_dehumid.txt","w")
        f.write(Qdot_load_dehumid_txt)
        f.close()

        f = open(loadPathDemandsWeatherSIM + "Qdot_load_hot.txt","w")
        f.write(Qdot_load_hot_txt)
        f.close()

        f = open(loadPathDemandsWeatherSIM + "T_amb.txt","w")
        f.write(T_amb_txt)
        f.close()
        ##End File preperation

        self.simulator = DymolaInterface()
        self.simulator.openModel(self.packagePath)
        self.simulator.translateModel(self.modelName)

    def setParams(self, stepSizeInSec=60.0):
        self.delta_t = stepSizeInSec
        self.t_start = 0
        self.t_end = self.delta_t
        self.iter_count = 0
        self.inputKeys = ["HP_mode_ext","CHS_mode_ext","GS_mode_ext","ST_mode_ext","HS_mode_ext","CS_mode_ext","AS_mode_ext","ASC_mode_ext"]
        self.initialInputValues = [1, 0, 0, 6, 1, 1, 1, 1]

        self.outputKeysModel = ["HP_mode_ext", "CHS_mode_ext", "GS_mode_ext", "ST_mode_ext", "HS_mode_ext", "CS_mode_ext", "AS_mode_ext", "ASC_mode_ext",
                            "T_hts", "T_lts", "T_lts_dehum","T_amb","T_gs_w","T_gs_c","T_gs_wc","T_chs_w","T_chs_c","T_chs_wc",
                            "T_rc", "T_hx_c", "T_hx_h", "T_hp_c_in","T_hp_c_out","T_hp_h_in","T_hp_h_out","T_header_rc","T_header_gs",
                            "Qdot_hp_ht", "Qdot_hp_lt", "P_hp_el", "Qdot_hts_dem_MW", "Qdot_lts_dem_MW", "Qdot_lts_dh_dem_MW"]

        self.sim_results = {}

    def runInitialSimulation(self):
        self.__runModelicaOnce(keys=self.inputKeys,
                             values=self.initialInputValues)

    def runSimulation(self,B_HP_0=1,B_HP_1=0,B_HP_2=0,B_HP_3=0,B_HP_4=0,B_HXH_HS=0,B_HGC_HGCHXC=0,B_HXA=0,B_HXH_HGC=0,B_HS_IS=0,B_IS_HGS=0,B_GS_HGS=0,B_GS_CS=0,B_GS_HGS_CS=0,
                        B_VP_0=1,B_VP_1=0,B_VP_2=0,B_VP_3=0,B_VP_4=0,B_VP_5=0,B_VP_6=0,B_VP_7=0):
        if round(B_HP_0,0) == 1:
            HP_mode_ext = 0
        if round(B_HP_1,0) == 1:
            HP_mode_ext = 1
        if round(B_HP_2,0) == 1:
            HP_mode_ext = 2
        if round(B_HP_3,0) == 1:
            HP_mode_ext = 3
        if round(B_HP_4,0) == 1:
            HP_mode_ext = 4

        if round(B_HS_IS,0) == 0 and round(B_IS_HGS,0) == 0:
            CHS_mode_ext = 0
        elif round(B_HS_IS,0) == 1 and round(B_IS_HGS,0) == 0:
            CHS_mode_ext = 1
        elif round(B_HS_IS,0) == 0 and round(B_IS_HGS,0) == 1:
            CHS_mode_ext = 2
        elif round(B_HS_IS,0) == 1 and round(B_IS_HGS,0) == 1:
            CHS_mode_ext = 3

        if round(B_GS_HGS,0) == 0 and round(B_GS_CS,0) == 0 and round(B_GS_HGS_CS,0) == 0:
            GS_mode_ext = 0
        if round(B_GS_HGS,0) == 1:
            GS_mode_ext = 1
        if round(B_GS_CS,0) == 1:
            GS_mode_ext = 2
        if round(B_GS_HGS_CS,0) == 1:
            GS_mode_ext = 3

        if round(B_VP_0,0) == 1:
            ST_mode_ext = 0
        if round(B_VP_1,0) == 1:
            ST_mode_ext = 1
        if round(B_VP_2,0) == 1:
            ST_mode_ext = 2
        if round(B_VP_3,0) == 1:
            ST_mode_ext = 3
        if round(B_VP_4,0) == 1:
            ST_mode_ext = 4
        if round(B_VP_5,0) == 1:
            ST_mode_ext = 5
        if round(B_VP_6,0) == 1:
            ST_mode_ext = 6
        if round(B_VP_7,0) == 1:
            ST_mode_ext = 7

        if round(B_HXH_HS,0) == 0:
            HS_mode_ext = 1
        if round(B_HXH_HS,0) == 1:
            HS_mode_ext = 0
        
        if round(B_HGC_HGCHXC,0) == 0:
            CS_mode_ext = 0
        if round(B_HGC_HGCHXC,0)== 1:
            CS_mode_ext = 1
    
        if round(B_HXA,0) == 0:
            AS_mode_ext = 0
        if round(B_HXA,0) == 1:
            AS_mode_ext = 1

        if round(B_HXH_HGC,0) == 0:
            ASC_mode_ext = 1
        if round(B_HXH_HGC,0) == 1:
            ASC_mode_ext = 0

        self.inputValues = [HP_mode_ext,CHS_mode_ext,GS_mode_ext,ST_mode_ext,HS_mode_ext,CS_mode_ext,AS_mode_ext,ASC_mode_ext]
        self.__runModelicaOnce(keys=self.inputKeys,
                             values=self.inputValues)
    
    def getResults(self):
        return pd.DataFrame(self.sim_results,index=[0])


# -----------------Modelica----------------- #

    def __runModelicaOnce(self, keys, values):
        print("Started simulation")

        return_status, output_vals = self.simulator.simulateExtendedModel(
            problem=self.modelName,
            startTime=self.t_start,
            stopTime=self.t_end,

            resultFile=os.path.join(self.simOutputPath, str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) +"_sim_" + str(self.iter_count)),

            outputInterval=self.delta_t,

            initialNames=keys,
            initialValues=values,

            finalNames=self.outputKeysModel)

        print("Done simulating")
        self.sim_results = dict(zip(self.outputKeysModel, output_vals))

        self.t_start = self.t_end
        self.t_end += self.delta_t
        self.iter_count += 1

        self.simulator.importInitial("dsfinal.txt")
        self.simulator.initialized()

        if not return_status:
            raise RuntimeError("Simulation Failed")
            



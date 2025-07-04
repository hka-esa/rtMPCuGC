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

import numpy as np
import pandas as pd
from datetime import datetime
from optimal_control.optimization_results_interface import *

class Measurements_Interface():

    def __init__(self,source="sim",loadPathMeasurements="",time="",timestamp=""):
        self.started = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if time == "extern":
            self.started = timestamp
        self.source=source
        self.loadPathMeasurements = loadPathMeasurements
        self.measurement_interface = Optimization_Results_Interface(source="csv",time="extern",timestamp=self.started)

    def getMeasurementHP_HT(self,update=True):
        if self.source == "standard":
            self.T_HP_HT = 45
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.T_HP_HT = self.measurement_data["T_HP_HT_T"][1]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.T_HP_HT = self.measurement_data["T_hp_h_out"][0]
        return self.T_HP_HT

    def getMeasurementHP_LT(self,update=True):
        if self.source == "standard":
            self.T_HP_LT = 10
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.T_HP_LT = self.measurement_data["T_HP_LT_T"][1]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.T_HP_LT = self.measurement_data["T_hp_c_out"][0]
        return self.T_HP_LT

    def getMeasurementHS(self,update=True):
        if self.source == "standard":
            self.T_HS = 35
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.T_HS = self.measurement_data["T_HS_T"][1]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.T_HS = self.measurement_data["T_hts"][0]
        return self.T_HS

    def getMeasurementHXA(self,update=True):
        if self.source == "standard":
            self.T_HXA = 20
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.T_HXA = self.measurement_data["T_HXA_T"][1]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.T_HXA = self.measurement_data["T_rc"][0]
        return self.T_HXA

    def getMeasurementHGC(self,update=True):
        if self.source == "standard":
            self.T_HGC = 30
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.T_HGC = self.measurement_data["T_HGC_T"][1]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.T_HGC = self.measurement_data["T_header_rc"][0]
        return self.T_HGC

    def getMeasurementHGS(self,update=True):
        if self.source == "standard":
            self.T_HGS = 25
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.T_HGS = self.measurement_data["T_HGS_T"][1]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.T_HGS = self.measurement_data["T_header_gs"][0]
        return self.T_HGS

    def getMeasurementISw(self,update=True):
        if self.source == "standard":
            self.T_ISw = 22
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.T_ISw = self.measurement_data["T_ISw_T"][1]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.T_ISw = self.measurement_data["T_chs_w"][0]
        return self.T_ISw

    def getMeasurementISc(self,update=True):
        if self.source == "standard":
            self.T_ISc = 22
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.T_ISc = self.measurement_data["T_ISc_T"][1]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.T_ISc = self.measurement_data["T_chs_c"][0]
        return self.T_ISc

    def getMeasurementISwc(self,update=True):
        if self.source == "standard":
            self.T_ISwc = 22
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.T_ISwc = self.measurement_data["T_ISc_T"][1]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.T_ISwc = self.measurement_data["T_chs_wc"][0]
        return self.T_ISwc

    def getMeasurementGSw(self,update=True):
        if self.source == "standard":
            self.T_GSw = 16
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.T_GSw = self.measurement_data["T_GSw_T"][1]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.T_GSw = self.measurement_data["T_gs_w"][0]
        return self.T_GSw

    def getMeasurementGSc(self,update=True):
        if self.source == "standard":
            self.T_GSc = 16
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.T_GSc = self.measurement_data["T_GSc_T"][1]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.T_GSc = self.measurement_data["T_gs_c"][0]
        return self.T_GSc

    def getMeasurementGSwc(self,update=True):
        if self.source == "standard":
            self.T_GSwc = 16
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.T_GSwc = self.measurement_data["T_GSc_T"][1]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.T_GSwc = self.measurement_data["T_gs_wc"][0]
        return self.T_GSwc

    def getMeasurementCS(self,update=True):
        if self.source == "standard":
            self.T_CS = 14
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.T_CS = self.measurement_data["T_CS_T"][1]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.T_CS = self.measurement_data["T_lts"][0]
        return self.T_CS

    def getMeasurementRLTS(self,update=True):
        if self.source == "standard":
            self.T_RLTS = 12
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.T_RLTS = self.measurement_data["T_RLTS_T"][1]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.T_RLTS = self.measurement_data["T_lts_dehum"][0]
        return self.T_RLTS

    def getMeasurementHXH(self,update=True):
        if self.source == "standard":
            self.T_HXH = 30
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.T_HXH = self.measurement_data["T_HXH_T"][1]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.T_HXH = self.measurement_data["T_hx_h"][0]
        return self.T_HXH

    def getMeasurementHXC(self,update=True):
        if self.source == "standard":
            self.T_HXC = 30
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.T_HXC = self.measurement_data["T_HXC_T"][1]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.T_HXC = self.measurement_data["T_hx_c"][0]
        return self.T_HXC
    
    def getMeasurementHP(self,update=True):
        if self.source == "standard":
            self.B_HP_0 = 1
            self.B_HP_1 = 0
            self.B_HP_2 = 0
            self.B_HP_3 = 0
            self.B_HP_4 = 0
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.B_HP_0 = self.measurement_data["B_HP_0_T"][0]
            self.B_HP_1 = self.measurement_data["B_HP_1_T"][0]
            self.B_HP_2 = self.measurement_data["B_HP_2_T"][0]
            self.B_HP_3 = self.measurement_data["B_HP_3_T"][0]
            self.B_HP_4 = self.measurement_data["B_HP_4_T"][0]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.B_HP_0 = 0
            self.B_HP_1 = 0
            self.B_HP_2 = 0
            self.B_HP_3 = 0
            self.B_HP_4 = 0
            if self.measurement_data["HP_mode_ext"][0] == 0:
                self.B_HP_0 = 1
            if self.measurement_data["HP_mode_ext"][0] == 1:
                self.B_HP_1 = 1
            if self.measurement_data["HP_mode_ext"][0] == 2:
                self.B_HP_2 = 1
            if self.measurement_data["HP_mode_ext"][0] == 3:
                self.B_HP_3 = 1
            if self.measurement_data["HP_mode_ext"][0] == 4:
                self.B_HP_4 = 1
        return [self.B_HP_0,self.B_HP_1,self.B_HP_2,self.B_HP_3,self.B_HP_4]

    def getMeasurementHXH_HS(self,update=True):
        if self.source == "standard":
            self.B_HXH_HS = 0
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.B_HXH_HS = self.measurement_data["B_HXH_HS_T"][0]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.B_HXH_HS = (1 - self.measurement_data["HS_mode_ext"][0])
        return self.B_HXH_HS

    def getMeasurementHGC_HGCHXC(self,update=True):
        if self.source == "standard":
            self.B_HGC_HGCHXC = 1
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.B_HGC_HGCHXC = self.measurement_data["B_HGC_HGCHXC_T"][0]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.B_HGC_HGCHXC = self.measurement_data["CS_mode_ext"][0]
        return self.B_HGC_HGCHXC

    def getMeasurementHXAb(self,update=True):
        if self.source == "standard":
            self.B_HXA = 1
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.B_HXA = self.measurement_data["B_HXA_T"][0]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.B_HXA = self.measurement_data["AS_mode_ext"][0]
        return self.B_HXA

    def getMeasurementHXH_HGC(self,update=True):
        if self.source == "standard":
            self.B_HXH_HGC = 0
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.B_HXH_HGC = self.measurement_data["B_HXH_HGC_T"][0]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.B_HXH_HGC = (1 - self.measurement_data["ASC_mode_ext"][0])
        return self.B_HXH_HGC

    def getMeasurementHS_IS(self,update=True):
        if self.source == "standard":
            self.B_HS_IS = 0
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.B_HS_IS = self.measurement_data["B_HS_IS_T"][0]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            if self.measurement_data["CHS_mode_ext"][0] == 0 or self.measurement_data["CHS_mode_ext"][0] == 2:
                self.B_HS_IS = 0
            if self.measurement_data["CHS_mode_ext"][0] == 1 or self.measurement_data["CHS_mode_ext"][0] == 3:
                self.B_HS_IS = 1
        return self.B_HS_IS

    def getMeasurementIS_HGS(self,update=True):
        if self.source == "standard":
            self.B_IS_HGS = 0
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.B_IS_HGS = self.measurement_data["B_IS_HGS_T"][0]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            if self.measurement_data["CHS_mode_ext"][0] == 0 or self.measurement_data["CHS_mode_ext"][0] == 1:
                self.B_IS_HGS = 0
            if self.measurement_data["CHS_mode_ext"][0] == 2 or self.measurement_data["CHS_mode_ext"][0] == 3:
                self.B_IS_HGS = 1
        return self.B_IS_HGS

    def getMeasurementGS_HGS(self,update=True):
        if self.source == "standard":
            self.B_GS_HGS = 0
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.B_GS_HGS = self.measurement_data["B_GS_HGS_T"][0]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            if self.measurement_data["GS_mode_ext"][0] == 0 or self.measurement_data["GS_mode_ext"][0] == 2:
                self.B_GS_HGS = 0
            if self.measurement_data["GS_mode_ext"][0] == 1 or self.measurement_data["GS_mode_ext"][0] == 3:
                self.B_GS_HGS = 1
        return self.B_GS_HGS

    def getMeasurementGS_CS(self,update=True):
        if self.source == "standard":
            self.B_GS_CS = 0
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.B_GS_CS = self.measurement_data["B_GS_CS_T"][0]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            if self.measurement_data["GS_mode_ext"][0] == 0 or self.measurement_data["GS_mode_ext"][0] == 1:
                self.B_GS_CS = 0
            if self.measurement_data["GS_mode_ext"][0] == 2 or self.measurement_data["GS_mode_ext"][0] == 3:
                self.B_GS_CS = 1
        return self.B_GS_CS
    
    def getMeasurementGS_HGS_CS(self,update=True):
        if self.source == "standard":
            self.B_GS_HGS_CS = 0
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.B_GS_HGS_CS = self.measurement_data["B_GS_HGS_CS_T"][0]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            if self.measurement_data["GS_mode_ext"][0] == 0 or self.measurement_data["GS_mode_ext"][0] == 1 or self.measurement_data["GS_mode_ext"][0] == 2:
                self.B_GS_HGS_CS = 0
            if self.measurement_data["GS_mode_ext"][0] == 3:
                self.B_GS_HGS_CS = 1
        return self.B_GS_HGS_CS

    def getMeasurementVP(self,update=True):
        if self.source == "standard":
            self.B_VP_0 = 1
            self.B_VP_1 = 0
            self.B_VP_2 = 0
            self.B_VP_3 = 0
            self.B_VP_4 = 0
            self.B_VP_5 = 0
            self.B_VP_6 = 0
            self.B_VP_7 = 0
        elif self.source == "sim":
            if update== True:
                self.getLatestSimUpdate()
            self.B_VP_0 = self.measurement_data["B_VP_0_T_1"][0]
            self.B_VP_1 = self.measurement_data["B_VP_1_T_1"][0]
            self.B_VP_2 = self.measurement_data["B_VP_2_T_1"][0]
            self.B_VP_3 = self.measurement_data["B_VP_3_T_1"][0]
            self.B_VP_4 = self.measurement_data["B_VP_4_T_1"][0]
            self.B_VP_5 = self.measurement_data["B_VP_5_T_1"][0]
            self.B_VP_6 = self.measurement_data["B_VP_6_T_1"][0]
            self.B_VP_7 = self.measurement_data["B_VP_7_T_1"][0]
        elif self.source == "sim-dymola":
            if update== True:
                self.getLatestSimDymUpdate()
            self.B_VP_0 = 0
            self.B_VP_1 = 0
            self.B_VP_2 = 0
            self.B_VP_3 = 0
            self.B_VP_4 = 0
            self.B_VP_5 = 0
            self.B_VP_6 = 0
            self.B_VP_7 = 0
            if self.measurement_data["ST_mode_ext"][0] == 0:
                self.B_VP_0 = 1
            if self.measurement_data["ST_mode_ext"][0] == 1:
                self.B_VP_1 = 1
            if self.measurement_data["ST_mode_ext"][0] == 2:
                self.B_VP_2 = 1
            if self.measurement_data["ST_mode_ext"][0] == 3:
                self.B_VP_3 = 1
            if self.measurement_data["ST_mode_ext"][0] == 4:
                self.B_VP_4 = 1
            if self.measurement_data["ST_mode_ext"][0] == 5:
                self.B_VP_5 = 1
            if self.measurement_data["ST_mode_ext"][0] == 6:
                self.B_VP_6 = 1
            if self.measurement_data["ST_mode_ext"][0] == 7:
                self.B_VP_7 = 1
        return [self.B_VP_0,self.B_VP_1,self.B_VP_2,self.B_VP_3,self.B_VP_4,self.B_VP_5,self.B_VP_6,self.B_VP_7]

    def getMeasurementsAll(self,update=True):
        if update == True:
            if self.source =="sim":
                self.getLatestSimUpdate()
            elif self.source == "sim-dymola":
                self.getLatestSimDymUpdate()
        measurementHP_HT = self.getMeasurementHP_HT(update=False)
        measurementHP_LT = self.getMeasurementHP_LT(update=False)
        measurementHS = self.getMeasurementHS(update=False)
        measurementHXA = self.getMeasurementHXA(update=False)
        measurementHGC = self.getMeasurementHGC(update=False)
        measurementHGS = self.getMeasurementHGS(update=False)
        measurementISw = self.getMeasurementISw(update=False)
        measurementISc = self.getMeasurementISc(update=False)
        measurementISwc = self.getMeasurementISwc(update=False)
        measurementGSw = self.getMeasurementGSw(update=False)
        measurementGSc = self.getMeasurementGSc(update=False)
        measurementGSwc = self.getMeasurementGSwc(update=False)
        measurementCS = self.getMeasurementCS(update=False)
        measurementRLTS = self.getMeasurementRLTS(update=False)
        measurementHXH = self.getMeasurementHXH(update=False)
        measurementHXC = self.getMeasurementHXC(update=False)
        measurementHP = self.getMeasurementHP(update=False)
        measurementHXH_HS = self.getMeasurementHXH_HS(update=False)
        measurementHGC_HGCHXC = self.getMeasurementHGC_HGCHXC(update=False)
        measurementHXAb = self.getMeasurementHXAb(update=False)
        measurementHXH_HGC = self.getMeasurementHXH_HGC(update=False)
        measurementHS_IS = self.getMeasurementHS_IS(update=False)
        measurementIS_HGS = self.getMeasurementIS_HGS(update=False)
        measurementGS_HGS = self.getMeasurementGS_HGS(update=False)
        measurementGS_CS = self.getMeasurementGS_CS(update=False)
        measurementGS_HGS_CS = self.getMeasurementGS_HGS_CS(update=False)
        measurementVP = self.getMeasurementVP(update=False)
        self.dictMeasurements = {"measurementHP_HT":measurementHP_HT,"measurementHP_LT":measurementHP_LT,"measurementHS":measurementHS,"measurementHXA":measurementHXA,"measurementHGC":measurementHGC,
        "measurementHGS":measurementHGS,"measurementISw":measurementISw,"measurementISc":measurementISc,"measurementISwc":measurementISwc,"measurementGSw":measurementGSw,"measurementGSc":measurementGSc,"measurementGSwc":measurementGSwc,"measurementCS":measurementCS,"measurementRLTS":measurementRLTS,"measurementHXH":measurementHXH,"measurementHXC":measurementHXC,
        "measurementHP":measurementHP,"measurementHXH_HS":measurementHXH_HS,"measurementHGC_HGCHXC":measurementHGC_HGCHXC,"measurementHXAb":measurementHXAb,"measurementHXH_HGC":measurementHXH_HGC,"measurementHS_IS":measurementHS_IS,
        "measurementIS_HGS":measurementIS_HGS,"measurementGS_HGS":measurementGS_HGS,"measurementGS_CS":measurementGS_CS,"measurementGS_HGS_CS":measurementGS_HGS_CS,"measurementVP":measurementVP}
        return self.dictMeasurements
    
    def getLatestSimUpdate(self):
        measurement_data = self.measurement_interface.getOptimizationResults(savePath=self.loadPathMeasurements)
        self.measurement_interface.setOptimizationResults(onlySetCounter=True)
        self.measurement_data = measurement_data.iloc[0:2,:]

    def getLatestSimDymUpdate(self):
        measurement_data = self.measurement_interface.getOptimizationResults(savePath=self.loadPathMeasurements)
        self.measurement_interface.setOptimizationResults(onlySetCounter=True)
        self.measurement_data = measurement_data

if __name__ == "__main__":
    test = Measurements_Interface()
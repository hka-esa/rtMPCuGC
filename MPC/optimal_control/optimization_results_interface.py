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
import numpy as np
import pandas as pd
from datetime import datetime

class Optimization_Results_Interface():

    def __init__(self,source="csv",time="bySet",timestamp=""):
        self.started = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if time == "extern":
            self.started = timestamp
        self.source=source
        self.time = time
        self.count = 0

    def setOptimizationResults(self,dataFrame=pd.DataFrame(),savePath="",onlySetCounter=False):
        if onlySetCounter == False:
            if self.source == "csv":
                if self.time == "bySet":
                    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                    dataFrame.to_csv((savePath+'Results_'+str(now)+'.csv'), sep = ";")
                elif self.time == "byCreate" or "extern":
                    dataFrame.to_csv((savePath+'Results_'+str(self.started)+'_iter_'+str(self.count)+'.csv'), sep = ";")
            self.count = self.count + 1
        if onlySetCounter == True:
            self.count = self.count + 1

    def getOptimizationResults(self,savePath=""):
        if self.source == "csv":
            if self.time == "byCreate" or "extern":
                try:
                    for i in range(-1,2):
                        try:
                            data = "Results_" + str(self.started) + "_iter_" + str(self.count+i) + ".csv"
                            resultsDf = pd.read_csv(os.path.join(savePath,data), sep = ";", index_col=0, parse_dates=False)
                        except:
                            pass
                    return resultsDf
                except:
                    for item in sorted(os.listdir(savePath)):
                        full_path = os.path.join(savePath, item)
                        if os.path.isfile(full_path):
                            resultsDf = pd.read_csv(full_path, sep = ";", index_col=0, parse_dates=False)
                    return resultsDf
            if self.time == "bySet":
                for i in os.listdir(savePath):
                    if os.path.isfile(os.path.join(savePath,i)) and "Results_" in i:
                        data = i
                resultsDf = pd.read_csv(os.path.join(savePath,data), sep = ";", index_col=0, parse_dates=False)
                return resultsDf
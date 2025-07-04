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
from datetime import timedelta

class Market_Interface():

    def __init__(self,type="trafficLight", directionSignal="pos", timestampSignalStart="", timestampSignalStop="", factorSignal=1.5, simTimeStart="", simTimeStop="", intervalInSec=600):
        self.started = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")   
        self.type = type
        self.directionSignal = directionSignal
        self.timestampSignalStart = timestampSignalStart
        self.timestampSignalStop = timestampSignalStop # Careful !! always minus the interval time e.g. 11:50 for ending on 12:00
        self.factorSignal = factorSignal
        self.simTimeStart = simTimeStart
        self.simTimeStop = simTimeStop
        self.intervalInSec = intervalInSec
        
        date_range = pd.date_range(start=self.simTimeStart,end=self.simTimeStop,freq=str(self.intervalInSec)+"S")
        self.market_signal = pd.DataFrame(index=date_range)
        self.market_signal["market_signal"] = 0
        if self.directionSignal == "pos":
            self.market_signal["market_signal"][timestampSignalStart:timestampSignalStop] = self.factorSignal 
        elif self.directionSignal == "neg":
            self.market_signal["market_signal"][timestampSignalStart:timestampSignalStop] = -(self.factorSignal)
        else:
            print("No selection of market direction!")


    def getProfileForecastMarket(self, timestampStart=datetime.strptime("2000-01-01 00:00:00","%Y-%m-%d %H:%M:%S"), intervals=[]):
        if self.type == "trafficLight":
            self.signal = []
            for i in intervals:
                self.signal.append(np.mean(self.market_signal.loc[(timestampStart).strftime("%Y-%m-%d %H:%M:%S"):(timestampStart + timedelta(seconds=(i-1))).strftime("%Y-%m-%d %H:%M:%S"),"market_signal"]))
                timestampStart = timestampStart + timedelta(seconds=i)
            return(self.signal)
        elif self.type == "demandResponse":
            self.counter = 0
            self.signal = []
            for i in intervals:
                if self.counter == 0:
                    self.signal.append(np.mean(self.market_signal.loc[(timestampStart).strftime("%Y-%m-%d %H:%M:%S"):(timestampStart + timedelta(seconds=(i-1))).strftime("%Y-%m-%d %H:%M:%S"),"market_signal"]))
                    self.counter = 1
                else:
                    self.signal.append(0)
            return(self.signal)
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

class Forecast_Interface():
        
    def __init__(self,source="sim",priceType="flat",loadPathDemand="",loadPathWeather="",loadPathPrice=""):
        self.started = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")   
        self.hour_in_sec = 3600
        self.source = source
        self.priceType = priceType
        #self.forecast_demand_csv = pd.read_csv(loadPathDemand,index_col=0) !! activate, if modelica model connected
        #self.forecast_weather_csv = pd.read_csv(loadPathWeather,index_col=0) !! activate, if modelica model connected
        #self.forecast_price_csv = pd.read_csv(loadPathPrice,index_col=0) !! activate, if modelica model connected

    def getProfileForecastHeat(self, timestampStart=datetime.strptime("2000-01-01 00:00:00","%Y-%m-%d %H:%M:%S"), intervals=[]):
        if self.source == "random":
            heat_dem_sim = 300
            self.profileForecastHeat = [0] * len(intervals)
            for i in range(0,len(intervals)):
                self.profileForecastHeat[i] = heat_dem_sim * np.random.random()
        elif self.source == "sim":
            self.profileForecastHeat = []
            for i in intervals:
                self.profileForecastHeat.append(np.mean(self.forecast_demand_csv.loc[(timestampStart).strftime("%Y-%m-%d %H:%M:%S"):(timestampStart + timedelta(seconds=(i-1))).strftime("%Y-%m-%d %H:%M:%S"),"Q_HP_Last_Waerme_NEW"]))
                timestampStart = timestampStart + timedelta(seconds=i)
        return self.profileForecastHeat

    def getProfileForecastCool(self, timestampStart=datetime.strptime("2000-01-01 00:00:00","%Y-%m-%d %H:%M:%S"), intervals=[]):
        if self.source == "random":
            cool_dem_sim = -100
            self.profileForecastCool = [0] * len(intervals)
            for i in range(0,len(intervals)):
                self.profileForecastCool[i] = cool_dem_sim * np.random.random()
        elif self.source == "sim":
            self.profileForecastCool = []
            for i in intervals:
                self.profileForecastCool.append(np.mean(self.forecast_demand_csv.loc[(timestampStart).strftime("%Y-%m-%d %H:%M:%S"):(timestampStart + timedelta(seconds=(i-1))).strftime("%Y-%m-%d %H:%M:%S"),"Q_HP_Last_Kältespeicher_NEW"]))
                timestampStart = timestampStart + timedelta(seconds=i)
        return self.profileForecastCool

    def getProfileForecastDry(self, timestampStart=datetime.strptime("2000-01-01 00:00:00","%Y-%m-%d %H:%M:%S"), intervals=[]):
        if self.source == "random":
            dry_dem_sim = -30
            self.profileForecastDry = [0] * len(intervals)
            for i in range(0,len(intervals)):
                self.profileForecastDry[i] = dry_dem_sim * np.random.random()
        elif self.source == "sim":
            self.profileForecastDry = []
            for i in intervals:
                self.profileForecastDry.append(np.mean(self.forecast_demand_csv.loc[(timestampStart).strftime("%Y-%m-%d %H:%M:%S"):(timestampStart + timedelta(seconds=(i-1))).strftime("%Y-%m-%d %H:%M:%S"),"Q_HP_Last_Pufferspeicher_NEW"]))
                timestampStart = timestampStart + timedelta(seconds=i)
        return self.profileForecastDry

    def getProfileForecastWeather(self, timestampStart=datetime.strptime("2000-01-01 00:00:00","%Y-%m-%d %H:%M:%S"), intervals=[]):
        if self.source == "random":
            weather_sim = 20
            random_factor = 2
            self.profileForecastWeather = [0] * len(intervals)
            for i in range(0,len(intervals)):
                self.profileForecastWeather[i] = weather_sim + random_factor * np.random.random()
        elif self.source == "sim":
            self.profileForecastWeather = []
            for i in intervals:
                self.profileForecastWeather.append(np.mean(self.forecast_weather_csv.loc[(timestampStart).strftime("%Y-%m-%d %H:%M:%S"):(timestampStart + timedelta(seconds=(i-1))).strftime("%Y-%m-%d %H:%M:%S"),"TT_10"]))
                timestampStart = timestampStart + timedelta(seconds=i)
        return self.profileForecastWeather

    def getProfileForecastPrice(self, timestampStart=datetime.strptime("2000-01-01 00:00:00","%Y-%m-%d %H:%M:%S"), intervals=[]):
        if self.source == "random" or self.source == "sim":
            if self.priceType == "flat":
                price_cost_sim = 0.16
                random_factor = 0.0
                self.profileForecastPrice = [0] * len(intervals)
                for i in range(0,len(intervals)):
                    self.profileForecastPrice[i] = price_cost_sim + random_factor * np.random.random()
            elif self.priceType == "variable":
                self.profileForecastPrice = []
                for i in intervals:
                    self.profileForecastPrice.append(np.mean(self.forecast_price_csv.loc[(timestampStart).strftime("%Y-%m-%d %H:%M:%S"):(timestampStart + timedelta(seconds=(i-1))).strftime("%Y-%m-%d %H:%M:%S"),"price"]))
                    timestampStart = timestampStart + timedelta(seconds=i)
        return self.profileForecastPrice

    def getProfileForecastFrost(self, timestampStart=datetime.strptime("2000-01-01 00:00:00","%Y-%m-%d %H:%M:%S"), intervals=[]):
        if self.source == "random":
            self.profileForecastFrost = [0] * len(intervals)
        elif self.source == "sim":
            self.profileForecastFrost = [0] * len(intervals)
            profileForecastWeather = []
            j = 0
            for i in intervals:
                profileForecastWeather.append(np.mean(self.forecast_weather_csv.loc[(timestampStart).strftime("%Y-%m-%d %H:%M:%S"):(timestampStart + timedelta(seconds=(i-1))).strftime("%Y-%m-%d %H:%M:%S"),"TT_10"]))
                timestampStart = timestampStart + timedelta(seconds=i)
            for i in profileForecastWeather:
                if i <= 0:
                    self.profileForecastFrost[j] == 1
                    j = j+1
        return self.profileForecastFrost

    def getForecastFrost(self, timestampStart=datetime.strptime("2000-01-01 00:00:00","%Y-%m-%d %H:%M:%S"), periodInHours=168):
        if self.source == "random":
            self.forecastFrost = 0
        elif self.source == "sim":
            profileForecastWeather = np.mean(self.forecast_weather_csv.loc[(timestampStart).strftime("%Y-%m-%d %H:%M:%S"):(timestampStart + timedelta(seconds=((3600*periodInHours)-1))).strftime("%Y-%m-%d %H:%M:%S"),"TT_10"])
            if profileForecastWeather <= 0:
                self.forecastFrost = 1
            else:
                self.forecastFrost = 0
        return self.forecastFrost

    def getProfilesAll(self, timestampStart=datetime.strptime("2000-01-01 00:00:00","%Y-%m-%d %H:%M:%S"), intervals=[], periodFrostInHours=168):
        profileForecastHeat = self.getProfileForecastHeat(timestampStart=timestampStart,intervals=intervals)
        profileForecastCool = self.getProfileForecastCool(timestampStart=timestampStart,intervals=intervals)
        profileForecastDry = self.getProfileForecastDry(timestampStart=timestampStart,intervals=intervals)
        profileForecastWeather = self.getProfileForecastWeather(timestampStart=timestampStart,intervals=intervals)
        profileForecastPrice = self.getProfileForecastPrice(timestampStart=timestampStart,intervals=intervals)
        profileForecastFrost = self.getProfileForecastFrost(timestampStart=timestampStart,intervals=intervals)
        forecastFrost = self.getForecastFrost(timestampStart=timestampStart,periodInHours=periodFrostInHours)
        self.dictProfiles = {"profileForecastHeat":profileForecastHeat,"profileForecastCool":profileForecastCool,
        "profileForecastDry":profileForecastDry,"profileForecastWeather":profileForecastWeather,
        "profileForecastPrice":profileForecastPrice,"profileForecastFrost":profileForecastFrost,"forecastFrost":forecastFrost}
        return self.dictProfiles

if __name__ == "__main__":
    test = Forecast_Interface()
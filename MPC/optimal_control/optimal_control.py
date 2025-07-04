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

class Optimal_Control():

    def __init__(self):
        self.started = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.m = pyo.ConcreteModel()
        self.position_symbol = {}
        self.position_object = {}
    
    def getModel(self):
        return self.m
    
    def addModelParts(self,model):
        self.m = model

    def addModelObject(self,object,position=0, symbol=""):
        self.position_symbol[position] = symbol
        self.position_object[position] = object
        print("Added object " +str(symbol) + " at position " +str(position) +" to the optmimal control problem.")

    def setObjective(self):
        collectedModels = ""

        for i in self.position_symbol.values():
            collectedModels = collectedModels + " self.m.C_TOT_" +str(i) +"_ + self.m.S_TOT_" +str(i) +"_ + self.m.T_TOT_" +str(i) +"_ +"
    
        l = len(collectedModels)
        collectedModels = collectedModels[:l-1]
        collectedModels = collectedModels + " )"

        exec("self.m.OBJ = pyo.Objective(expr=" + str(collectedModels))
        
    def setSolverAndRunOptimization(self,solver = 0, warmstart = False, timeLimit = 180, showSolverOutput = 0, writeILP = 0, writeMPSfile = 0):
        print("### Main optimization started ###")
        if writeMPSfile == 1:
            self.m.write(filename = "WB.mps", io_options = {"symbolic_solver_labels":True})

        if solver == 0:
            self.opt = pyo.SolverFactory('gurobi', solver_io="python")
            self.opt.options['TimeLimit'] = timeLimit
            self.opt.options['threads'] = 8
            #self.opt.options['MIPFocus'] = 1
            #self.opt.options['ObjBound'] = 50
            #self.opt.options['Cutoff'] = 500
            if writeILP == 1:
                self.opt.options['resultFile'] = 'test.ilp'
        elif solver == 1:
            self.opt = pyo.SolverFactory('cbc')
            self.opt.options['Sec'] = timeLimit
        elif solver == 2:
            self.opt = pyo.SolverFactory('glpk')
            self.opt.options['tmlim'] = timeLimit
            #self.opt.options['mipgap'] = 1e-6 # not needed atm
        
        if writeILP == 1:
            self.results = self.opt.solve(self.m,warmstart=warmstart,tee=True,symbolic_solver_labels=True) 
        else:
            self.results = self.opt.solve(self.m,warmstart=warmstart,tee=True)

        if showSolverOutput == 1:
            print(self.results)

    def getResults(self,source,savePath,combinedFile,singleFile,timestampStart,intervals):
        resultsFile = {} 
        results = pd.DataFrame()
        j = 0
        length_last_files = 0

        try:
            for i in self.position_object.values():
                resultsFile[j] = i.getResults(model=self.m,source=source,savePath=savePath,singleFile=singleFile)
                if length_last_files == 0:
                    pass
                else:
                    resultsFile[j].index += length_last_files
                length_last_files = (length_last_files-1) + resultsFile[j].index.size
                results = pd.concat([results,resultsFile[j]],axis=1)  #results.append(resultsFile[j])
                j = j+1
        except:
            raise RuntimeError("Optimization didn't come to a solution.")

        if combinedFile == True:
            try:
                timestampArray = [timestampStart.strftime("%Y-%m-%d %H:%M:%S")]
                for i in intervals[1:]:
                    timestampArray.append((timestampStart+timedelta(seconds=i)).strftime("%Y-%m-%d %H:%M:%S"))
                    timestampStart = timestampStart+timedelta(seconds=i)
                results = results.set_index(pd.Index(timestampArray))
            except:
                raise RuntimeError("Optimization didn't come to a solution.")
            source.setOptimizationResults(dataFrame=results,savePath=savePath)
            return results

if __name__ == "__main__":
    optimal_control = Optimal_Control()

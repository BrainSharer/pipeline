#DEPRECATED - DELETE
#QUESTIONS: SEE DUANE
#11-AUG-2025

from numpy import *
from pylab import plot

class Predictor:
    #TODO: FUNCTIONALITY DUPLICATED IN cell_manager.py (get_prediction_and_label) ~line 675
    def __init__(self,std=1.5):
        self.std = std
    
    def decision(self,mean,std):
        if mean <= -self.std:
            return -2
        elif mean>-self.std and mean <= self.std:
            return 0
        elif mean >self.std:
            return 2

class BetterPredictor:
    def __init__(self,std=1.5):
        self.std = std
    
    def decision(self,mean,std):
        if mean<0:
            return -2
        elif mean-std<0 and mean >0:
            return 0
        elif mean-std>0:
            return 2

class GreedyPredictor:
    def __init__(self,boundary_points=[[0,3],[3,4.5],[1,6],[-3,4],[-10,7],[10,7]]):
        self.set_boundary_points(boundary_points)

    def set_boundary_points(self,boundary_points):
        self.boundary_points=boundary_points
        self.boundary_lines=[]
        self.boundary_lines.append(self.points2line(self.boundary_points[0],self.boundary_points[1],0))
        self.boundary_lines.append(self.points2line(self.boundary_points[1],self.boundary_points[2],1))
        self.boundary_lines.append(self.points2line(self.boundary_points[2],self.boundary_points[3],2))
        self.boundary_lines.append(self.points2line(self.boundary_points[3],self.boundary_points[0],3))
        self.boundary_lines.append(self.points2line(self.boundary_points[1],self.boundary_points[5],4))
        self.boundary_lines.append(self.points2line(self.boundary_points[3],self.boundary_points[4],5))

    def print_boundary_points(self):
        print(self.boundary_points)

    def points2line(self,p1,p2,i):
        x1,y1=p1
        x2,y2=p2
        a=(y1-y2)/(x1-x2)
        b=y1-a*x1
        return a,b

    def plotline(self,a,b,i):
        X=arange(-5,5,0.01)
        Y=a*X+b
        plot(X,Y,label=str(i))
        
    def aboveline(self,p,l):
        return l[0]*p[0]+l[1] < p[1]

    def decision(self,x,y):
        p=[x,y]
        if self.aboveline(p,self.boundary_lines[0]) and not self.aboveline(p,self.boundary_lines[1])\
        and not self.aboveline(p,self.boundary_lines[2]) and self.aboveline(p,self.boundary_lines[3]):
            return 0
        if (x<0 and not self.aboveline(p,self.boundary_lines[5])) or (x>0 and self.aboveline(p,self.boundary_lines[4])):
            return -2
        if (x>0 and not self.aboveline(p,self.boundary_lines[4])) or (x<0 and self.aboveline(p,self.boundary_lines[5])):
            return 2

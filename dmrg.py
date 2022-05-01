# -*- coding: utf-8 -*-
#author:Miyan
#Start data:2021.6.6
#End data:2021.6.7

import numpy as np
import time
from scipy.sparse.linalg import eigsh
import scipy
import time
import matplotlib.pyplot as plt
from scipy.linalg import norm

def dagger(Array):
    return np.conj(np.transpose(Array))

def dim(Array):
    return max(Array.shape)

def getRandomVector(dimension):
    res = np.random.random(dimension)
    res /= np.sqrt(norm(res))
    return res

class Solution:
    def __init__(self):
        self.sx = np.array([[0,1/2+0j],[1/2,0]])
        self.sy = np.array([[0,-0.5j],[0.5j,0]])
        self.sz = np.array([[1/2+0j,0],[0,-1/2+0j]])

        self.bo = np.array([
                    [0.25,0,0,0],
                    [0,-0.25,0.5,0],
                    [0,0.5,-0.25,0],
                    [0,0,0,0.25]
                    ])
        #截断指标
        self.truncationIndex = 20
        self.eps = 1e-5
        # self.loop = 2**31 - 1
        #格点数目
        self.loop = 300
    
    def dmrg(self):
        Sys = np.copy(self.bo)
        Env = np.copy(self.bo)

        '''
        超块矩阵： S1 S2 I3 I4 + I1 S2 S3 I4 + I1 I2 S3 S4 
        '''
        Super = np.kron(self.bo,np.eye(4)) + np.kron(np.kron(np.eye(2),self.bo),np.eye(2)) + np.kron(np.eye(4),self.bo)
        lastEnergy = 0
        e = []
        for L in range(4, self.loop, 2):
            startVector = getRandomVector(dim(Super))
            [va,ve] = eigsh(Super,k = 1,v0 = startVector)

            energy = va[0] #平均化能量
            print(energy/L)
            # energy = (va[0] - lastEnergy) / 2 #能量差作为能量
            # lastEnergy = va[0]
            print(L)
            length = int(np.sqrt(dim(ve)))
            psi = np.reshape(ve,(length,-1))

            SystemRho = psi @ dagger(psi)
            EnvironmentRho = dagger(psi) @ psi
            [SystemValue,SystemVector] = np.linalg.eigh(-SystemRho) #eigh 从小到大排，为了取大需要如此
            [EnvironmentValue,EnvironmentVector] = np.linalg.eigh(-EnvironmentRho)
            SystemVector = SystemVector[:, 0 : self.truncationIndex]
            EnvironmentVector = EnvironmentVector[:, 0 : self.truncationIndex]

            #系统和环境各自插入一个格点，构造下一个系统和环境的哈密顿量
            length = length >> 1 #位运算不会导致数值类型由int 变换为 float
            eye = np.eye(length)

            sxbar = dagger(SystemVector) @ np.kron(eye,self.sx) @ SystemVector
            sybar = dagger(SystemVector) @ np.kron(eye,self.sy) @ SystemVector
            szbar = dagger(SystemVector) @ np.kron(eye,self.sz) @ SystemVector
            Sys = np.kron(dagger(SystemVector) @ Sys @ SystemVector,np.eye(2)) + np.kron(sxbar, self.sx) + np.kron(sybar, self.sy) + np.kron(szbar, self.sz)

            sxbar = dagger(EnvironmentVector) @ np.kron(self.sx,eye) @ EnvironmentVector
            sybar = dagger(EnvironmentVector) @ np.kron(self.sy,eye) @ EnvironmentVector
            szbar = dagger(EnvironmentVector) @ np.kron(self.sz,eye) @ EnvironmentVector
            Env = np.kron(np.eye(2),dagger(EnvironmentVector) @ Env @ EnvironmentVector) + np.kron(self.sx,sxbar) + np.kron(self.sy,sybar) + np.kron(self.sz,szbar)

            spaceDimension = dim(Env) >> 1
            Super=np.kron(Sys,np.eye(spaceDimension << 1)) + np.kron(np.eye(spaceDimension),np.kron(self.bo,np.eye(spaceDimension))) + np.kron(np.eye(spaceDimension << 1),Env)
            
            lastEnergy = energy
            e.append(lastEnergy/L)

            # e.append(lastEnergy/L)
            
        return [lastEnergy / L,L,e]


s = Solution()
# time_start = time.time()
[energy,L,e20] = s.dmrg()
# s.truncationIndex = 15
# [energy,L,e15] = s.dmrg()
# s.truncationIndex = 25
# [energy,L,e25] = s.dmrg()
# time_end = time.time()

Lens = np.array([2*i+4 for i in range(0,dim(np.array(e20)))])
plt.figure(figsize=(10,7))
# plt.plot(Lens,e15,linewidth=3.0)
plt.plot(Lens,e20,linewidth=3.0)
# plt.plot(Lens,e25,linewidth=3.0)
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.title(r"$E_g - L$ in ",fontsize = 18)
plt.ylabel(r"$E_g$",fontsize=21)
plt.xlabel(r"$L$",fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend([r"$\chi = 15$",r"$\chi = 20$",r"$\chi = 25$"],fontsize=20)

print("Go for length = %d  and  the energy is %2.12f"%(L,energy))


Lens = np.array([i+1 for i in range(0,dim(np.array(e20)))])
plt.figure(figsize=(10,7))

k = np.polyfit(1/Lens,e20,1)
plt.plot(1/Lens,k[0]/Lens+k[1],linewidth=2.5,linestyle="--")
k = np.polyfit(1/Lens,e20,2)
plt.plot(1/Lens,k[0]/Lens/Lens+k[1]/Lens+k[2],linewidth=2.5,linestyle='-.')
k = np.polyfit(1/Lens,e20,3)
plt.plot(1/Lens,k[0]/Lens/Lens/Lens+k[1]/Lens/Lens+k[2]/Lens+k[3],linewidth=2.5,linestyle='-')
plt.scatter(1/Lens,e20,s=200,marker="1",color="red",linewidths=3.5)
plt.legend([r"$y = kx + b$",r"$y = ax^2 + bx + c$",r"$y = ax^3 + bx^2 + cx +d$","原始数据"],fontsize = 20)
plt.xlabel(r"$\frac{1}{n}$",fontsize=20)
plt.ylabel(r"$E_g$",fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.text(0.01,-0.40,r"迭代次数$n = (L - 2)/2$",fontsize=20,color="blue")
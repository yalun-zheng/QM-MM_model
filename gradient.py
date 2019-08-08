import numpy as np
import os

# (r0 in angstom,epsilon in kcal/mol)
f = 1
LJParas = {"C":(2.04,0.027*f),"H":(1.620,0.020*f),"O":(1.820,0.059*f),"N":(1.930,0.043*f),
           "S":(2.150,0.202*f),"P":(2.220,0.168*f),"c":(2.04,0.027*f),"h":(1.620,0.020*f),
           "o":(1.820,0.059*f),"n":(1.930,0.043*f),"s":(2.150,0.202*f),"p":(2.220,0.168*f)}

def LJPotential(r0,epsilon,r):
    return epsilon*(-2.25*(r0/r)**6+1.84*1e5*np.exp(-12*(r/r0)))
def LJGradient(r0,epsilon,r):
    return epsilon*(6*2.25*r0**6/r**7-1.84*1e5*np.exp(-12*r/r0)*12/r0)
def LJGradbt2atoms(atomx,atom0):
    r0 = LJParas[atom0.type][0]+LJParas[atomx.type][0]
#    r0 = 2*np.sqrt(LJParas[atom0.type][0]*LJParas[atomx.type][0])
    epsilon = np.sqrt(LJParas[atom0.type][1]*LJParas[atomx.type][1])
    r_v = atomx.coordinate-atom0.coordinate
    r = np.linalg.norm(r_v)
    return LJGradient(r0,epsilon,r)*r_v/r
def LJPotbt2atoms(atom0,atomx):
    r0 = LJParas[atom0.type][0]+LJParas[atomx.type][0]
#    r0 = 2*np.sqrt(LJParas[atom0.type][0]*LJParas[atomx.type][0])
    epsilon = np.sqrt(LJParas[atom0.type][1]*LJParas[atomx.type][1])
    r_v = atom0.coordinate-atomx.coordinate
    r = np.linalg.norm(r_v)
    return LJPotential(r0,epsilon,r)
def Grad_central(cluster):
    m0 = cluster[0]
    grads = []
    for atom0x in m0.atomDic:
        grad = np.zeros(3)
        for mm in cluster[1:]:
            for atommx in mm.atomDic:
                grad += LJGradbt2atoms(atom0x,atommx)
        grads.append(grad)
    return np.array(grads)
def Pot_central(cluster):
    m0 = cluster[0]
    potential = 0
    for atom0x in m0.atomDic:
        for mm in cluster[1:-3]:
            for atommx in mm.atomDic:
                potential += LJPotbt2atoms(atom0x,atommx)
    return potential/27.2113834/23.0605        
def Grad_cluster(cluster):
    m0 = cluster[0]
    grads = []
    for mm in cluster[1:]:
        grad_each_mol = []
        for atommx in mm.atomDic:
            grad = np.zeros(3)
            for atom0x in m0.atomDic:
                grad += LJGradbt2atoms(atommx,atom0x)
            grad_each_mol.append(grad)
        grads.append(grad_each_mol)
    return np.array(grads)

def Grad_LJ(cluster):
    gs = []
    gcenter = Grad_central(cluster)
    gcluster = Grad_cluster(cluster)
    avercenter = gcenter.mean(axis=0)
    avercluster = gcluster.mean(axis=1)
    for i in range(len(gcenter)):
        g = -avercenter
        g += gcenter[i,:]
        for j in range(len(cluster)-1):
#            g += np.transpose(cluster[0].getRotationMatrix(cluster[j])).dot(
#            gcluster[j,i,:]-avercluster[j,:])
            g += np.linalg.inv(cluster[j+1].matrix).dot(
            gcluster[j,i,:]-avercluster[j,:])
        gs.append(g)
    #change to be Eh/angstrom
    return np.array(gs)/27.2113834/23.0605*0.5
    
def Grad_pc(targetdir,cluster):
    pcgradfiles = [f for f in os.listdir(targetdir) if f[-6:]=="pcgrad"]
    pcgradfiles.sort()
    pcgradfile = pcgradfiles[-1]
    molNum_cluster = len(cluster)-1
    atomNum_eachmol = len(cluster[0].atomDic)
    grads = np.zeros((molNum_cluster,atomNum_eachmol,3))
    with open(targetdir+"/"+pcgradfile,"r") as pcgrad:
        pcgrad.readline()
        line = pcgrad.readline().strip()
        for i in range(molNum_cluster):
            for j in range(atomNum_eachmol):
                grads[i,j,:] = np.array([float(comp) for comp in line.split(" ") if len(comp)>0])
                line = pcgrad.readline().strip()
    avercluster = grads.mean(axis=1)  
    gs = []
    for i in range(atomNum_eachmol):
        g = 0
        for j in range(len(cluster)-1):
#            g += np.transpose(cluster[0].getRotationMatrix(cluster[j])).dot(
#            grads[j,i,:]-avercluster[j,:])
            g += np.linalg.inv(cluster[j+1].matrix).dot(
            grads[j,i,:]-avercluster[j,:])
        gs.append(g)
    return 0.5*np.array(gs)/0.5291772083

def Grad_gas_plus_QM(targetdir,cluster):
    files = [f for f in os.listdir(targetdir) if f[-6:]=="engrad"]
    files.sort()
    gasfile = files[0]
    QMfile = files[-1]
    atomNum_eachmol = len(cluster[0].atomDic)
    
    grads_gas = np.zeros((atomNum_eachmol,3))
    with open(targetdir+"/"+gasfile,"r") as gas:
        for i in range(11):
            gas.readline()
        g = gas.readline().strip()
        for i in range(atomNum_eachmol):
            for j in range(3):
                grads_gas[i,j] = float(g)
                g = gas.readline().strip()  
    
    grads_gas -= grads_gas.mean(axis=0)
    grads_QM = np.zeros((atomNum_eachmol,3))
    with open(targetdir+"/"+QMfile,"r") as QM:
        for i in range(11):
            QM.readline()
        g = QM.readline().strip()
        for i in range(atomNum_eachmol):
            for j in range(3):
                grads_QM[i,j] = float(g)
                g = QM.readline().strip()
    grads_QM -= grads_QM.mean(axis=0)
    grads = grads_gas+grads_QM
#    QM_cluster = grads*(len(cluster)-1)
#    return (grads+0.5*QM_cluster)/0.5291772083
    return grads/0.5291772083*0.5
#    return grads_gas/0.5291772083
    
    

def Grad_sum(targetdir,cluster):
    cluster = cluster[:-3]
    return Grad_gas_plus_QM(targetdir,cluster)+Grad_LJ(cluster)+Grad_pc(targetdir,cluster)#Eh/A
#    return Grad_gas_plus_QM(targetdir)





















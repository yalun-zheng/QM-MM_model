import numpy as np
import os
from gradient2 import LJParas,LJGradient

# (r0 in angstom,epsilon in kcal/mol)

def cartesian2frac(mol,coordinate):
    """
    change fractional coordinate to become cartesian coordinate.
    """
    matrix = np.array([[mol.a,mol.b*np.cos(mol.gamma),mol.c*np.cos(mol.beta)],
    [0,mol.b*np.sin(mol.gamma),mol.c*(np.cos(mol.alpha)-np.cos(mol.beta)*np.cos(mol.gamma))/np.sin(mol.gamma)],[0,0,
     mol.c*np.sqrt(np.sin(mol.beta)**2-((np.cos(mol.alpha)-np.cos(mol.beta)*np.cos(mol.gamma))/np.sin(mol.gamma))**2)]])
    return np.linalg.inv(matrix).dot(coordinate)

def LJGradbt2atoms(atom0,atomx):
    r0 = LJParas[atom0.type][0]+LJParas[atomx.type][0]
#    r0 = 2*np.sqrt(LJParas[atom0.type][0]*LJParas[atomx.type][0])
    epsilon = np.sqrt(LJParas[atom0.type][1]*LJParas[atomx.type][1])
    r_v = atom0.coordinate-atomx.coordinate
    r = np.linalg.norm(r_v)
    return LJGradient(r0,epsilon,r)*r_v/r

def Grad_central(cluster):
    m0 = cluster[0]
    grads = []
    for atom0x in m0.atomDic:
        grad = np.zeros(3)
        for mm in cluster[1:]:
            for atommx in mm.atomDic:
                grad += LJGradbt2atoms(atom0x,atommx)
        grads.append(grad)
    return np.array(grads)/27.2113834/23.0605

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

def stress_LJ(cluster):
    gcluster = Grad_cluster(cluster)
    stress_matrix = np.zeros((3,3))
    for i in range(len(gcluster)):
        g = np.zeros(3)
        for j in range(len(cluster[0].atomDic)):
            g += gcluster[i,j]
        stress_matrix += np.outer(cluster[1+i].center(),g)
    #change to be Eh/angstrom
    return stress_matrix/27.2113834/23.0605
            
    
def stress_pc(targetdir,cluster):
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
    stress_matrix = np.zeros((3,3))
    for i in range(len(grads)):
        g = np.zeros(3)
        for j in range(len(cluster[0].atomDic)):
            g += grads[i,j]
        stress_matrix += np.outer(cluster[1+i].center(),g/0.5291772083)
    #change to be Eh/angstrom
    return stress_matrix
    
def stress_center(targetdir,cluster):
    files = [f for f in os.listdir(targetdir) if f[-6:]=="engrad"]
    files.sort()
    QMfile = files[-1]
    atomNum_eachmol = len(cluster[0].atomDic)
    
    grads_QM = np.zeros((atomNum_eachmol,3))
    with open(targetdir+"/"+QMfile,"r") as QM:
        for i in range(11):
            QM.readline()
        g = QM.readline().strip()
        for i in range(atomNum_eachmol):
            for j in range(3):
                grads_QM[i,j] = float(g)
                g = QM.readline().strip()
    grads = grads_QM+Grad_central(cluster)*0.5291772083#-gradsum of gaseous mol
    stress_matrix = np.zeros((3,3))
    g = np.zeros(3)
    for j in range(len(cluster[0].atomDic)):
        g += grads[j]
    stress_matrix += np.outer(cluster[0].center(),g/0.5291772083)
    #change to be Eh/angstrom
    return stress_matrix

def stress_sum(targetdir,cluster):
    cluster = cluster[:-3]
    mol = cluster[0]
    coords_cell = np.array([[mol.a,mol.b*np.cos(mol.gamma),mol.c*np.cos(mol.beta)],
    [0,mol.b*np.sin(mol.gamma),mol.c*(np.cos(mol.alpha)-np.cos(mol.beta)
    *np.cos(mol.gamma))/np.sin(mol.gamma)],[0,0,mol.c*np.sqrt(np.sin(mol.beta)
    **2-((np.cos(mol.alpha)-np.cos(mol.beta)*np.cos(mol.gamma))/np.sin(mol.gamma))
    **2)]]).T.reshape(9)
    V = np.cross(coords_cell.reshape(3,3)[0],coords_cell.reshape(3,3)[1]
    ).dot(coords_cell.reshape(3,3)[2])
    #in Eh/angstrom
    return 0.5*(stress_LJ(cluster)+stress_pc(targetdir,cluster))/V


















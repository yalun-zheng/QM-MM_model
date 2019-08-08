import toolkit as tk
import numpy as np
from rfotest import get_RFOStep
import os
from execute import start
import matplotlib.pyplot as plt
import sys


from gradient_x import Pot_central,Grad_sum
from stressastt_x import stress_sum



def voigt(matrix):
    voigt = []
    for i in range(3):
        voigt.append(matrix[i,i])
    voigt.append((matrix[1,2]+matrix[2,1])/2)
    voigt.append((matrix[0,2]+matrix[2,0])/2)
    voigt.append((matrix[1,0]+matrix[0,1])/2)
    return np.array(voigt)

def voigt2matrix(voigt):
    matrix = np.zeros((3,3))
    for i in range(3):
        matrix[i,i] = voigt[i]
    matrix[1,2] = voigt[3]
    matrix[2,1] = voigt[3]
    matrix[0,2] = voigt[4]
    matrix[2,0] = voigt[4]
    matrix[0,1] = voigt[5]
    matrix[1,0] = voigt[5]
    return matrix

def update_cellParas(cluster,strain):
    for mol in cluster:
        mol.translate(strain.dot(mol.center()))
    a = np.linalg.norm(cluster[-3].center())
    b = np.linalg.norm(cluster[-2].center())
    c = np.linalg.norm(cluster[-1].center())
    alpha = np.arccos(cluster[-2].center().dot(cluster[-1].center())/b/c)
    beta = np.arccos(cluster[-3].center().dot(cluster[-1].center())/a/c)
    gamma = np.arccos(cluster[-3].center().dot(cluster[-2].center())/a/b)
    for mol in cluster:
        mol.setCellpara(a,b,c,alpha,beta,gamma)

def update_molCoords(cluster,s):
    for mol in cluster[:-3]:
        for j in range(len(mol.atomDic)):
            mol.atomDic[j].translate(mol.matrix.dot(s[j*3:3*(j+1)]))
            
def run(targetdir_old,radius,qm_command,charge,spin,program_path):            
#    root = "/home/yalun/Desktop/trans_nosolvent/"
#    file=os.listdir(root)
#    for f in sorted(file):
#        targetdir_old = os.path.join(root,f)
    radius = int(radius)
    cluster = np.load(targetdir_old+"/"+[f for f in os.listdir(targetdir_old) 
    if f[-3:]=="npy"][0])
    start(targetdir_old,cluster,radius,qm_command,program_path,charge,spin)
    outfiles = [f for f in os.listdir(targetdir_old) if f[-3:]=="out"]
    outfiles.sort()
    Egas = tk.read_energy(os.path.join(targetdir_old,outfiles[0]))
    E = 0.5*(tk.read_energy(os.path.join(targetdir_old,outfiles[-1]))+Egas
        +Pot_central(cluster))
    targetdir = targetdir_old
    Eref = E + 1
    fileTag = 1
    mol = cluster[0]
    coords_cell = np.array([[mol.a,mol.b*np.cos(mol.gamma),mol.c*np.cos(mol.beta)],
    [0,mol.b*np.sin(mol.gamma),mol.c*(np.cos(mol.alpha)-np.cos(mol.beta)
    *np.cos(mol.gamma))/np.sin(mol.gamma)],[0,0,mol.c*np.sqrt(np.sin(mol.beta)
    **2-((np.cos(mol.alpha)-np.cos(mol.beta)*np.cos(mol.gamma))/np.sin(mol.gamma))
    **2)]])
    V0 = np.cross(coords_cell[:,0],coords_cell[:,1]).dot(coords_cell[:,2])

    coords_cell = np.array([0]*6)
    coords_mol = np.array([0]*(3*len(cluster[0].atomDic)))
    
    hess_cell = np.eye(len(coords_cell))
    hess_mol = np.eye(len(coords_mol))

    E_array = []
    E_array.append(E)
    E_array = E_array*3
#    print(f)
    stress_matrix2 = 1
    grad_matrix2 = 1
        
    while (np.abs(E_array[-1]-E_array[-2])>1e-6 or np.linalg.norm(grad_matrix2
           )>1e-4/0.5291772083) or np.linalg.norm(stress_matrix2
            )>1e-4/0.5291772083/265 or np.abs(E_array[-3]-E_array[-2])>1e-6:
        Eref = E
        stress_matrix = voigt(stress_sum(targetdir,cluster))
        s_cell = get_RFOStep(hess_cell,stress_matrix,coords_cell,0.3)
        strain = voigt2matrix(s_cell)
        update_cellParas(cluster,strain)
        os.system("mkdir "+targetdir_old+"/"+"cellupdate"+str(fileTag))
        targetdir = targetdir_old+"/"+"cellupdate"+str(fileTag)+"/"+cluster[0].name
        os.system("mkdir "+targetdir)
        cluster[0].getxyzFile(targetdir)
        tk.cluster_getxyzFile(cluster,radius,targetdir)
        start(targetdir,cluster,radius,qm_command,program_path,charge,spin)
        outfiles = [f for f in os.listdir(targetdir) if f[-3:]=="out"]
        outfiles.sort()
        E = 0.5*(tk.read_energy(targetdir+"/"+outfiles[-1])+tk.read_energy(
                targetdir+"/"+outfiles[0])+Pot_central(cluster))
        E_array.append(E)
        stress_matrix2 = voigt(stress_sum(targetdir,cluster))
        y = stress_matrix2-stress_matrix
        hess_cell = hess_cell + y.reshape(len(coords_cell),1)*y/y.dot(s_cell)-(
        hess_cell.dot(s_cell).reshape(len(coords_cell),1)*(hess_cell.dot(s_cell)))/(
        s_cell.dot(hess_cell.dot(s_cell)))
        
        grad_inner = Grad_sum(targetdir,cluster)
        grad_matrix = grad_inner.reshape(len(grad_inner)*3)
        s_mol = get_RFOStep(hess_mol,grad_matrix,coords_mol,0.3)
        update_molCoords(cluster,s_mol)
        os.system("mkdir "+targetdir_old+"/"+"molupdate"+str(fileTag))
        targetdir = targetdir_old+"/"+"molupdate"+str(fileTag)+"/"+cluster[0].name
        
        os.system("mkdir "+targetdir)
        cluster[0].getxyzFile(targetdir)
        tk.cluster_getxyzFile(cluster,radius,targetdir)
        start(targetdir,cluster,radius,qm_command,program_path,charge,spin)
        outfiles = [f for f in os.listdir(targetdir) if f[-3:]=="out"]
        outfiles.sort()
        E = 0.5*(tk.read_energy(targetdir+"/"+outfiles[-1])+tk.read_energy(
                targetdir+"/"+outfiles[0])+Pot_central(cluster))
        E_array.append(E)  
        grad_inner2 = Grad_sum(targetdir,cluster)
        grad_matrix2 = grad_inner2.reshape(len(grad_inner2)*3)
        y = grad_matrix2-grad_matrix

        hess_mol = hess_mol + y.reshape(len(coords_mol),1)*y/y.dot(s_mol)-(
        hess_mol.dot(s_mol).reshape(len(coords_mol),1)*(hess_mol.dot(s_mol)))/(
        s_mol.dot(hess_mol.dot(s_mol)))
        fileTag += 1
#        print("E-Eold",E-Eref)
#        plt.plot(range(len(E_array)-2),(np.array(E_array[2:])-E_array[0])*27.2113834*23.0605*4.184 )
#        plt.xlabel("optimizing process")
#        plt.ylabel("(E-E0)/(kJ/mol)")
#        plt.tight_layout()
#        plt.show()
#        print("|grad|:",str(np.linalg.norm(grad_matrix)))
#        print("cell paras: ",cluster[0].a,cluster[0].b,cluster[0].c,cluster[0].alpha,cluster[0].beta,cluster[0].gamma)
#        print("---------------------------------------")
        
    np.save(targetdir_old+"/"+str(radius)+"finalcluster.npy",cluster)
    np.save(targetdir_old+"/"+str(radius)+"energy.npy",E_array)
    plt.plot(range(len(E_array)-2),(np.array(E_array[2:])-E_array[0])*27.2113834*23.0605*4.184 )
    plt.xlabel("optimizing process")
    plt.ylabel("(E-E0)/(kJ/mol)")
    plt.tight_layout()
    plt.savefig(targetdir_old+"/energy_profile.png")
#    plt.show()
#    tk.cluster_getxyzFile(cluster,radius,targetdir_old)
    mol = cluster[0]
    coords_cell = np.array([[mol.a,mol.b*np.cos(mol.gamma),mol.c*np.cos(mol.beta)],
    [0,mol.b*np.sin(mol.gamma),mol.c*(np.cos(mol.alpha)-np.cos(mol.beta)
    *np.cos(mol.gamma))/np.sin(mol.gamma)],[0,0,mol.c*np.sqrt(np.sin(mol.beta)
    **2-((np.cos(mol.alpha)-np.cos(mol.beta)*np.cos(mol.gamma))/np.sin(mol.gamma))
    **2)]])
    V = np.cross(coords_cell[:,0],coords_cell[:,1]).dot(coords_cell[:,2])
    with open(targetdir_old+"/res.txt","w") as res:
      #  res.write("Egas: "+str(Egas)+"\n")
      #  res.write("E0: "+str(E_array[0])+"\n")
        res.write("E: "+str(E_array[-1])+"\n")
      #  res.write("deltaE: "+str(E_array[-1]-E_array[0])+"Eh\n")
      #  res.write("Ecohexp: "+str(E_array[0]-Egas)+"\n")
      #  res.write("Ecoh: "+str(E_array[-1]-Egas)+"\n")
      #  res.write("detaEcoh: "+str(E_array[-1]-E_array[0])+"\n")
        res.write("V0: "+str(V0)+"\n")
        res.write("V: "+str(V)+"\n")
        res.write("a: "+str(cluster[0].a)+"\n")
        res.write("b: "+str(cluster[0].b)+"\n")
        res.write("c: "+str(cluster[0].c)+"\n")
        res.write("alpha: "+str(cluster[0].alpha*180/np.pi)+"\n")
        res.write("beta: "+str(cluster[0].beta*180/np.pi)+"\n")
        res.write("gamma: "+str(cluster[0].gamma*180/np.pi)+"\n")
    
if __name__=="__main__":
    run(*sys.argv[1:7])

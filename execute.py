#!usr/bin/python3
import toolkit as tk
import numpy as np
import os
from molecule import Molecule
from atom import Atom

def generate_a_certain_cluster(xyz_path,name,spg,a,b,c,alpha,beta,gamma,targetDir,radius):
    original = read_xyz(xyz_path,name,spg,a,b,c,alpha,beta,gamma)
    os.system("mkdir "+targetDir+"/"+original.name)
    single_mols = tk.searchSingleMol2(original)
    cluster = tk.singleMol2cluster(single_mols,radius) #for P 21 21 21 space group, default case="1"
    cluster[0].getxyzFile(targetDir+"/"+original.name)
    np.save(targetDir+"/"+original.name+"/"+str(radius)+"cluster.npy",cluster)
    tk.cluster_getxyzFile(cluster,radius,targetDir+"/"+original.name)
def read_xyz(path,name,spg,a,b,c,alpha,beta,gamma):
    """
    from C21 database.txt extract infomation to form 21(22) molecules/fragemets
    in a unit cell and return a list of them
    """
    with open(path,"r") as data:
        line = data.readline().strip()
        line = data.readline().strip()
        line = data.readline().strip()
        mol = Molecule()
        while line:
            info = []
            for part in line.split(" "):
                if len(part)>0:
                    info.append(part)
            mol.addAtom(Atom(typ=info[0],coordinate=np.array([float(info[1])
            ,float(info[2]),float(info[3])])))
            line = data.readline().strip()
        mol.setCellpara(a,b,c,alpha,beta,gamma)
        mol.setSpg(spg)
        mol.setName(name)
    return mol
    
def generate_benchmark_clusters(c21path,targetDir,radius):
    """
    generate original directories and xyz files for the following calculations
    molecule.xyz, cluster.npy
    """
    original = tk.read_c21database(c21path)
    for mol in original:
        os.system("mkdir "+targetDir+"/"+mol.name)
        single_mols = tk.searchSingleMol(mol)
        cluster = tk.singleMol2cluster(single_mols,radius) #for P 21 21 21 space group, default case="1"
        cluster[0].getxyzFile(targetDir+"/"+mol.name)
        np.save(targetDir+"/"+mol.name+"/"+str(radius)+"cluster.npy",cluster)
        tk.cluster_getxyzFile(cluster,radius,targetDir+"/"+mol.name)

    
def start(targetDir,cluster,radius,cmd,program_path,charge,spin,chargeupdate=True):
    if chargeupdate:
        if targetDir[-1]=="/":
            mol_name = targetDir[:-1].split("/")[-1]
        else:
            mol_name = targetDir.split("/")[-1]
        path = targetDir+"/"+mol_name
        tk.xyz2inp(path+"_fromClass.xyz","orca",cmd, "",False,radius,1,chg=charge,spin=spin)
        os.system(program_path+" "
                  +path+str(1)+".inp"+">"+path+str(1)+".out")
        charges = np.array(tk.updateCluster(path+"_fromClass.xyz",cluster,radius,1))
        i = 2
        last_charges = 0
        while (np.abs(last_charges-charges)>1e-3).any():
            last_charges = charges
            tk.xyz2inp(path+"_fromClass.xyz","orca",cmd, "",True,radius,i,chg=charge,spin=spin)
            os.system(program_path+" "
                      +path+str(i)+".inp"+">"+path+str(i)+".out")
            charges = np.array(tk.updateCluster(path+"_fromClass.xyz",cluster,radius,i))
            i += 1
    else:
        if targetDir[-1]=="/":
            mol_name = targetDir[:-1].split("/")[-1]
        else:
            mol_name = targetDir.split("/")[-1]
        path = targetDir+"/"+mol_name
        tk.xyz2inp(path+"_fromClass.xyz","orca",cmd, "",True,radius,1,chg=charge,spin=spin)
        os.system(program_path+" "
                  +path+str(1)+".inp"+">"+path+str(1)+".out")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

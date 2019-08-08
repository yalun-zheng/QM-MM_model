import numpy as np
import copy
from toolkit import cif2molecule,frac2cartesian,singleMol2cluster2,cluster_getxyzFile,spgDic
from toolkit import cartesian2frac,xyz2inp
import os
import sys

root = input("root_directory(where you will store all your generated files):\n")
print("-----------------------------------------------------------------------")
cif = input("Please input the path of .cif file:\n")
print("-----------------------------------------------------------------------")
def applyspg(mols,case="1"):
    """
    apply all symmetric operation from the space group to mol and generate a small cluster
    """
    a = mols.a
    b = mols.b
    c = mols.c
    alpha = mols.alpha
    beta = mols.beta
    gamma = mols.gamma
    matrix = np.array([[a,b*np.cos(gamma),c*np.cos(beta)],
    [0,b*np.sin(gamma),c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma)],[0,0,
    c*np.sqrt(np.sin(beta)**2-((np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma))**2)]])
    try:
        operation = spgDic[mols.spg]
    except:
        try:
            operation = spgDic[mols.spg+" "+case]
        except:
            raise ValueError
    operation[0].insert(0,np.eye(3))
    operation[1].insert(0,np.array([0,0,0]))
    minimal_cluster = []
    for mol in [mols]:
        buffer = copy.deepcopy(mol)
        buffer_original = copy.deepcopy(mol)
        minimal_cluster.append(mol)
        for i in range(len(operation[1])):
            if_add = True
            for atom in buffer.atomDic:
                atom.coordinate = frac2cartesian(a,b,c,alpha,beta,gamma,
                operation[0][i].dot(cartesian2frac(buffer,atom.coordinate))
                +operation[1][i])
            buffer.setMatrix(((matrix.dot(operation[0][i])).dot(np.linalg.inv(
            matrix))).dot(buffer.matrix))
            for m in minimal_cluster:
                if np.dot((m.center()-buffer.center()),(m.center()-buffer.center()))<25e-6:
                    if_add = False
            if if_add:
                minimal_cluster.append(buffer)
            buffer = copy.deepcopy(buffer_original)
            
    for mol in minimal_cluster:
        mol.translate(frac2cartesian(a,b,c,alpha,beta,gamma,-np.floor(cartesian2frac(
                minimal_cluster[0],mol.center()))))
    for m in minimal_cluster[1:]:
        for a in m.atomDic:
            minimal_cluster[0].atomDic.append(a)
    return minimal_cluster[0]

asymunit = cif2molecule(cif)
asymunit = applyspg(asymunit)

asymunit.name=cif.split("/")[-1][:-4]
a=asymunit.a
b=asymunit.b
c=asymunit.c
alpha=asymunit.alpha
beta=asymunit.beta
gamma=asymunit.gamma

translate_v = [np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]
for i in range(3):
    buffer = copy.deepcopy(asymunit)
    asymunit.translate(frac2cartesian(a,b,c,
            alpha,beta,gamma,translate_v[i]))
    for atom in buffer.atomDic:
        asymunit.addAtom(atom)
asymunit.getxyzFile(targetDic=root)
exam = "n"
def readdat(datpath):
    data = []
    qmlabels = []
    for line in open(datpath):
        if line[0]!="#" and (line[0]!="\n"):
            if "," in line and line[:3]!="PBS":
                qmlabels+=[int(part) for part in line.strip().split(",") if len(part)>0]
            else:
                data.append(line.strip())
    data.insert(0,qmlabels)            
    return data

while exam.lower()!="y":
    dat = input("please check .xyz file your root directory, write and assign your .dat file:\n")
    print("-----------------------------------------------------------------------")
    data = readdat(dat)
    qm_label = data[0]
    mm_model = data[1]
    qm_program = data[2]
    qm_command = data[3]
    charge = data[4]
    spin = data[5]
    radius = int(float(data[6]))
    if data[7].lower()=="yes":
        opt = True
    else:
        opt = False
    program_path=data[8]
    if data[9].lower()=="local":
        local = True
    else: local = False
    opt_path = data[10]
    sys.path.append(opt_path)
    pbs = "#"+data[11]
    queue = data[12]
    anaconda = data[13]
    exam = input("make sure that QM_labels are right (y/n)\n"+str(qm_label)+"\n").strip()
    print("-----------------------------------------------------------------------")


central = copy.deepcopy(asymunit)
central.atomDic = []


def optshell():
    ends = "\n"
    dirname = asymunit.name
    with open(root+""+dirname+".sh","w") as sh:
        sh.writelines([r"#!/bin/bash",ends,
                       pbs,ends,
                       "#PBS -q %s"%queue,ends,
                       r"source /etc/profile.d/modules.sh",ends,
                       "module load orca_4_1_1",ends,
                       "module load openmpi/3.1.3",ends,
                       "WORKDIR=\""+os.path.join(root,dirname)+"\"",ends,
                       "echo \"DBG: Running on ${HOSTNAME}\"",ends,
                       "export RSH_COMMAND=\"ssh\"",ends,
                       "DIRNAME=$(date +%s)_pythonrun",ends,
                       r"mkdir -p /scr/${USER}/${DIRNAME}",ends,
                       "cd ${WORKDIR}",ends,
                       r"cp --preserve *  /scr/${USER}/${DIRNAME}/",ends,
                       r"cd /scr/${USER}/${DIRNAME}/",ends,
                       program_path+" "+gas_root+"/"+"gasPhase"+str(1)+".inp"+">"+gas_root+"/"+"gasPhase"+str(1)+".out",ends,
                       r"rm /scr/${USER}/${DIRNAME}/*.tmp*",ends,
                       "mv * ${WORKDIR}",ends,
                       "cd ..",ends,
                       r"rm -rf /scr/${USER}/${DIRNAME}",ends,
                       r"rm ${WORKDIR}/*.sh.*",])
    os.system("qsub "+dirname+".sh")
#    os.system("rm "+dirname+".sh")
def shell(script_name):
    ends = "\n"
    dirname = asymunit.name
    with open(root+"/"+dirname+".sh","w") as sh:
        sh.writelines([r"#!/bin/bash",ends,
                       pbs,ends,
                       "#PBS -q %s"%queue,ends,
                       r"source /etc/profile.d/modules.sh",ends,
                       "module load orca_4_1_1",ends,
                       "module load openmpi/3.1.3",ends,
                       "WORKDIR=\""+os.path.join(root,dirname)+"\"",ends,
                       "echo \"DBG: Running on ${HOSTNAME}\"",ends,
                       "export RSH_COMMAND=\"ssh\"",ends,
                       "DIRNAME=$(date +%s)_pythonrun",ends,
                       r"mkdir -p /scr/${USER}/${DIRNAME}",ends,
                       "cd ${WORKDIR}",ends,
                       r"cp --preserve *  /scr/${USER}/${DIRNAME}/",ends,
                       r"cd /scr/${USER}/${DIRNAME}/",ends,
                       anaconda+" "+script_name+" "+mol_root+" "+str(radius)+" "+'"'+qm_command+'"'+" "+charge+" "+spin+" "+program_path,ends,
                       r"rm /scr/${USER}/${DIRNAME}/*.tmp*",ends,
                       "mv * ${WORKDIR}",ends,
                       "cd ..",ends,
                       r"rm -rf /scr/${USER}/${DIRNAME}",ends,
                       r"rm ${WORKDIR}/*.sh.*",])
    os.system("qsub "+dirname+".sh")
#    os.system("rm "+dirname+".sh")
    
for i in qm_label:
    central.atomDic.append(copy.deepcopy(asymunit.atomDic[i-1]))

cluster = singleMol2cluster2(central,radius) #for P 21 21 21 space group, default case="1"
if qm_program.lower()=="orca":
    if opt:
        if "ENGRAD" in qm_command.upper():
            i = qm_command.upper().index("ENGRAD")
            qm_command = qm_command[:i]+qm_command[i+7:]
        gas_root = os.path.join(root,"gasPhase")
        os.system("mkdir "+gas_root)
        cluster[0].getxyzFile(os.path.join(gas_root))
        cluster_getxyzFile(cluster,radius,gas_root)
        xyz2inp(gas_root+"/"+asymunit.name+"_fromClass.xyz",qm_program,qm_command+
                " Opt", "",False,radius,1,chg=charge,spin=spin)
        if local:
            os.system(program_path+" "+gas_root+"/"+"gasPhase"+str(1)+".inp"+">"+
                      gas_root+"/"+"gasPhase"+str(1)+".out")
        else:
            optshell()
    else:
        mol_root = os.path.join(root,asymunit.name)
        os.system("mkdir "+mol_root)
        cluster[0].getxyzFile(os.path.join(mol_root))
        cluster_getxyzFile(cluster,radius,mol_root)
        np.save(mol_root+"/"+str(radius)+"cluster.npy",cluster)
        if local:
            if mm_model.lower()=="lennard-jones":
                from combined_as_tt import run
                run(mol_root,radius,qm_command,charge,spin,program_path)
            elif mm_model.lower()=="buckingham":
                from combined_new import run
                run(mol_root,radius,qm_command,charge,spin,program_path)
            elif mm_model.lower()=="potential_x":
                from combined_x import run
                run(mol_root,radius,qm_command,charge,spin,program_path)
            else:
                print("unknown mm_model! Try Lennard-Jones or Buckingham!")
        else:
            if mm_model.lower()=="lennard-jones":
                shell(opt_path+"/combined_as_tt.py")
            elif mm_model.lower()=="buckingham":
                shell(opt_path+"/combined_new.py")
            elif mm_model.lower()=="potential_x":
                shell(opt_path+"/combined_x.py")
            else:
                print("unknown mm_model! Try Lennard-Jones or Buckingham!")
        
    
else:
    print("*************with Gaussian??? No! Pls orca!**************")






















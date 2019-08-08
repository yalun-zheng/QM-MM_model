#!/home/yalun/anaconda3/bin/python3
from molecule import Molecule
from atom import Atom
import numpy as np
import copy

"""
define all used symm. operations of space groups here as a dictionary of 2d lists
except for Identity operation
"""
spgDic = {"P 1":[[],[]],
"P 21":[[np.array([[-1,0,0],[0,1,0],[0,0,-1]])],[np.array([0,0.5,0])]],
"P n a 21":[[np.array([[-1,0,0],[0,1,0],[0,0,1]]),np.array([[1,0,0],[0,-1,0],[0,0,1]]),
         np.array([[-1,0,0],[0,-1,0],[0,0,1]])],[np.array([0.5,0.5,0.5]),np.array([0.5,0.5,0]),np.array([0,0,0.5])]],
"P -4 21 c":[[np.array([[-1,0,0],[0,-1,0],[0,0,1]]),np.array([[0,1,0],[-1,0,0],[0,0,-1]]),
         np.array([[0,-1,0],[1,0,0],[0,0,-1]]),np.array([[1,0,0],[0,-1,0],[0,0,-1]]),
         np.array([[-1,0,0],[0,1,0],[0,0,-1]]),np.array([[0,1,0],[1,0,0],[0,0,1]]),
         np.array([[0,-1,0],[-1,0,0],[0,0,1]])],[np.array([0,0,0]),np.array([0,0,0]),
         np.array([0,0,0]),np.array([0.5,0.5,0.5]),np.array([0.5,0.5,0.5]),np.array([0.5,0.5,0.5]),
         np.array([0.5,0.5,0.5])]],
"P 21 3":[[np.array([[1,0,0],[0,-1,0],[0,0,-1]]),np.array([[-1,0,0],[0,1,0],[0,0,-1]]),
         np.array([[-1,0,0],[0,-1,0],[0,0,1]]),np.array([[0,0,1],[1,0,0],[0,1,0]]),
         np.array([[0,0,-1],[-1,0,0],[0,1,0]]),np.array([[0,0,1],[-1,0,0],[0,-1,0]]),
         np.array([[0,0,-1],[1,0,0],[0,-1,0]]),np.array([[0,1,0],[0,0,1],[1,0,0]]),
         np.array([[0,-1,0],[0,0,1],[-1,0,0]]),np.array([[0,-1,0],[0,0,-1],[1,0,0]]),
         np.array([[0,1,0],[0,0,-1],[-1,0,0]])],[np.array([0.5,0.5,0]),np.array([0,0.5,0.5]),
         np.array([0.5,0,0.5]),np.array([0,0,0]),np.array([0.5,0,0.5]),np.array([0.5,0.5,0]),
         np.array([0,0.5,0.5]),np.array([0,0,0]),np.array([0,0.5,0.5]),np.array([0.5,0,0.5]),np.array([0.5,0.5,0])]],
"P 21/a":[[np.array([[-1,0,0],[0,1,0],[0,0,-1]]),np.array([[-1,0,0],[0,-1,0],[0,0,-1]]),
         np.array([[1,0,0],[0,-1,0],[0,0,1]])],[np.array([0.5,0.5,0]),np.array([0,0,0]),np.array([0.5,0.5,0])]],
"P b c a":[[np.array([[-1,0,0],[0,1,0],[0,0,1]]),np.array([[1,0,0],[0,-1,0],[0,0,1]]),
         np.array([[1,0,0],[0,1,0],[0,0,-1]]),np.array([[-1,0,0],[0,-1,0],[0,0,-1]]),
         np.array([[1,0,0],[0,-1,0],[0,0,-1]]),np.array([[-1,0,0],[0,1,0],[0,0,-1]]),
         np.array([[-1,0,0],[0,-1,0],[0,0,1]])],[np.array([0.5,0.5,0]),
         np.array([0,0.5,0.5]),np.array([0.5,0,0.5]),np.array([0,0,0]),np.array([0.5,0.5,0]),
         np.array([0,0.5,0.5]),np.array([0.5,0,0.5])]],
"P a -3":[[np.array([[1,0,0],[0,-1,0],[0,0,-1]]),np.array([[-1,0,0],[0,1,0],[0,0,-1]]),
         np.array([[-1,0,0],[0,-1,0],[0,0,1]]),np.array([[0,0,1],[1,0,0],[0,1,0]]),
         np.array([[0,0,-1],[-1,0,0],[0,1,0]]),np.array([[0,0,1],[-1,0,0],[0,-1,0]]),
         np.array([[0,0,-1],[1,0,0],[0,-1,0]]),np.array([[0,1,0],[0,0,1],[1,0,0]]),
         np.array([[0,-1,0],[0,0,1],[-1,0,0]]),np.array([[0,-1,0],[0,0,-1],[1,0,0]]),
         np.array([[0,1,0],[0,0,-1],[-1,0,0]]),np.array([[-1,0,0],[0,-1,0],[0,0,-1]]),
         np.array([[-1,0,0],[0,1,0],[0,0,1]]),np.array([[1,0,0],[0,-1,0],[0,0,1]]),
         np.array([[1,0,0],[0,1,0],[0,0,-1]]),np.array([[0,0,-1],[-1,0,0],[0,-1,0]]),
         np.array([[0,0,1],[1,0,0],[0,-1,0]]),np.array([[0,0,-1],[1,0,0],[0,1,0]]),
         np.array([[0,0,1],[-1,0,0],[0,1,0]]),np.array([[0,-1,0],[0,0,-1],[-1,0,0]]),
         np.array([[0,1,0],[0,0,-1],[1,0,0]]),np.array([[0,1,0],[0,0,1],[-1,0,0]]),
         np.array([[0,-1,0],[0,0,1],[1,0,0]])],[np.array([0.5,0.5,0]),np.array([0,0.5,0.5]),
         np.array([0.5,0,0.5]),np.array([0,0,0]),np.array([0.5,0,0.5]),np.array([0.5,0.5,0]),
         np.array([0,0.5,0.5]),np.array([0,0,0]),np.array([0,0.5,0.5]),np.array([0.5,0,0.5]),
         np.array([0.5,0.5,0]),np.array([0,0,0]),np.array([0.5,0.5,0]),np.array([0,0.5,0.5]),
         np.array([0.5,0,0.5]),np.array([0,0,0]),np.array([0.5,0,0.5]),np.array([0.5,0.5,0]),
         np.array([0,0.5,0.5]),np.array([0,0,0]),np.array([0,0.5,0.5]),np.array([0.5,0,0.5]),np.array([0.5,0.5,0])]],
"P 21 21 21 1":[[np.array([[1,0,0],[0,-1,0],[0,0,-1]]),np.array([[-1,0,0],[0,1,0],[0,0,-1]]),
         np.array([[-1,0,0],[0,-1,0],[0,0,1]])],[np.array([0.5,0.5,0]),np.array([0,0.5,0.5]),np.array([0.5,0,0.5])]],
"P 21 21 21 2":[[np.array([[1,0,0],[0,-1,0],[0,0,-1]]),np.array([[-1,0,0],[0,1,0],[0,0,-1]]),
         np.array([[-1,0,0],[0,-1,0],[0,0,1]])],[np.array([0.5,0,0.5]),np.array([0.5,0.5,0]),np.array([0,0.5,0.5])]],
"P -1":[[np.array([[-1,0,0],[0,-1,0],[0,0,-1]])],[np.array([0,0,0])]],
"P 21/n":[[np.array([[-1,0,0],[0,1,0],[0,0,-1]]),np.array([[-1,0,0],[0,-1,0],[0,0,-1]]),
         np.array([[1,0,0],[0,-1,0],[0,0,1]])],[np.array([0.5,0.5,0.5]),np.array([0,0,0]),np.array([0.5,0.5,0.5])]],
"P 21/c":[[np.array([[-1,0,0],[0,1,0],[0,0,-1]]),np.array([[-1,0,0],[0,-1,0],[0,0,-1]]),
         np.array([[1,0,0],[0,-1,0],[0,0,1]])],[np.array([0,0.5,0.5]),np.array([0,0,0]),np.array([0,0.5,0.5])]],
"P c a b":[[np.array([[-1,0,0],[0,1,0],[0,0,1]]),np.array([[1,0,0],[0,-1,0],[0,0,1]]),
         np.array([[1,0,0],[0,1,0],[0,0,-1]]),np.array([[-1,0,0],[0,-1,0],[0,0,-1]]),
         np.array([[1,0,0],[0,-1,0],[0,0,-1]]),np.array([[-1,0,0],[0,1,0],[0,0,-1]]),
         np.array([[-1,0,0],[0,-1,0],[0,0,1]])],[np.array([0.5,0,0.5]),np.array([0.5,0.5,0]),
         np.array([0,0.5,0.5]),np.array([0,0,0]),np.array([0.5,0,0.5]),np.array([0.5,0.5,0]),np.array([0,0.5,0.5])]],
"P m n n":[[np.array([[-1,0,0],[0,1,0],[0,0,1]]),np.array([[1,0,0],[0,-1,0],[0,0,1]]),
         np.array([[1,0,0],[0,1,0],[0,0,-1]]),np.array([[-1,0,0],[0,-1,0],[0,0,-1]]),
         np.array([[1,0,0],[0,-1,0],[0,0,-1]]),np.array([[-1,0,0],[0,1,0],[0,0,-1]]),
         np.array([[-1,0,0],[0,-1,0],[0,0,1]])],[np.array([0,0,0]),np.array([0.5,0.5,0.5]),
         np.array([0.5,0.5,0.5]),np.array([0,0,0]),np.array([0,0,0]),np.array([0.5,0.5,0.5]),np.array([0.5,0.5,0.5])]],
"P 21 c n":[[np.array([[1,0,0],[0,-1,0],[0,0,-1]]),np.array([[1,0,0],[0,-1,0],[0,0,1]]),
           np.array([[1,0,0],[0,1,0],[0,0,-1]])],[np.array([0.5,0,0]),np.array([0,0.5,0.5]),np.array([0.5,0.5,0.5])]],
"R -3 c":[[np.array([[0,-1,0],[1,-1,0],[0,0,1]]),np.array([[-1,1,0],[-1,0,0],[0,0,1]]),
          np.array([[0,-1,0],[-1,0,0],[0,0,1]]),np.array([[-1,1,0],[0,1,0],[0,0,1]]),
          np.array([[1,0,0],[1,-1,0],[0,0,1]]),np.array([[1,0,0],[0,1,0],[0,0,1]]),
          np.array([[0,-1,0],[1,-1,0],[0,0,1]]),np.array([[-1,1,0],[-1,0,0],[0,0,1]]),
          np.array([[0,-1,0],[-1,0,0],[0,0,1]]),np.array([[-1,1,0],[0,1,0],[0,0,1]]),
          np.array([[1,0,0],[1,-1,0],[0,0,1]]),np.array([[1,0,0],[0,1,0],[0,0,1]]),
          np.array([[0,-1,0],[1,-1,0],[0,0,1]]),np.array([[-1,1,0],[-1,0,0],[0,0,1]]),
          np.array([[0,-1,0],[-1,0,0],[0,0,1]]),np.array([[-1,1,0],[0,1,0],[0,0,1]]),
          np.array([[1,0,0],[1,-1,0],[0,0,1]]),np.array([[-1,0,0],[0,-1,0],[0,0,-1]]),
          np.array([[0,1,0],[-1,1,0],[0,0,-1]]),np.array([[1,-1,0],[1,0,0],[0,0,-1]]),
          np.array([[0,1,0],[1,0,0],[0,0,-1]]),np.array([[1,-1,0],[0,-1,0],[0,0,-1]]),
          np.array([[-1,0,0],[-1,1,0],[0,0,-1]]),np.array([[-1,0,0],[0,-1,0],[0,0,-1]]),
          np.array([[0,1,0],[-1,1,0],[0,0,-1]]),np.array([[1,-1,0],[1,0,0],[0,0,-1]]),
          np.array([[0,1,0],[1,0,0],[0,0,-1]]),np.array([[1,-1,0],[0,-1,0],[0,0,-1]]),
          np.array([[-1,0,0],[-1,1,0],[0,0,-1]]),np.array([[-1,0,0],[0,-1,0],[0,0,-1]]),
          np.array([[0,1,0],[-1,1,0],[0,0,-1]]),np.array([[1,-1,0],[1,0,0],[0,0,-1]]),
          np.array([[0,1,0],[1,0,0],[0,0,-1]]),np.array([[1,-1,0],[0,-1,0],[0,0,-1]]),np.array([[-1,0,0],[-1,1,0],[0,0,-1]])],
          [np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0.5]),np.array([0,0,0.5]),np.array([0,0,0.5]),np.array([2/3,1/3,1/3]),
          np.array([2/3,1/3,1/3]),np.array([2/3,1/3,1/3]),np.array([2/3,1/3,0.5+1/3]),
          np.array([2/3,1/3,0.5+1/3]),np.array([2/3,1/3,0.5+1/3]),np.array([1/3,2/3,2/3]),
          np.array([1/3,2/3,2/3]),np.array([1/3,2/3,2/3]),np.array([1/3,2/3,0.5+2/3]),
          np.array([1/3,2/3,0.5+2/3]),np.array([1/3,2/3,0.5+2/3]),np.array([0,0,0]),
          np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0.5]),np.array([0,0,0.5]),
          np.array([0,0,0.5]),np.array([2/3,1/3,1/3]),
          np.array([2/3,1/3,1/3]),np.array([2/3,1/3,1/3]),np.array([2/3,1/3,0.5+1/3]),
          np.array([2/3,1/3,0.5+1/3]),np.array([2/3,1/3,0.5+1/3]),np.array([1/3,2/3,2/3]),
          np.array([1/3,2/3,2/3]),np.array([1/3,2/3,2/3]),np.array([1/3,2/3,0.5+2/3]),
          np.array([1/3,2/3,0.5+2/3]),np.array([1/3,2/3,0.5+2/3])]],
#"R -3 c":[[np.array([[0,0,1],[1,0,0],[0,1,0]]),np.array([[0,1,0],[0,0,1],[1,0,0]]),
#          np.array([[1,0,0],[0,0,1],[0,1,0]]),np.array([[0,0,1],[0,1,0],[1,0,0]]),
#          np.array([[0,1,0],[1,0,0],[0,0,1]]),np.array([[-1,0,0],[0,-1,0],[0,0,-1]]),
#          np.array([[0,0,-1],[-1,0,0],[0,-1,0]]),np.array([[0,-1,0],[0,0,-1],[-1,0,0]]),
#          np.array([[-1,0,0],[0,0,-1],[0,-1,0]]),np.array([[0,0,-1],[0,-1,0],[-1,0,0]]),
#          np.array([[0,-1,0],[-1,0,0],[0,0,-1]])],
#          [np.array([0,0,0]),np.array([0,0,0]),np.array([0.5,0.5,0.5]),np.array([0.5,0.5,0.5]),
#          np.array([0.5,0.5,0.5]),np.array([0,0,0]),
#          np.array([0,0,0]),np.array([0,0,0]),np.array([0.5,0.5,0.5]),np.array([0.5,0.5,0.5]),
#          np.array([0.5,0.5,0.5])]],
"R 3 c":[[np.array([[0,-1,0],[1,-1,0],[0,0,1]]),np.array([[-1,1,0],[-1,0,0],[0,0,1]]),
          np.array([[0,-1,0],[-1,0,0],[0,0,1]]),np.array([[-1,1,0],[0,1,0],[0,0,1]]),
          np.array([[1,0,0],[1,-1,0],[0,0,1]]),np.array([[1,0,0],[0,1,0],[0,0,1]]),
          np.array([[0,-1,0],[1,-1,0],[0,0,1]]),np.array([[-1,1,0],[-1,0,0],[0,0,1]]),
          np.array([[0,-1,0],[-1,0,0],[0,0,1]]),np.array([[-1,1,0],[0,1,0],[0,0,1]]),
          np.array([[1,0,0],[1,-1,0],[0,0,1]]),np.array([[1,0,0],[0,1,0],[0,0,1]]),
          np.array([[0,-1,0],[1,-1,0],[0,0,1]]),np.array([[-1,1,0],[-1,0,0],[0,0,1]]),
          np.array([[0,-1,0],[-1,0,0],[0,0,1]]),np.array([[-1,1,0],[0,1,0],[0,0,1]]),np.array([[1,0,0],[1,-1,0],[0,0,1]])],
          [np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0.5]),np.array([0,0,0.5]),
          np.array([0,0,0.5]),np.array([2/3,1/3,1/3]),
          np.array([2/3,1/3,1/3]),np.array([2/3,1/3,1/3]),np.array([2/3,1/3,0.5+1/3]),
          np.array([2/3,1/3,0.5+1/3]),np.array([2/3,1/3,0.5+1/3]),np.array([1/3,2/3,2/3]),
          np.array([1/3,2/3,2/3]),np.array([1/3,2/3,2/3]),np.array([1/3,2/3,0.5+2/3]),
          np.array([1/3,2/3,0.5+2/3]),np.array([1/3,2/3,0.5+2/3])]],
#"R 3 c":[[np.array([[0,0,1],[1,0,0],[0,1,0]]),np.array([[0,1,0],[0,0,1],[1,0,0]]),
#          np.array([[1,0,0],[0,0,1],[0,1,0]]),np.array([[0,0,1],[0,1,0],[1,0,0]]),
#          np.array([[0,1,0],[1,0,0],[0,0,1]])],
#          [np.array([0,0,0]),np.array([0,0,0]),np.array([0.5,0.5,0.5]),np.array([0.5,0.5,0.5]),
#          np.array([0.5,0.5,0.5])]],
"P -4 21 m":[[np.array([[-1,0,0],[0,-1,0],[0,0,1]]),np.array([[0,1,0],[-1,0,0],[0,0,-1]]),
            np.array([[0,-1,0],[1,0,0],[0,0,-1]]),np.array([[1,0,0],[0,-1,0],[0,0,-1]]),
            np.array([[-1,0,0],[0,1,0],[0,0,-1]]),np.array([[0,1,0],[1,0,0],[0,0,1]]),
            np.array([[0,-1,0],[-1,0,0],[0,0,1]])],[np.array([0,0,0]),np.array([0,0,0]),
            np.array([0,0,0]),np.array([0.5,0.5,0]),np.array([0.5,0.5,0]),np.array([0.5,0.5,0]),np.array([0.5,0.5,0])]]}


def cifStr2Float(string):
    """
    change cif coordinate e.g. '0.123(45)' to be float 0.123
    """
    if "(" in string:
        return float(string.split('(')[0]+string.split('(')[1][:-1])
    else:
        return float(string)
    
    
def readEquivPos(path):
    """
    generate all transform matrices (rotation and translation) of
    equiv. points according to a given path of a cif file in 2 lists
    """
    r_matrices = []
    r_matrices.append(np.array([[1,  0,  0],[ 0,  1,  0],[ 0,  0,  1]]))
    t_matrices = []
    t_matrices.append(np.array([0, 0, 0]))
    x=0
    y=0
    z=0
    with open(path,'r') as cif:
        line = cif.readline()
        while line:
            if line.strip() == "'x, y, z'" or line.strip() =="'x,y,z'":
                line = cif.readline()
                while not (line == 'loop_\n'):
                    t_list = []
                    r_list = []
                    for component in line.strip()[1:-1].split(","):
                        comp = component.strip()
                        t_list.append(eval(comp))
                        xIndex = comp.find("x")
                        if xIndex==-1:pass
                        elif xIndex==0:
                            r_list.append([1,0,0])
                        elif comp[xIndex-1]=="+":
                            r_list.append([1,0,0])
                        else:
                            r_list.append([-1,0,0])
                        yIndex = comp.find("y")
                        if yIndex==-1:pass
                        elif yIndex==0:
                            r_list.append([0,1,0])
                        elif comp[yIndex-1]=="+":
                            r_list.append([0,1,0])
                        else:
                            r_list.append([0,-1,0])
                        zIndex = comp.find("z")
                        if zIndex==-1:pass
                        elif zIndex==0:
                            r_list.append([0,0,1])
                        elif comp[zIndex-1]=="+":
                            r_list.append([0,0,1])
                        else:
                            r_list.append([0,0,-1])
                    t_matrices.append(np.array(t_list))
                    r_matrices.append(np.array(r_list))
                    line = cif.readline()
                cif.seek(0,2)
            line = cif.readline()
    return (r_matrices,t_matrices)

def frac2cartesian(a,b,c,alpha,beta,gamma,coordinate):
    """
    change fractional coordinate to become cartesian coordinate.
    """
    a_v = a*np.array([1,0,0])
    b_v = b*np.array([np.cos(gamma),np.sin(gamma),0])
    c_v = c*np.array([np.cos(beta),(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma),
    np.sqrt(np.sin(beta)**2-((np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma))**2)])
    return coordinate[0]*a_v+coordinate[1]*b_v+coordinate[2]*c_v

def cif2molecule(path):
    """
    generate all atoms to be a pre-mol in asymm. unit according to a given path of its .cif file
    """
    mol = Molecule()
    atoms = []
    append = False
    for line in open(path,"r"):
        l = line.strip()
        if l.split(" ")[0]=="_space_group_name_H-M_alt":
            mol.spg = l.split("'")[-2]
            continue
        if l.split(" ")[0]=="_symmetry_space_group_name_H-M":
            mol.spg = l.split("'")[-2]
            continue
        if l.split(" ")[0]=="_cell_length_a":
            mol.a = cifStr2Float(l.split(" ")[-1])
            continue
        if l.split(" ")[0]=="_cell_length_b":
            mol.b = cifStr2Float(l.split(" ")[-1])
            continue
        if l.split(" ")[0]=="_cell_length_c":
            mol.c = cifStr2Float(l.split(" ")[-1])
            continue
        if l.split(" ")[0]=="_cell_angle_alpha":
            mol.alpha = cifStr2Float(l.split(" ")[-1])/180*np.pi
            continue
        if l.split(" ")[0]=="_cell_angle_beta":
            mol.beta = cifStr2Float(l.split(" ")[-1])/180*np.pi
            continue
        if l.split(" ")[0]=="_cell_angle_gamma":
            mol.gamma = cifStr2Float(l.split(" ")[-1])/180*np.pi
            continue       
        if l == '_atom_site_label':
            append = True
        if l == "loop_":
            append = False
        if append:
            if len(l)>0:
                if not l[0] == "_":
                    atoms.append(l)
    for atom in atoms:
        info = atom.split(" ")
        mol.addAtom(Atom(name=info[1],coordinate=frac2cartesian(mol.a,mol.b,mol.c,
                         mol.alpha,mol.beta,mol.gamma,np.array([cifStr2Float(info[2]),
                         cifStr2Float(info[3]),cifStr2Float(info[4])]))))
    return mol

def xyz2normalized(path):
    """
    change a non-nornalized .xyz file to be in the following form:
    4 
    name of the molecule
    C        0.000000000      0.000000000      0.000000000
    O        2.362157486      0.000000000      0.000000000
    H       -1.109548835      1.774545300      0.000000000
    H       -1.109548835     -1.774545300      0.000000000
    which only includes 1 molecule
    """
    filename = path.split("/")[-1].split(".")[0]+"_normalized"+".xyz"
    with open(path[:-len(path.split("/")[-1])]+filename,"w") as normalxyz:
        with open(path,"r") as xyz:
            normalxyz.writelines([xyz.readline(),path.split("/")[-1].split(".")[0]+"\n"])
            xyz.readline()
            line = xyz.readline()
            while line:
                non_empty_parts=[]
                if len(line.strip().split(" "))>3:
                    for part in line.strip().split(" "):
                        if not part=="":
                            non_empty_parts.append(part)
                else:
                    for part in line.strip().split("\t"):
                        non_empty_parts.append(part)
                if len(non_empty_parts)>3:
                    for part in non_empty_parts[:4]:
                        normalxyz.write(part)
                        normalxyz.write("        ")
                    normalxyz.write("\n")
                line = xyz.readline()

                            
def xyz2inp(xyzfile,QMprogram,command, description,charge,radius,cycle,chg=0,spin=1):
    """
    produce a corresponding input file from .xyz file for different QM programs
    """
    if QMprogram.lower() == "orca":
        parts = xyzfile.split("/")
        mol_name = [part for part in parts if len(part)>0][-2]
        targetDir = xyzfile[::-1].split("/",1)[1][::-1]+"/"
        if not charge:
            if description=="":
                description = xyzfile.split("/")[-1]
            with open(targetDir+mol_name+"1.inp",'w') as inp:
                ends = "\n"
                inp.writelines(["! "+command,ends,"#",description,ends,
                "%loc",ends,"T_CORE -10000000",ends,"Locmet IAOIBO",ends,"end",
                 ends,"* xyzfile %s %s "%(chg,spin)+xyzfile,ends])
        else:
            if description=="":
                description = xyzfile.split("/")[-1]
            with open(targetDir+mol_name+str(cycle)+".inp",'w') as inp:
                ends = "\n"
                inp.writelines(["! "+command,ends,"#",description,ends,
                "%loc",ends,"T_CORE -10000000",ends,"Locmet IAOIBO",ends,"end",ends,
                "%pointcharges "+"\""+targetDir+mol_name+str(cycle-1)+".chg"+"\"\n",
                "* xyzfile %s %s "%(chg,spin)+xyzfile,ends])
    elif QMprogram.lower()=="gaussian":
        print(QMprogram)
    else:
        print("Not yet developed for this QM program!")



def read_c21database(path):
    """
    from C21 database.txt extract infomation to form 21(22) molecules/fragemets
    in a unit cell and return a list of them
    """
    mols = []
    with open(path,"r") as data:
        line = data.readline().strip()
        atoms = []
        mol = Molecule()
        while line:
            if line[:4]=="atom":
                info = []
                for part in line.split(" ")[1:]:
                    if len(part)>0:
                        info.append(part)
                atoms.append(Atom(typ=info[0],coordinate=np.array([float(info[1]),float(info[2]),float(info[3])])))
            elif line[:4]=="name":
                name = line.split(" ")[-1]
            elif line[:2]=="a ":
                a=float(line.split(" ")[-1])
            elif line[:2]=="b ":
                b=float(line.split(" ")[-1])
            elif line[:2]=="c ":
                c=float(line.split(" ")[-1])
            elif line[:5]=="alpha":
                alpha=float(line.split(" ")[-1])/180*np.pi
            elif line[:4]=="beta":
                beta=float(line.split(" ")[-1])/180*np.pi
            elif line[:5]=="gamma":
                gamma=float(line.split(" ")[-1])/180*np.pi
            elif line[:3]=="spg":
                spg=line[4:]
            elif line == "end": 
                for atom in atoms:
                    atom.setCoordinate(frac2cartesian(a,b,c,alpha,beta,gamma,atom.coordinate))
                    mol.addAtom(atom)
                mol.setCellpara(a,b,c,alpha,beta,gamma)
                mol.setName(name)
                mol.setSpg(spg)
                atoms = []
                mols.append(mol)
                mol = Molecule()
            line = data.readline().strip()
    return mols
def searchSingleMol(unitcell_mol,case="1"):
    """
    according to atoms in unit cell and cell parameters generate the coordinate
    of a single whole molecule most close to origin and return this molecule
    """
    name = unitcell_mol.name
    a = unitcell_mol.a
    b = unitcell_mol.b
    c = unitcell_mol.c
    alpha = unitcell_mol.alpha
    beta = unitcell_mol.beta
    gamma = unitcell_mol.gamma
    spg = unitcell_mol.spg
    before_translate = copy.deepcopy(unitcell_mol)
    translate_v = [np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]
#    try:
#        operation = spgDic[spg]
#    except:
#        try:
#            operation = spgDic[spg+" "+case]
#        except:
#            raise ValueError
    for i in range(3):
        buffer = copy.deepcopy(before_translate)
        before_translate.translate(frac2cartesian(a,b,c,
                alpha,beta,gamma,translate_v[i]))
        after_translate = before_translate
        for atom in after_translate.atomDic:
            buffer.addAtom(atom)
        before_translate = buffer
    buffer = copy.deepcopy(before_translate)
    center = frac2cartesian(a,b,c,alpha,beta,gamma,np.array([1,1,1]))
    min_dis = np.linalg.norm(center)
    min_dis_atom = Atom("H")
    for start_atom in buffer.atomDic:
        if np.linalg.norm(start_atom.coordinate-center)<min_dis:
            min_dis = np.linalg.norm(start_atom.coordinate-center)
            min_dis_atom = start_atom
    mol = Molecule(min_dis_atom)
    mol.setName(name)
    mol.setCellpara(a,b,c,alpha,beta,gamma)
    mol.setSpg(spg)
    det_to_be_added(before_translate,mol,1.63)
    mol.translate(-center)
    atom_number = len(mol.atomDic)
    mols = [mol]
    before_translate = copy.deepcopy(unitcell_mol)
    translate_v = [np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]
    for i in range(3):
        buffer = copy.deepcopy(before_translate)
        before_translate.translate(frac2cartesian(a,b,c,
                alpha,beta,gamma,translate_v[i]))
        after_translate = before_translate
        for atom in after_translate.atomDic:
            buffer.addAtom(atom)
        before_translate = buffer
        
        buffer = copy.deepcopy(before_translate)
        mini_cell = copy.deepcopy(before_translate)
    center = frac2cartesian(a,b,c,alpha,beta,gamma,np.array([1,1,1]))
    for start_atom in buffer.atomDic:
        mol = Molecule(start_atom)
        mol.setName(name)
        mol.setCellpara(a,b,c,alpha,beta,gamma)
        mol.setSpg(spg)
#            det_to_be_added(mini_cell,mol,1.63)
        det_to_be_added(mini_cell,mol,1.63)
        mini_cell = copy.deepcopy(before_translate)
        if len(mol.atomDic)==atom_number:
            mol.translate(-center)
            add_flag = True
            for existed in mols:
                if np.linalg.norm(existed.center()-mol.center())<1.63:
                    add_flag = False
#                    else:
#                        for i in range(len(operation[1])):
#                            if np.linalg.norm(frac2cartesian(a,b,c,alpha,beta,gamma,
#                            operation[0][i].dot(cartesian2frac(unitcell_mol,
#                            existed.center()))+operation[1][i])-mol.center())<1.63:
#                                add_flag = False
            if add_flag:
                mol = orderatoms(mols[0],mol)
                mol.setMatrix(mols[0].getRotationMatrix(mol))
                mols.append(mol)    
    return mols
def searchSingleMol2(unitcell_mol,case="1"):
    """
    according to atoms in unit cell and cell parameters generate the coordinate
    of a single whole molecule most close to origin and return this molecule
    """
    name = unitcell_mol.name
    a = unitcell_mol.a
    b = unitcell_mol.b
    c = unitcell_mol.c
    alpha = unitcell_mol.alpha
    beta = unitcell_mol.beta
    gamma = unitcell_mol.gamma
    spg = unitcell_mol.spg
    before_translate = copy.deepcopy(unitcell_mol)
    translate_v = [np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]
#    try:
#        operation = spgDic[spg]
#    except:
#        try:
#            operation = spgDic[spg+" "+case]
#        except:
#            raise ValueError
    buffer = copy.deepcopy(before_translate)
    min_dis = np.linalg.norm(1000)
    min_dis_atom = Atom("H")
    for start_atom in buffer.atomDic:
        if np.linalg.norm(start_atom.coordinate)<min_dis:
            min_dis = np.linalg.norm(start_atom.coordinate)
            min_dis_atom = start_atom
    mol = Molecule(min_dis_atom)
    mol.setName(name)
    mol.setCellpara(a,b,c,alpha,beta,gamma)
    mol.setSpg(spg)
    det_to_be_added(before_translate,mol,2)
    atom_number = len(mol.atomDic)
    mols = [mol]
    before_translate = copy.deepcopy(unitcell_mol)
    for i in range(3):
        buffer = copy.deepcopy(before_translate)
        before_translate.translate(frac2cartesian(a,b,c,
                alpha,beta,gamma,translate_v[i]))
        after_translate = before_translate
        for atom in after_translate.atomDic:
            buffer.addAtom(atom)
        before_translate = buffer
        
    buffer = copy.deepcopy(before_translate)
    mini_cell = copy.deepcopy(before_translate)
    center = frac2cartesian(a,b,c,alpha,beta,gamma,np.array([0,0,0]))
    for start_atom in buffer.atomDic:
        mol = Molecule(start_atom)
        mol.setName(name)
        mol.setCellpara(a,b,c,alpha,beta,gamma)
        mol.setSpg(spg)
#            det_to_be_added(mini_cell,mol,1.63)
        det_to_be_added(mini_cell,mol,2)
        mini_cell = copy.deepcopy(before_translate)
        if len(mol.atomDic)==atom_number:
            mol.translate(-center)
            add_flag = True
            for existed in mols:
                if np.linalg.norm(existed.center()-mol.center())<1.63:
                    add_flag = False
#                    else:
#                        for i in range(len(operation[1])):
#                            if np.linalg.norm(frac2cartesian(a,b,c,alpha,beta,gamma,
#                            operation[0][i].dot(cartesian2frac(unitcell_mol,
#                            existed.center()))+operation[1][i])-mol.center())<1.63:
#                                add_flag = False
            if add_flag:
                mol = orderatoms(mols[0],mol)
                mol.setMatrix(mols[0].getRotationMatrix(mol))
                mols.append(mol)    
    return mols

def cluster_getxyzFile(cluster,radius,targetDir):
    """
    generate a .xyz file of a cluster (list of molecules) in the following form:
        
    6
    water
    O        2.362157486      0.000000000      0.000000000
    H       -1.109548835      1.774545300      0.000000000
    H       -1.109548835     -1.774545300      0.000000000
    O        2.362157486      0.000000000      0.000000000
    H       -1.109548835      1.774545300      0.000000000
    H       -1.109548835     -1.774545300      0.000000000
    
    """
    if len(cluster)>0:
        filename = cluster[0].name+"_radius"+str(radius)+"_cluster.xyz"
        if targetDir[-1]=="/":
            file = targetDir+filename
        else:
            file = targetDir+"/"+filename
        with open(file,"w") as xyz:
            xyz.write(str((len(cluster)-3)*cluster[0].atomNum())+"\n")
            xyz.write(filename[:-4]+"\n")
            for mol in cluster:
                for atom in mol.atomDic:
                    xyz.write(atom.type+" "+str(atom.coordinate[0])+" "
                    +str(atom.coordinate[1])+" "+str(atom.coordinate[2])+"\n")
    else:
        filename = "radius"+str(radius)+"_cluster.xyz"
        if targetDir[-1]=="/":
            file = targetDir+filename
        else:
            file = targetDir+"/"+filename
        with open(file,"w") as xyz:
            xyz.write(str(0)+"\n")
            xyz.write(filename[:-4]+"\n")
            
def cluster_getchgFile(cluster,cycle,targetDir):
    """
    generate a point charge file of the cluster
    
    """
    if len(cluster)>0:
        filename = cluster[0].name+str(cycle)+".chg"
        file = targetDir+"/"+filename
        with open(file,"w") as xyz:
            xyz.write(str((len(cluster)-1-3)*cluster[0].atomNum())+"\n")
            for mol in cluster[1:-3]:
                for atom in mol.atomDic:
                    xyz.write(str(atom.charge)+" "+str(atom.coordinate[0])+" "
                    +str(atom.coordinate[1])+" "+str(atom.coordinate[2])+"\n")
    else:
        filename = str(cycle)+".chg"
        file = targetDir+"/"+filename
        with open(file,"w") as xyz:
            xyz.write(str(0)+"\n")   
            
def cartesian2frac(mol,coordinate):
    """
    change fractional coordinate to become cartesian coordinate.
    """
    matrix = np.array([[mol.a,mol.b*np.cos(mol.gamma),mol.c*np.cos(mol.beta)],
    [0,mol.b*np.sin(mol.gamma),mol.c*(np.cos(mol.alpha)-np.cos(mol.beta)*np.cos(mol.gamma))/np.sin(mol.gamma)],[0,0,
     mol.c*np.sqrt(np.sin(mol.beta)**2-((np.cos(mol.alpha)-np.cos(mol.beta)*np.cos(mol.gamma))/np.sin(mol.gamma))**2)]])
    return np.linalg.inv(matrix).dot(coordinate)
            
def singleMol2cluster(mols,radius,case="1"):
    """
    generate all whole molecules from a single molecule most close to origin by
    applying all symm. operations in the space group of this crystal
    """
    a = mols[0].a
    b = mols[0].b
    c = mols[0].c
    alpha = mols[0].alpha
    beta = mols[0].beta
    gamma = mols[0].gamma
#    matrix = np.array([[a,b*np.cos(gamma),c*np.cos(beta)],
#    [0,b*np.sin(gamma),c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma)],[0,0,
#    c*np.sqrt(np.sin(beta)**2-((np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma))**2)]])
#    try:
#        operation = spgDic[mols[0].spg]
#    except:
#        try:
#            operation = spgDic[mols[0].spg+" "+case]
#        except:
#            raise ValueError
#    operation[0].insert(0,np.eye(3))
#    operation[1].insert(0,np.array([0,0,0]))
#    minimal_cluster = []
#    for mol in mols:
#        buffer = copy.deepcopy(mol)
#        buffer_original = copy.deepcopy(mol)
#        minimal_cluster.append(mol)
#        if_add = True
#        for i in range(len(operation[1])):
#            for atom in buffer.atomDic:
#                atom.coordinate = frac2cartesian(a,b,c,alpha,beta,gamma,
#                operation[0][i].dot(cartesian2frac(buffer,atom.coordinate))
#                +operation[1][i])
#            buffer.setMatrix(((matrix.dot(operation[0][i])).dot(np.linalg.inv(
#            matrix))).dot(buffer.matrix))
#            for m in minimal_cluster:
#                if np.dot((m.center()-buffer.center()),(m.center()-buffer.center()))<25e-6:
#                    if_add = False
#            if if_add:
#                minimal_cluster.append(buffer)
#            buffer = copy.deepcopy(buffer_original)
    minimal_cluster = copy.deepcopy(mols)
    a_step = 0
    b_step = 0
    c_step = 0
    cluster = []
    flag1 = -1
    flag2 = -1
    flag3 = -1
    while flag3 < len(cluster):
        flag3 = len(cluster)
        while flag2 < len(cluster):
            flag2 = len(cluster)
            while flag1<len(cluster):
                flag1 = len(cluster)
                for a_v in [-a_step,a_step]:
                    for b_v in [-b_step,b_step]:
                        for c_v in [-c_step,c_step]:
                            to_add = []
                            translated = copy.deepcopy(minimal_cluster)
                            cartesian = frac2cartesian(a,b,c,alpha,beta,gamma,np.array([a_v,b_v,c_v]))
                            for mol in translated:
                                mol.translate(cartesian)
                            for mol in translated:
                                if np.linalg.norm(mol.center()-minimal_cluster[0].center())<=radius:
#                                if np.linalg.norm(mol.center())<=radius:
                                    to_add.append(mol)
                                    for exist in cluster:
                                        if np.dot(exist.center()-mol.center(),exist.center()-mol.center())<25e-6:
                                            to_add = to_add[:-1]
                            for to_add_mol in to_add:
                                cluster.append(to_add_mol)
                c_step += 1
            c_step = 0
            b_step += 1
            flag1 = -1
        c_step = 0
        b_step = 0
        a_step += 1
        flag2 = -1
        flag1 = -1
    center = cluster[0].center()
    for mol in cluster:
        mol.translate(-center)
    #add 3 dummy molecules to hold coordinates of a, b, c vectors
    coords_cell = np.array([[a,b*np.cos(gamma),c*np.cos(beta)],
    [0,b*np.sin(gamma),c*(np.cos(alpha)-np.cos(beta)
    *np.cos(gamma))/np.sin(gamma)],[0,0,c*np.sqrt(np.sin(beta)
    **2-((np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma))
    **2)]])
    cluster.append(Molecule(Atom(typ="DA",coordinate=coords_cell[:,0])))
    cluster.append(Molecule(Atom(typ="DA",coordinate=coords_cell[:,1])))
    cluster.append(Molecule(Atom(typ="DA",coordinate=coords_cell[:,2])))
    return cluster


def singleMol2cluster2(mols,radius,case="1"):
    """
    generate all whole molecules from a single molecule most close to origin by
    applying all symm. operations in the space group of this crystal
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

#    cluster_getxyzFile(minimal_cluster,20,"/home/yalun/Desktop")
#    minimal_cluster = copy.deepcopy(mols)
    a_step = 0
    b_step = 0
    c_step = 0
    cluster = []
    flag1 = -1
    flag2 = -1
    flag3 = -1
    while flag3 < len(cluster):
        flag3 = len(cluster)
        while flag2 < len(cluster):
            flag2 = len(cluster)
            while flag1<len(cluster):
                flag1 = len(cluster)
                for a_v in [-a_step,a_step]:
                    for b_v in [-b_step,b_step]:
                        for c_v in [-c_step,c_step]:
                            to_add = []
                            translated = copy.deepcopy(minimal_cluster)
                            cartesian = frac2cartesian(a,b,c,alpha,beta,gamma,np.array([a_v,b_v,c_v]))
                            for mol in translated:
                                mol.translate(cartesian)
                            for mol in translated:
                                if np.linalg.norm(mol.center()-minimal_cluster[0].center())<=radius:
                                    to_add.append(mol)
                                    for exist in cluster:
                                        if np.dot(exist.center()-mol.center(),exist.center()-mol.center())<25e-6:
                                            to_add = to_add[:-1]
                            for to_add_mol in to_add:
                                cluster.append(to_add_mol)
                c_step += 1
            c_step = 0
            b_step += 1
            flag1 = -1
        c_step = 0
        b_step = 0
        a_step += 1
        flag2 = -1
        flag1 = -1
    center = cluster[0].center()
    for mol in cluster:
        mol.translate(-center)
    #add 3 dummy molecules to hold coordinates of a, b, c vectors
    coords_cell = np.array([[a,b*np.cos(gamma),c*np.cos(beta)],
    [0,b*np.sin(gamma),c*(np.cos(alpha)-np.cos(beta)
    *np.cos(gamma))/np.sin(gamma)],[0,0,c*np.sqrt(np.sin(beta)
    **2-((np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma))
    **2)]])
    cluster.append(Molecule(Atom(typ="DA",coordinate=coords_cell[:,0])))
    cluster.append(Molecule(Atom(typ="DA",coordinate=coords_cell[:,1])))
    cluster.append(Molecule(Atom(typ="DA",coordinate=coords_cell[:,2])))
    return cluster        

def det_to_be_added(small_cluster,mol,criterion):
    """
    determine which atoms in "before_translate" are belong to the same single
    molecule with atoms in "mol" together and add them into this molecule
    """
    to_be_added = []
    for atom in small_cluster.atomDic:
        for exist_atom in mol.atomDic:
            if (np.linalg.norm(exist_atom.coordinate-atom.coordinate)
            <criterion) and (np.linalg.norm(exist_atom.coordinate-atom.coordinate)>5e-3):
                if not atom in to_be_added:
                    to_be_added.append(atom)
    if to_be_added:
        for a in to_be_added:
            mol.addAtom(a)
            small_cluster.atomDic.remove(a)
        det_to_be_added(small_cluster,mol,criterion)



        
def out2chg(outfile,cycle):
    charges = []
    FLAG = "MULLIKEN ATOMIC CHARGES"
    parts = outfile.split("/")
    mol_name = [part for part in parts if len(part)>0][-2]
    targetDir = outfile[::-1].split("/",1)[1][::-1]+"/"
    with open(outfile,"r") as out:
        line = out.readline()
        while line:
            if line.strip()==FLAG:
                line = out.readline()
                line = out.readline()
                with open(targetDir+mol_name+str(cycle)+"_save.chg","w") as chg:
                    with open(targetDir+mol_name+"_fromClass.xyz","r") as xyz:
                        xyz_line = xyz.readline()
                        chg.write(xyz_line)
                        xyz_line = xyz.readline()
                        while line.strip().split(" ")[0] != "Sum":
                            xyz_line = xyz.readline()
                            info = line.strip().split(" ")
                            chg.write(info[-1]+"  "+xyz_line.lstrip()[2:].lstrip())
                            charges.append(float(info[-1]))
                            line = out.readline()
                out.seek(0,2)
            line = out.readline()
    return charges

def out2IAOchg(outfile,cycle):
    charges = []
    FLAG = "IAO PARTIAL CHARGES"
    parts = outfile.split("/")
    mol_name = [part for part in parts if len(part)>0][-2]
    targetDir = outfile[::-1].split("/",1)[1][::-1]+"/"
    with open(outfile,"r") as out:
        line = out.readline()
        while line:
            if line.strip()==FLAG:
                line = out.readline()
                line = out.readline()
                line = out.readline()
                with open(targetDir+mol_name+str(cycle)+"_save.chg","w") as chg:
                    with open(targetDir+mol_name+"_fromClass.xyz","r") as xyz:
                        xyz_line = xyz.readline()
                        chg.write(xyz_line)
                        xyz_line = xyz.readline()
                        while line.strip().split(" ")[0] != "Sum":
                            xyz_line = xyz.readline()
                            info = line.strip().split(" ")
                            chg.write(info[-1]+"  "+xyz_line.lstrip()[2:].lstrip())
                            charges.append(float(info[-1]))
                            line = out.readline()
                out.seek(0,2)
            line = out.readline()
    return charges

def updateCluster(path,cluster,radius,cycle):
    """
    rewrite a cluster according to a given path of normalized .xyz file
    of its central molecule
    """
    index = 0
    targetDir = path[::-1].split("/",1)[1][::-1]
    mol_name = cluster[0].name
    charges = out2IAOchg(targetDir+"/"+mol_name+str(cycle)+".out",cycle)
    for mol in cluster[:-3]:
        for atom in mol.atomDic:
            atom.setCharge(charges[index%len(mol.atomDic)])
            index += 1
    cluster_getchgFile(cluster,cycle,targetDir)
    return charges
        


def read_energy(outfile):
    """
    read the "FINAL SINGLE POINT ENERGY" from the .out file
    """
    FLAG = "FINAL SINGLE POINT ENERGY"
    energy = 0
    out = open(outfile,"r")
    line = out.readline()
    while line:
        if len(line.strip()[::-1].split(" ",1))>1:
            if line.strip()[::-1].split(" ",1)[1].strip()[::-1]==FLAG:
                energy = float(line.strip()[::-1].split(" ",1)[0][::-1])
                out.seek(0,2)
        line = out.readline()
    out.close()
    return energy

#def orderatoms(orderedmol,unorderedmol):
#    """
#    mol2 whose atoms in different order to mol1 will be ordered according to 
#    mol1, return this reordered mol
#    """
#    ZERO = 1e-2
#    mol0 = copy.deepcopy(orderedmol)
#    mol1 = copy.deepcopy(unorderedmol)
#    ref0 = mol0.center()
#    index_dic = {}
#    while mol0.atomDic:
#        dsquares0 = []
#        for atom in mol0.atomDic:
#            r = atom.coordinate-ref0
#            dsquares0.append(r.dot(r))
#        countofsameD = []
#        for dsquare in dsquares0:
#            countofsameD.append(dsquares0.count(dsquare))
#        dmin = np.array(countofsameD).min()
#        index=countofsameD.index(dmin)
#        
#        dsquares0 = []
#        for atom0 in mol0.atomDic:
#            r = atom0.coordinate-mol0.atomDic[index].coordinate
#            dsquares0.append(r.dot(r))
#        for atom1 in mol1.atomDic:
#            if atom1.type==mol0.atomDic[index].type:
#                dsquares1 = []
#                for atom in mol1.atomDic:
#                    r = atom1.coordinate-atom.coordinate
#                    dsquares1.append(r.dot(r))
#                if (np.abs(np.array(sorted(dsquares0))-np.array(sorted(dsquares1)))<ZERO).all():
#                    real_index = index
#                    for key in sorted(index_dic.keys()):
#                        if key <=real_index:
#                            real_index += 1
#                    real_index2 = mol1.atomDic.index(atom1)
#                    for value in sorted(index_dic.values()):
#                        if value <= real_index2:
#                            real_index2 += 1
#                    index_dic[real_index] = real_index2
#                    mol1.atomDic.remove(atom1)
#                    break
#        if len(mol0.atomDic)>0:
#            ref0 = mol0.atomDic[index].coordinate
#            del(mol0.atomDic[index])
#    for i in range(len(orderedmol.atomDic)):
#        mol1.atomDic.append(unorderedmol.atomDic[index_dic[i]])
#    return mol1



def orderatoms(orderedmol,unorderedmol):
    """
    mol2 whose atoms in different order to mol1 will be ordered according to 
    mol1, return this reordered mol
    """
    #search 1st ref-atom
    mol0 = copy.deepcopy(orderedmol)
    mol1 = copy.deepcopy(unorderedmol)
    
    ref0 = mol0.atomDic[0]
    dsquares0 = []
    for atom in mol0.atomDic:
        r = atom.coordinate-ref0.coordinate
        dsquares0.append(r.dot(r))

    max_diff = 1e12
    buffer = []
    for ref1 in [a for a in mol1.atomDic if a.type==ref0.type]:
        dsquares1 = []
        for atom in mol1.atomDic:
            r = atom.coordinate-ref1.coordinate
            dsquares1.append(r.dot(r))
        new_diff = np.abs(np.array(sorted(dsquares1))-np.array(sorted(dsquares0))).max()
        if max_diff>new_diff:
            if np.abs(max_diff-new_diff)<1e-2:
                buffer.append(mol1.atomDic.index(ref1))
            else: buffer=[mol1.atomDic.index(ref1)]
            max_diff = new_diff
        elif np.abs(max_diff-new_diff)<1e-2:
            buffer.append(mol1.atomDic.index(ref1))
    cost = [1e12]
    res = []
    for buf in buffer:
        index_dic = [buf]
        recursiveorder(mol0.atomDic,mol1.atomDic,index_dic,cost,res)
#    print(res)
    resmol = copy.deepcopy(mol1)
    for i in range(len(mol1.atomDic)):
        resmol.atomDic[i]=mol1.atomDic[res[i]]
    return resmol
        
def recursiveorder(atomlist0,atomlist1,index_dic,cost,res):
    depth = len(index_dic)
    ref0 = atomlist0[depth]
    dsquares0 = []
    max_diff = 1e12
    buffer = []
    for atom in atomlist0[:depth]:
        r = atom.coordinate-ref0.coordinate
        dsquares0.append(r.dot(r))
    for ref1 in [a for a in atomlist1 if a.type==ref0.type and atomlist1.index(a)
            not in index_dic]:
        dsquares1 = []
        for atom in [atomlist1[index] for index in index_dic]:
            r = atom.coordinate-ref1.coordinate
            dsquares1.append(r.dot(r))
        new_diff = np.abs(np.array(dsquares1)-np.array(dsquares0)).max()
        if max_diff>new_diff:
            if np.abs(max_diff-new_diff)<1e-2:
                buffer.append(atomlist1.index(ref1))
            else: buffer=[atomlist1.index(ref1)]
            max_diff = new_diff
        elif np.abs(max_diff-new_diff)<1e-2:
            buffer.append(atomlist1.index(ref1))      
    for buf in buffer:
        index_dic.append(buf)
        if depth == len(atomlist0)-1:
            testmol1 = Molecule()
            for i in range(len(atomlist1)):
                testmol1.addAtom(atomlist1[index_dic[i]])
            testmol0 = Molecule(*copy.deepcopy(atomlist0))
            testmol0.rotate(testmol0.getRotationMatrix(testmol1))
            testmol0.translate(testmol1.center()-testmol0.center())
            cal_cost = 0
            for i in range(len(testmol1.atomDic)):
                cal_cost += np.linalg.norm(testmol1.atomDic[i].coordinate-
                                           testmol0.atomDic[i].coordinate)
            if cal_cost<cost[0]:
                cost[0] = cal_cost
                res[:] = index_dic
                
        if depth<len(atomlist0)-1:
            recursiveorder(atomlist0,atomlist1,index_dic,cost,res)
        index_dic = index_dic[:depth]

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
#            matrix = cluster[0].getRotationMatrix(mol)
            matrix = mol.matrix
            mol.atomDic[j].translate(matrix.dot(s[j*3:3*(j+1)]))
#            mol.matrix = matrix

























































































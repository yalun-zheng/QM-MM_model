import numpy as np
import copy

def rotationMatrix(alpha,v):
    """
    return a rotation matrix based on an angle alpha and normalized axis vector v
    """
    vx, vy, vz = v[0], v[1], v[2]
    return np.array([[(1-np.cos(alpha))*vx**2+np.cos(alpha),
    vx*vy*(1-np.cos(alpha))-vz*np.sin(alpha),vx*vz*(1-np.cos(alpha))+vy*np.sin(alpha)],
    [vx*vy*(1-np.cos(alpha))+vz*np.sin(alpha),(1-np.cos(alpha))*vy**2+np.cos(alpha),
    vz*vy*(1-np.cos(alpha))-vx*np.sin(alpha)],[vx*vz*(1-np.cos(alpha))-vy*np.sin(alpha),
    vy*vz*(1-np.cos(alpha))+vx*np.sin(alpha),(1-np.cos(alpha))*vz**2+np.cos(alpha)]])

class Molecule:
    """
    A class for molecule composed of some atoms
    """
    def __init__(self, *atoms):
        self.setName("unknown")
        self.atomDic = []
        self.name = "unknown"
        self.dipole = "unset"
        self.matrix = np.identity(3)
        if (not self.atomDic) and atoms:
            atoms[0].setName(atoms[0].type+str(1))
            self.atomDic.append(atoms[0])
        for atom in atoms[1:]:
            if_overlap = False
            for selfAtom in self.atomDic:
                if np.linalg.norm(selfAtom.coordinate-atom.coordinate)<5e-3:
                    if_overlap = True
            if not if_overlap:
                num = 1
                for selfAtom in self.atomDic:
                    if selfAtom.type==atom.type:
                        num += 1
                atom.setName(atom.type+str(num))
                self.atomDic.append(atom)

    def addAtom(self, atom):
        """
        add an atom to this molecule
        """
        if not self.atomDic:
            atom.setName(atom.type+str(1))
            self.atomDic.append(atom)
        else:
            if_overlap = False
            for selfAtom in self.atomDic:
                if np.linalg.norm(selfAtom.coordinate-atom.coordinate)<5e-3:
                    if_overlap = True
            if not if_overlap:
                num = 1
                for selfAtom in self.atomDic:
                    if selfAtom.type==atom.type:
                        num += 1
                atom.setName(atom.type+str(num))
                self.atomDic.append(atom)
    def atomNum(self):
        """
        return how many atoms there are in the molecule
        """
        return len(self.atomDic)
    def getInfo(self):
        """
        return a string of the molecular info: name, atomic type, coordinate x,y,z and charge for each atom
        """
        info = self.name+"\n"
        for atom in self.atomDic:
            info += (atom.getInfo()+"\n")
        return info
    def translate(self, deltaR):
        """
        translate the molecule
        """
        for atom in self.atomDic:
            atom.translate(deltaR)
    def rotate(self, matrix):
        """
        rotate the molecule
        """
        for atom in self.atomDic:
            atom.rotate(matrix)
    def setDipole(self,dipole):
        """
        set/change dipole moment for the molecule
        """
        self.dipole=dipole
    def massCenter(self):
        """
        return the vector of molecule's masscenter
        """
        center = 0.
        m = 0.
        for atom in self.atomDic:
            center += atom.mass*atom.coordinate
            m += atom.mass
        return center/m 
    def center(self):
        """
        return the vector of molecule's geo-center
        """
        center = 0.
        for atom in self.atomDic:
            center += atom.coordinate
        return center/self.atomNum()
    def setCellpara(self,a,b,c,alpha,beta,gamma):
        """
        set cell pareameters for the molecule
        """
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    def setName(self,name):
        """
        set name for the molecule
        """
        self.name = name
    def setMatrix(self,matrix):
        """
        set the rotation matrix for the molecule corresponding to .cif file
        """
        self.matrix = matrix
    def setSpg(self,spg):
        """
        set the space group for the molecule
        """
        self.spg = spg
    def getRotationMatrix(self, mol1):
        """
        return a matrix which turns this molecule to be another, the same kind of molecule
        Note: self and  the para. molecule must be the same molecule or chiral isomers
        """
        mol0 = copy.deepcopy(self)
        ZERO = 1e-10
        matrix = np.identity(3)
        rotatedAxis = []
        for i in range(len(mol0.atomDic)-1):
            origin1 = mol0.atomDic[0].coordinate
            origin2 = mol1.atomDic[0].coordinate
            x1 = mol0.atomDic[i+1].coordinate-origin1
            x2 = mol1.atomDic[i+1].coordinate-origin2
            for axis in rotatedAxis:
                x1 = x1-x1.dot(axis)*axis
                x2 = x2-x2.dot(axis)*axis
            x1_model = np.linalg.norm(x1)
            x2_model = np.linalg.norm(x2)
            if x1_model<ZERO : continue
            else:
                if len(rotatedAxis)<2:
                    rotatedAxis.append(x2/x2_model)
                    v = np.cross(x1,x2)
                    v_model = np.linalg.norm(v)
                    if v_model <ZERO:
                        if len(rotatedAxis)==1:
                            if np.linalg.norm(x1/x1_model-x2/x2_model)<ZERO:
                                pass
                            else:
                                matrix = -matrix
                                mol0.rotate(-1*np.eye(3))
                        elif len(rotatedAxis)==2:
                            if np.linalg.norm(x1/x1_model-x2/x2_model) <ZERO:
                                pass
                            else:
                                matrix = rotationMatrix(np.pi,rotatedAxis[0]).dot(matrix)
                                mol0.rotate(rotationMatrix(np.pi,rotatedAxis[0]))
                    else:
                        v = v/v_model
                        alpha = np.arccos(x1.dot(x2)/np.linalg.norm(x1)/np.linalg.norm(x2))
                        matrix = rotationMatrix(alpha,v).dot(matrix)
                        mol0.rotate(matrix)
                else:
                    if np.linalg.norm(x1/x1_model-x2/x2_model) <ZERO:
                        pass
                    else:
                        normal = np.cross(rotatedAxis[0],rotatedAxis[1])
                        normal = normal/np.linalg.norm(normal)
                        ax = np.cross(normal,np.array([0,0,1]))
                        if np.linalg.norm(ax) <ZERO:
                            matrix = np.array([[1,0,0],
                                [0,1,0],[0,0,-1]]).dot(matrix)
                        else:
                            ax = ax/np.linalg.norm(ax)
                            alpha = np.arccos(normal.dot(np.array([0,0,1])))
                            transform = rotationMatrix(alpha,ax)
                            matrix = np.linalg.inv(transform).dot(np.array([[1,0,0],
                                    [0,1,0],[0,0,-1]]).dot(transform.dot(matrix)))
                    break
        matrix[(np.abs(matrix)<ZERO)] = 0
        return matrix

    def getxyzFile(self,targetDic):
        """
        generate a .xyz file in the following form:
            
        4 
        water
        O        2.362157486      0.000000000      0.000000000
        H       -1.109548835      1.774545300      0.000000000
        H       -1.109548835     -1.774545300      0.000000000
        
        """
        filename = self.name+"_fromClass.xyz"
        if targetDic[-1]=="/":
            file = targetDic+filename
        else:
            file = targetDic+"/"+filename
        with open(file,"w") as xyz:
            xyz.write(str(self.atomNum())+"\n")
            xyz.write(self.name+"\n")
            for atom in self.atomDic:
                xyz.write(atom.type+"    "+str(atom.coordinate[0])+"    "
                +str(atom.coordinate[1])+"    "+str(atom.coordinate[2])+"\n")
































import numpy as np
class Atom:
    """
    A class for atoms with a certain name (or type) and a coordinate vector
    """
    def __init__(self, typ="unknown", coordinate=np.array([0.,0.,0.]),charge=0.,name="unknown"):
        self.name = name
        self.type = typ
        self.coordinate = coordinate
        if name!="unknown" and typ=="unknown":
            try:
                int(self.name[:2][-1])
                self.type = self.name[:1]
            except:
                self.type = self.name[:2]
        self.charge = charge
    def setName(self, name):
        """
        set/change the atom tag
        """
        self.name = name
    def setCoordinate(self, coordinate):
        """
        set/change the atom coordinates
        """
        self.coordinate = coordinate
    def setMass(self,mass):
        """
        set/change the atom mass
        """
        self.mass = mass
    def setCharge(self, charge):
        """
        set/change the atom charge
        """
        self.charge = charge
    def setType(self, kind):
        """
        set/change the atom type
        """
        self.type = kind
    def getInfo(self):
        """
        return a string of the atom info
        """
        return '%5s %2s %10.6f %10.6f %10.6f %10.4f' % (self.name, self.type, self.coordinate[0], 
        self.coordinate[1], self.coordinate[2], self.charge)
    def getVector(self, atom):
        """
        return a vector from this to another atom.
        """
        return atom.coordinate-self.coordinate
    def getDistance(self, atom):
        """
        return the distance between this and another atom.
        """
        return np.sqrt(np.dot(self.coordinate-atom.coordinate,(self.coordinate-atom.coordinate)))
    def translate(self, deltaR):
        """
        translate the atom to R+deltaR
        """
        self.coordinate += deltaR
    def rotate(self, matrix):
        """
        rotate the atom with matrix
        """
        self.coordinate = np.dot(matrix,self.coordinate)
    def findLJParas(self, atom):
        type1 = self.type
        type2 = atom.type
        e = 2
        r0 = 1
        return e, r0
    def getLJPotential(self,atom):
        dist = self.getDistance(atom)
        e,r0 = self.findLJParas(atom)
        return e*((r0/dist)**12-2*(r0/dist)**6)





































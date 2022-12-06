import numpy as np
import matplotlib.pyplot as plt

#define a loss function
def loss(x):
    return (x/2)**-12-2*(x/2)**-6

def gradient(x):
    return -12*(x/2)**-13*0.5+12*(x/2)**-7*0.5

#RFO
def RFOstep(curr_hess = 1, curr_grad = 0, lr=0):
    #First build augmented Hessian matrix
    lastrow = np.append(curr_grad, 0.0)
    aughess = np.vstack((np.column_stack((curr_hess, curr_grad)), lastrow))
    eigenvals, eigenvecs = np.linalg.eigh(aughess)
    # scale eigenvector
    lowestev = np.argmin(eigenvals)
    if eigenvals[lowestev] > 1.0E-4:
        raise ValueError('ERROR: I don\'t want to go up the PES!')
    else:
        stepvec = eigenvecs[:, lowestev]
        if stepvec[-1] == 0.0:
            scaledstepvec = stepvec[:-1]
        else:
            scaledstepvec = stepvec[:-1]/stepvec[-1]
    if np.linalg.norm(scaledstepvec)>lr:
        scaledstepvec = scaledstepvec/np.linalg.norm(scaledstepvec)*lr
    return scaledstepvec[0]


if __name__=="__main__":
    import sys
    #BFGS
    lr = 0.03 if len(sys.argv)==1 else float(sys.argv[1])
    x = 1.8
    x_rfo = x
    g = gradient(x)
    g_rfo = gradient(x_rfo)
    B=1
    B_rfo = 1
    i = 0


    x_ = np.linspace(1.8,2.8,200)
    plt.plot(x_, loss(x_),'b')
    plt.scatter(x, loss(x), label="BFGS")
    plt.scatter(x_rfo, loss(x_rfo), label="RFO+BFGS")
    plt.xticks([],[])
    plt.yticks([],[])
    plt.legend()
    plt.show()

    plt.ion()
    while abs(g)>1e-4 and abs(g_rfo)>1e-4:
        plt.cla()
        plt.plot(x_, loss(x_),'b')
        s = -lr*g/B
        s_rfo = RFOstep(B_rfo,g_rfo,lr=lr)
        print("x_BFGS %.4f\tstep_BFGS %7.3f\tgrad_BFGS %7.3f\tB_BFGS %7.3f"%(x,s,g,B))
        print("x_rfo  %.4f\tstep_rfo  %7.3f\tgrad_rfo  %7.3f\tB_rfo  %7.3f"%(x_rfo,s_rfo,g_rfo,B_rfo))
        print()
        plt.scatter(x, loss(x), label="BFGS")
        plt.scatter(x_rfo, loss(x_rfo), label="RFO+BFGS")
        plt.legend()
        plt.xticks([],[])
        plt.yticks([],[])
        plt.text(2.55,-0.9,"Iter %d"%i)
        i += 1
        x += s
        x_rfo += s_rfo
        y = gradient(x) - g
        y_rfo = gradient(x_rfo)-g_rfo
        g = gradient(x)
        g_rfo = gradient(x_rfo)
        B += (y/s-B)
        B_rfo += (y_rfo/s_rfo-B_rfo)
        plt.pause(1.5)
    plt.ioff()
    plt.show()

import numpy as np

def preprocessIris(infile,outfile):
    stext1 = "Iris-setosa"
    stext2 = "Iris-versicolor"
    stext3 = "Iris-virginica"
    rtext1 = "0"
    rtext2 = "1"
    rtext3 = "2"
    fid = open(infile,"r") 
    oid = open(outfile,"w")
    for s in fid:
        if s.find(stext1)>-1:
            oid.write(s.replace(stext1, rtext1))
        elif s.find(stext2)>-1:
            oid.write(s.replace(stext2, rtext2))
        elif s.find(stext3)>-1:
            oid.write(s.replace(stext3, rtext3))
    fid.close()
    oid.close()

#preprocessIris("iris.data","iris_1.data")
iris = np.loadtxt("iris_1.data", delimiter=",")
iris[:,:4] = iris[:,:4]-iris[:,:4].mean(axis=0)
imax = np.concatenate((iris.max(axis=0)*np.ones((1,5)),np.abs(iris.min( axis=0))*np.ones((1,5))),axis=0).max(axis=0)
print(imax)
iris[:,:4] = iris[:,:4]/imax[:4]

target = np.zeros((np.shape(iris)[0],3))
print(np.shape(iris)[0])
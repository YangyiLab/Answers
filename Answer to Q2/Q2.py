import numpy as np
import math as m
import matplotlib.pyplot as plt

earray=[]
sarray=[]
esarray=[]
parray=[]
tarray=[]

### e denotes E
### s denotes S
### es denotes ES
### p denotes P
### k1,k2,k3 = 100,600,150 


## ds/dt = g(e,s,es,p,k1,k2,k3)
def g(e,s,es,p,k1,k2,k3):
    return k2*es-k1*s*e

def f(e,s,es,p,k1,k2,k3):
    return (k2+k3)*es-k1*s*e

def h(e,s,es,p,k1,k2,k3):
    return k1*s*e-(k2+k3)*es

def o(e,s,es,p,k1,k2,k3):
    return k3*es


def RK4():
    e = 1
    s = 10
    es = 0
    p = 0
    h_ = 0.001
    k1,k2,k3 = 100,600,150
    t = 0
    while t<=3:
        earray.append(e)
        sarray.append(s)
        esarray.append(es)
        parray.append(p)
        tarray.append(t)
        t+=h_

        f1=f(e,s,es,p,k1,k2,k3) #第一步
        m1=e+f1*h_/2
        g1=g(e,s,es,p,k1,k2,k3)
        n1=s+g1*h_/2
        h1=h(e,s,es,p,k1,k2,k3)
        p1=es+h1*h_/2
        o1=o(e,s,es,p,k1,k2,k3)
        q1=p+o1*h_/2

        f2=f(m1,n1,p1,q1,k1,k2,k3) #第二步
        m2=e+f2*h_/2
        g2=g(m1,n1,p1,q1,k1,k2,k3)
        n2=s+g2*h_/2
        h2=h(m1,n1,p1,q1,k1,k2,k3)
        p2=es+h2*h_/2
        o2=o(m1,n1,p1,q1,k1,k2,k3)
        q2=p+o2*h_/2

        f3=f(m2,n2,p2,q2,k1,k2,k3) #第三步
        m3=e+f3*h_
        g3=g(m2,n2,p2,q2,k1,k2,k3)
        n3=s+g3*h_
        h3=h(m2,n2,p2,q2,k1,k2,k3)
        p3=es+h3*h_
        o3=o(m2,n2,p2,q2,k1,k2,k3)
        q3=p+o3*h_

        f4=f(m3,n3,p3,q3,k1,k2,k3) #第四步
        g4=g(m3,n3,p3,q3,k1,k2,k3)
        h4=h(m3,n3,p3,q3,k1,k2,k3)
        o4=o(m3,n3,p3,q3,k1,k2,k3)

        e=e+(f1+2*f2+2*f3+f4)*h_/6
        s=s+(g1+2*g2+2*g3+g4)*h_/6
        es=es+(h1+2*h2+2*h3+h4)*h_/6
        p=p+(o1+2*o2+2*o3+o4)*h_/6
    return 

def main():
    RK4()
    for i in sarray:
        print(i)
# thr results are the earray[E] sarray[S] esarray[ES] parray[P]

if __name__ == "__main__":
    main()
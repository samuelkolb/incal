
from pysmt.shortcuts import *
from problem import Domain, Problem
import string
import math
from smt_print import pretty_print

def xy_domain():
    variables = ["x", "y"]
    var_types = {"x": REAL, "y": REAL}
    var_domains = {"x": (0, 1), "y": (0, 1)}
    return Domain(variables, var_types, var_domains)

def xyz_domain():
    variables = ["x", "y","z"]
    var_types = {"x": REAL, "y": REAL,"z":REAL}
    var_domains = {"x": (0, 1), "y": (0, 1),"z":(0,1)}
    return Domain(variables, var_types, var_domains)


def simple_2DProblem():
    domain=xy_domain()
    x, y = (domain.get_symbol(v) for v in ["x", "y"])
    l=[]

    l.append(0.3*x+2*y<= 0.75)
    l.append(2*y + 0.5* x <= 2)
    theory = And(i for i in l)

    return Problem(domain, theory, "simple_2DProblem")

def hexagon():
    domain=xy_domain()
    x, y = (domain.get_symbol(v) for v in ["x", "y"])

    c1= y<=0.8
    c2= y>=0.2
    c3= 1.5*x-y>=-0.35
    c4= -0.65 >= -1.5 * x - y
    c5=1.5*x-y<=0.85
    c6= -1.5*x-y>=-1.85

    return Problem(domain, And(c1,c2,c3,c4,c5,c6), "hexagon")

def simple_3DProblem():
    domain=xyz_domain()
    x, y ,z= (domain.get_symbol(v) for v in ["x", "y","z"])

    c1 = x <= 0.3
    c2= x+ 0.5*y+ z<=1

    return Problem(domain, And(c1,c2), "simple_3DProblem")

def cuben(n):

    def normilisation(i,x):
        return (x-(i-2.7))/((i+2*2.7)-(i-2.7))

    variables=[]
    var_types={}
    var_domains={}
    l=[]
    letters = list(string.ascii_lowercase)

    for i in letters[:n]:
        variables.append(i)
        var_types[i]= REAL
        var_domains[i]=(0,1)

    domain= Domain(variables, var_types, var_domains)
    counter=2
    a=domain.get_symbol("a")
    l.append(a>=normilisation(1,1))
    l.append(a<=normilisation(1,(1+2.7)))

    for i in letters[1:n]:
        r= domain.get_symbol(i)
        l.append(r>=normilisation(counter,counter))
        l.append(r<normilisation(counter,(counter+2.7)))
        counter=counter+1

    theory=And(i for i in l)
    pretty_print(theory)
    return Problem(domain,theory,"cuben")


def simplexn(dimension):
    count=0
    l=[]
    def normilisation(x):
        return (x+1)/((2+2.7)-1)
    variables=[]
    var_types={}
    var_domains={}

    letters = list(string.ascii_lowercase)

    for i in letters[:dimension]:
        variables.append(i)
        var_types[i]= REAL
        var_domains[i]=(0,1)

    domain= Domain(variables, var_types, var_domains)
    s=[Symbol(s, REAL) for s in domain.variables]
    l.append(Plus(s)<=normilisation(2.7))

    for a in letters[:dimension]:
        count+=1
        x=domain.get_symbol(a)
        for b in letters[count:dimension]:
            y=domain.get_symbol(b)

            l.append(x*normilisation(1/math.tan(math.pi/12))-y*normilisation(math.tan(math.pi/12))>=0)
            l.append(y*normilisation((1/math.tan(math.pi/12)))-x*((math.tan(math.pi/12)))>=0)

    theory = And(i for i in l)
    return Problem(domain,theory,"simplexn")

def blending():

    def normalization(x):
        return x/200

    variables=["x1r","x2r","x3r","x1p","x2p","x3p"]
    var_types={"x1r":REAL,"x2r":REAL,"x3r":REAL,"x1p":REAL,"x2p":REAL,"x3p":REAL}
    var_domains={"x1r":(0,1),"x2r":(0,1),"x3r":(0,1),"x1p":(0,1),"x2p":(0,1),"x3p":(0,1)}
    domain= Domain(variables, var_types, var_domains)
    x1r,x2r,x3r,x1p,x2p,x3p=(domain.get_symbol(i)for i in ["x1r","x2r","x3r","x1p","x2p","x3p"])
    l=[]

    l.append(x1r+x1p<=normalization(50))
    l.append(x2r+x2p<=normalization(100))
    l.append(x3r+x3p<=normalization(100))
    l.append(0.7*x1r-0.3*x2r-0.3*x3r<=0)
    l.append(-0.4*x1r+0.6*x2r-0.4*x3r>=0)
    l.append(-0.2 * x1r - 0.2 * x2r +0.8 * x3r <= 0)
    l.append(0.75 * x1p - 0.25 * x2p -0.25 * x3p >= 0)
    l.append(-0.4 * x1p + 0.6 * x2p - 0.4 * x3p <= 0)
    l.append(-0.3 * x1p -0.3 * x2p + 0.7 * x3p <= 0)
    l.append(x1r+x2r+x3r>=normalization(100))

    theory = And(i for i in l)
    return Problem(domain,theory,"blending")


def police():

    variables=["x1","x2","x3","x4","x5"]
    var_types={"x1":REAL,"x2":REAL,"x3":REAL,"x4":REAL,"x5":REAL}
    var_domains={"x1":(0,1),"x2":(0,1),"x3":(0,1),"x4":(0,1),"x5":(0,1)}
    domain= Domain(variables, var_types, var_domains)
    x1,x2,x3,x4,x5=(domain.get_symbol(i)for i in ["x1","x2","x3","x4","x5"])
    l=[]

    l.append(x1>=48/200)
    l.append(x1+x2>=79/200)
    l.append(x1+x2>=65/200)
    l.append(x1+x2+x3>=87/200)
    l.append(x2+x3>=64/200)
    l.append(x3+x4>=73/200)
    l.append(x3+x4>=82/200)
    l.append(x4>=43/200)
    l.append(x4+x5>=52/200)
    l.append(x5>=15/200)

    theory = And(i for i in l)
    return Problem(domain,theory,"police")


def polutionreduction():

    variables=["x1","x2","x3","x4","x5","x6"]
    var_types={"x1":REAL,"x2":REAL,"x3":REAL,"x4":REAL,"x5":REAL,"x6":REAL}
    var_domains={"x1":(0,1),"x2":(0,1),"x3":(0,1),"x4":(0,1),"x5":(0,1),"x6":(0,1)}
    domain= Domain(variables, var_types, var_domains)
    x1,x2,x3,x4,x5,x6=(domain.get_symbol(i)for i in ["x1","x2","x3","x4","x5","x6"])
    l=[]

    l.append(12*x1+9*x2+25*x3+20*x4+17*x5+13*x6>=60)
    l.append(35 * x1 + 42 * x2 + 18 * x3 + 31 * x4 + 56 * x5 + 49 * x6 >= 150)
    l.append(37 * x1 + 53 * x2 + 28 * x3 + 24 * x4 + 29 * x5 + 20 * x6 >= 125)

    theory = And(i for i in l)
    return Problem(domain,theory,"polution")
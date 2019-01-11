

from pysmt.shortcuts import *
from problem import Domain, Problem
import csv
from parse import smt_to_nested,nested_to_smt
import pywmi
from smt_print import pretty_print

def classs1():
    variables=[ 'Slag', 'Flyash', 'Water', 'SP', 'CoarseAggr.', 'FineAggr.', 'Cement', 'SlagSlag', 'SlagFlyash', 'SlagWater', 'SlagSP', 'SlagCoarseAggr.', 'SlagFineAggr.', 'SlagCement', 'FlyashFlyash', 'FlyashWater', 'FlyashSP', 'FlyashCoarseAggr.', 'FlyashFineAggr.', 'FlyashCement', 'WaterWater', 'WaterSP', 'WaterCoarseAggr.', 'WaterFineAggr.', 'WaterCement', 'SPSP', 'SPCoarseAggr.', 'SPFineAggr.', 'SPCement', 'CoarseAggr.CoarseAggr.', 'CoarseAggr.FineAggr.', 'CoarseAggr.Cement', 'FineAggr.FineAggr.', 'FineAggr.Cement', 'CementCement']
    var_types={}
    var_domains={}

    for i in variables:
        var_types[i] = REAL
        var_domains[i] = (-5329229*2,5329229*2) #(-1, 1102291*2)

    domain = Domain(variables, var_types, var_domains)

    #flyashcoarseaggr,coarseaggrcement,slag ,watersp,flyashsp= (domain.get_symbol(v) for v in ['FlyashCoarseAggr.','CoarseAggr.Cement',"Slag",'WaterSP','FlyashSP' ])
    #s1=0.0073*flyashcoarseaggr-0.002893*coarseaggrcement-3.513*slag+watersp-1.044*flyashsp<=0

    variable_values={'FlyashCoarseAggr.':0.0073, 'CoarseAggr.Cement':-0.002893 , "Slag":-3.513, 'WaterSP':1, 'FlyashSP':-1.004}

    getvaraibles = [Symbol(j, REAL) for j in variables]

    inequality=GE(Real(0), Plus(variable_values[str(x)] * x for x in getvaraibles if str(x) in variable_values))


    return Problem(domain, And(inequality), "class_s1")


def classs3():
    variables = ['Slag', 'Flyash', 'Water', 'SP', 'CoarseAggr.', 'FineAggr.', 'Cement', 'SlagSlag', 'SlagFlyash',
                 'SlagWater', 'SlagSP', 'SlagCoarseAggr.', 'SlagFineAggr.', 'SlagCement', 'FlyashFlyash', 'FlyashWater',
                 'FlyashSP', 'FlyashCoarseAggr.', 'FlyashFineAggr.', 'FlyashCement', 'WaterWater', 'WaterSP',
                 'WaterCoarseAggr.', 'WaterFineAggr.', 'WaterCement', 'SPSP', 'SPCoarseAggr.', 'SPFineAggr.',
                 'SPCement', 'CoarseAggr.CoarseAggr.', 'CoarseAggr.FineAggr.', 'CoarseAggr.Cement',
                 'FineAggr.FineAggr.', 'FineAggr.Cement', 'CementCement']
    var_types = {}
    var_domains = {}

    for i in variables:
        var_types[i] = REAL
        var_domains[i] = (1000, 1000)

    domain = Domain(variables, var_types, var_domains)

    variable_values_one = {'Water':37.68, 'WaterCoarseAggr.':-0.02183, 'WaterFineAggr.':-0.02857, 'FlyashCement':-0.02449, 'CoarseAggr.FineAggr.':0.003091,
         'CoarseAggr.Cement':-0.003236,"FlyashFlyash":0.005181,"SlagFlyash":0.01029,"Flyash":-4.589,"SP":-7.647}

    variable_values_two={"CoarseAggr.":0.337, 'CoarseAggr.FineAggr.':-0.000835, "Water":Real(1)}

    getvaraibles = [Symbol(j, REAL) for j in variables]

    inequality_one = LE(Real(1), Plus(variable_values_one[str(x)] * x for x in getvaraibles if str(x) in variable_values_one))
    inequality_two = GE(Real(0), Plus(variable_values_two[str(x)] * x for x in getvaraibles if str(x) in variable_values_two))

    return Problem(domain, And(inequality_one, inequality_two), "class_s3")


def classs4():
    variables = ['Slag', 'Flyash', 'Water', 'SP', 'CoarseAggr.', 'FineAggr.', 'Cement', 'SlagSlag', 'SlagFlyash',
                 'SlagWater', 'SlagSP', 'SlagCoarseAggr.', 'SlagFineAggr.', 'SlagCement', 'FlyashFlyash', 'FlyashWater',
                 'FlyashSP', 'FlyashCoarseAggr.', 'FlyashFineAggr.', 'FlyashCement', 'WaterWater', 'WaterSP',
                 'WaterCoarseAggr.', 'WaterFineAggr.', 'WaterCement', 'SPSP', 'SPCoarseAggr.', 'SPFineAggr.',
                 'SPCement', 'CoarseAggr.CoarseAggr.', 'CoarseAggr.FineAggr.', 'CoarseAggr.Cement',
                 'FineAggr.FineAggr.', 'FineAggr.Cement', 'CementCement']
    var_types = {}
    var_domains = {}

    for i in variables:
        var_types[i] = REAL
        var_domains[i] = (1000, 1000)

    domain = Domain(variables, var_types, var_domains)

    variable_values_one = {"FlyAsh":18.69, "FlyAshFlyAsh":-0.01979, "Slag":-30.17, "WaterWater":-0.03053, "SlagFineAggr.":0.03787, "WaterFineAggr":0.01417, "FlyashFineAggr":-0.01539,
          "SlagCoarseAggr":0.005304,"CoarseAggr":0.7613,"Cement":-0.6043}

    variable_values_two={"WaterWater":0.008107, "Cement":-0.6571, "SP":1, "Slag":1}
    variable_values_three={"Cement":1, "CementCement":-0.002028, "Slag":0.04184, "Water":0.554}
    variable_values_four={"Cement":1}

    getvaraibles = [Symbol(j, REAL) for j in variables]

    inequality_one = GE(Real(679.5), Plus(variable_values_one[str(x)] * x for x in getvaraibles if str(x) in variable_values_one))
    inequality_two = LE(Real(1), Plus(variable_values_two[str(x)] * x for x in getvaraibles if str(x) in variable_values_two))
    inequality_three = GE(Real(235.1), Plus(variable_values_three[str(x)] * x for x in getvaraibles if str(x) in variable_values_three))
    inequality_four = GE(Real(354), Plus(variable_values_four[str(x)] * x for x in getvaraibles if str(x) in variable_values_four))

    return Problem(domain, And(inequality_one, inequality_two, inequality_three, inequality_four), "class_s4")

def classs5():
    variables = ['Slag', 'Flyash', 'Water', 'SP', 'CoarseAggr.', 'FineAggr.', 'Cement', 'SlagSlag', 'SlagFlyash',
                 'SlagWater', 'SlagSP', 'SlagCoarseAggr.', 'SlagFineAggr.', 'SlagCement', 'FlyashFlyash', 'FlyashWater',
                 'FlyashSP', 'FlyashCoarseAggr.', 'FlyashFineAggr.', 'FlyashCement', 'WaterWater', 'WaterSP',
                 'WaterCoarseAggr.', 'WaterFineAggr.', 'WaterCement', 'SPSP', 'SPCoarseAggr.', 'SPFineAggr.',
                 'SPCement', 'CoarseAggr.CoarseAggr.', 'CoarseAggr.FineAggr.', 'CoarseAggr.Cement',
                 'FineAggr.FineAggr.', 'FineAggr.Cement', 'CementCement']
    var_types = {}
    var_domains = {}

    for i in variables:
        var_types[i] = REAL
        var_domains[i] = (1000, 1000)

    domain = Domain(variables, var_types, var_domains)

    variable_values_one = {"SlagCement":0.06552, "SlagSlag":0.2676, "SlagSP":1, "SlagFineAggr.":-0.08421, "FlyashSP":1, "FlyashFlyash":-7.254,
          "Water":6.722,"WaterSP":-0.9654}
    variable_values_two={"Slag":34.12, "SlagSP":-7.593, "SPFineAggr.":0.002812, "Cement":-0.06889}
    variable_values_three={"CementCement":0.0009766, "Flyash":-0.2784, "CoarseAggr":1, "FineAggr":-0.5486}
    variable_values_four={"Slag":1, "SP":1, "CoarseAggr":0.2611, "FineAggr.":-0.5486}
    variable_values_five={"Water":3.301, "SP":13.8, "Cement":-0.3316, "Slag":1}
    variable_values_six={"Water":1}

    getvaraibles = [Symbol(j, REAL) for j in variables]

    inequality_one = GE(Real(0), Plus(variable_values_one[str(x)] * x for x in getvaraibles if str(x) in variable_values_one))
    inequality_two = GE(Real(0), Plus(variable_values_two[str(x)] * x for x in getvaraibles if str(x) in variable_values_two))
    inequality_three = GE(Real(0), Plus(variable_values_three[str(x)] * x for x in getvaraibles if str(x) in variable_values_three))
    inequality_four = GE(Real(1), Plus(variable_values_four[str(x)] * x for x in getvaraibles if str(x) in variable_values_four))
    inequality_five = LE(Real(550), Plus(variable_values_five[str(x)] * x for x in getvaraibles if str(x) in variable_values_five))
    inequality_six = LE(Real(171.3), Plus(variable_values_six[str(x)] * x for x in getvaraibles if str(x) in variable_values_six))

    return Problem(domain, And(inequality_one,inequality_two,inequality_three,inequality_four,inequality_five,inequality_six), "class_s5")


learned_theory=[]
with open("/Users/Elias/Documents/GitHub/smtlearn/output/slump/theories_learned/class:1.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        learned_theory.append(nested_to_smt(row[0]))

paperclas=classs1()
sample_count=1000000
for i in learned_theory:
    print(pretty_print(paperclas.theory))
    print(pretty_print(i))
    tpr = pywmi.RejectionEngine(paperclas.domain, paperclas.theory, Real(1.0),
                                        sample_count).compute_probability(i)
    #tnr = pywmi.RejectionEngine(paperclas.domain, ~paperclas.theory, Real(1.0),
     #                                   sample_count).compute_probability(~i)
    print(tpr)

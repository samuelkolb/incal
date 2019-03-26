
def qualifiyingwine(lenght):
    data, head = importwine()
    labeled = partioning(data, 5, 5)
    random.shuffle(labeled)
    # lenght=round(0.8*(len(labeled)))

    train = labeled[:lenght]
    test = labeled[lenght:lenght + 200]

    variables = []
    var_types = {}
    var_domains = {}
    for i in head:
        variables.append(i[1:-1])
        var_types[i[1:-1]] = REAL
        var_domains[i[1:-1]] = (0, 1)
    domain = Domain(variables, var_types, var_domains)
    problem = Problem(domain, 0, "wine")
    theory, km, numberofconstrains = learn_parameter_free(problem, train, 3)
    a = accuracy(theorylearned=theory, data1=test)
    print(a)
    return theory, a


def import_slump():
    with open('/Users/Elias/Desktop/slump_test.data.txt', newline='') as csvfile:
        slumpdata = csv.reader(csvfile, delimiter=',', quotechar='|')
        linecount = 0
        dataoutput = []
        for row in slumpdata:
            if linecount > 0:
                row = [float(x) for x in row]
                row[8], row[1] = row[1], row[8]
                dataoutput.append(row[1:9])
            else:
                row[8], row[1] = row[1], row[8]
                header = row[1:9]
            linecount += 1

    return dataoutput, header



def definingslumpclass(data):
    def normilisation(x, min, max, p):
        return (x - min[p]) / (max[p] - min[p])

    # minn=list(map(min, zip(*data)))
    # maxx=list(map(max, zip(*data)))

    # for i in data:             #aktivate normlisation again
    #   for j in range(1,len(i)):
    #      i[j]=normilisation(i[j],minn,maxx,j)

    for i in data:
        lenght = len(i)
        for j in range(1, lenght):
            for k in range(j, lenght):
                i.append(i[j] * i[k])

    for i in data[:]:
        if (i[0] * 10) >= 10 and (i[0] * 10) <= 40:
            i[0] = 1
        elif (i[0] * 10) >= 50 and (i[0] * 10) <= 90:
            i[0] = 2
        elif (i[0] * 10) >= 100 and (i[0] * 10) <= 150:
            i[0] = 3
        elif (i[0] * 10) >= 160 and (i[0] * 10) <= 210:
            i[0] = 4
        elif (i[0] * 10) >= 220:
            i[0] = 5
        else:
            i[0] = 0

    return data


def creatingvaraiblenames(head):
    head = [h.replace(" ", "") for h in head]
    new = head[:]

    for i in range(1, len(head)):
        for j in range(i, len(head)):
            new.append(head[i] + head[j])
    return new


def mergeslumpdataandhead(head, data):
    outputlist = []
    for i in data:
        outputlist.append(dict(zip(head, i)))

    return outputlist


def classification(cl, data):
    outputlist = []
    for i in data:
        if i["SLUMP(cm)"] == cl:
            del i["SLUMP(cm)"]
            outputlist.append((i, True))
        else:
            del i["SLUMP(cm)"]
            outputlist.append((i, False))
    return outputlist


def slamp(clas):
    postivies = 0

    data, variablenames = import_slump()
    data = definingslumpclass(data)
    variablenames = creatingvaraiblenames(variablenames)
    q = mergeslumpdataandhead(variablenames, data)
    labeled = classification(clas, q)
    # random.shuffle(labeled)

    for i in labeled:
        if i[1]:
            postivies += 1

    variables = []
    var_types = {}
    var_domains = {}

    for i in variablenames[1:]:
        variables.append(i)
        var_types[i] = REAL
        var_domains[i] = (None, None)
    domain = Domain(variables, var_types, var_domains)
    problem = Problem(domain, 0, "slump")
    dir_name_for_equaltions = "../output/{}/{}/".format(problem.name, "theories_learned")
    if not os.path.exists(dir_name_for_equaltions):
        os.makedirs(dir_name_for_equaltions)

    k_fold = KFold(n_splits=5)
    overallruns = []
    foldcount = 1
    equationsfound = []
    for train_indices, test_indices in k_fold.split(labeled):
        train = [labeled[i] for i in train_indices]
        test = [labeled[i] for i in test_indices]

        start = time.time()
        learned_theory, km, hm = learn_parameter_free(problem, train, 2)
        timee = time.time() - start
        overallruns.append([clas, foldcount, timee, accuracy(theorylearned=learned_theory, data1=test),
                            pretty_print(normilisation_rightsideto1(learned_theory))])
        foldcount += 1
        print(overallruns)
        equationsfound.append([smt_to_nested(learned_theory)])

    dir_name_for_equaltions = "../output/{}/{}/class:{}.csv".format(problem.name, "theories_learned", clas)

    with open(dir_name_for_equaltions, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(equationsfound)

    dir_name = "../output/{}/{}{}.csv".format(problem.name, "Results", clas)
    with open(dir_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(overallruns)

    return overallruns, postivies
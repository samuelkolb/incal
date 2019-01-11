import csv


def importwine():
    with open('/Users/Elias/Desktop/winequality/winequality-red.csv', newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=';', quotechar='|')
        linecount = 0
        datalist = []
        for row in datareader:
            if linecount > 0:
                row = [float(x) for x in row]
                datalist.append(row)
            else:
                header = row[:-1]
            linecount += 1
    return datalist, header


def partioning(data, upper, lower):
    datalist = []

    def normilisation(x, min, max, p):
        return (x - min[p]) / (max[p] - min[p])

    minimum = list(map(min, zip(*data)))
    maximum = list(map(max, zip(*data)))

    for i in data:
        if i[-1] <= upper and i[-1] >= lower:
            x = {"fixed acidity": normilisation(i[0], minimum, maximum, 0),
                 "volatile acidity": normilisation(i[1], minimum, maximum, 1),
                 "citric acid": normilisation(i[2], minimum, maximum, 2),
                 "residual sugar": normilisation(i[3], minimum, maximum, 3),
                 "chlorides": normilisation(i[4], minimum, maximum, 4),
                 "free sulfur dioxide": normilisation(i[5], minimum, maximum, 5),
                 "total sulfur dioxide": normilisation(i[6], minimum, maximum, 6),
                 "density": normilisation(i[7], minimum, maximum, 7),
                 "pH": normilisation(i[8], minimum, maximum, 8), "sulphates": normilisation(i[9], minimum, maximum, 9),
                 "alcohol": normilisation(i[10], minimum, maximum, 10)}
            datalist.append((x, True))

        else:
            x = {"fixed acidity": normilisation(i[0], minimum, maximum, 0),
                 "volatile acidity": normilisation(i[1], minimum, maximum, 1),
                 "citric acid": normilisation(i[2], minimum, maximum, 2),
                 "residual sugar": normilisation(i[3], minimum, maximum, 3),
                 "chlorides": normilisation(i[4], minimum, maximum, 4),
                 "free sulfur dioxide": normilisation(i[5], minimum, maximum, 5),
                 "total sulfur dioxide": normilisation(i[6], minimum, maximum, 6),
                 "density": normilisation(i[7], minimum, maximum, 7),
                 "pH": normilisation(i[8], minimum, maximum, 8), "sulphates": normilisation(i[9], minimum, maximum, 9),
                 "alcohol": normilisation(i[10], minimum, maximum, 10)}
            datalist.append((x, False))

    return (datalist)


def unique(list):
    classandclasscount = {}
    for i in list:
        if i[0] not in classandclasscount.keys():
            classandclasscount[i[0]] = 1
        else:
            classandclasscount[i[0]] = classandclasscount[i[0]] + 1
    return classandclasscount

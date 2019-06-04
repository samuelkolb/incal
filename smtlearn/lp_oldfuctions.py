#from lplearing import sample_half_half, evaluate_assignment2


def accuracy(theorylearned, theory=None, size=500, seed=None, data=None):
    if data:
        data=data
    else:
        data = sample_half_half(theory, size, seed)
    #l=[]
    tp=0
    fp=0
    tn=0
    fn=0
    for i in data:
        value=evaluate_assignment2(theorylearned, i[0])

        if i[1] and value:
            tp+=1
        elif i[1] and not value:
            fn+=1
        elif  not i[1] and value:
            fp+=1
        elif not i[1] and not value:
            tn+=1

    return (tn+tp)/(tp+fp+tn+fn)


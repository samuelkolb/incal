#from lplearing import sample_half_half, evaluate_assignment2


def accuracy(theorylearned, theory=None, size=None, seed=None, data1=None):
    if data1:
        data=data1
    else:
        data = sample_half_half(theory, size, seed)
    #l=[]
    tp=0
    fp=0
    tn=0
    fn=0
    for i in data:
        value=evaluate_assignment2(theorylearned, i[0])
        #l.append([i[0],i[1],r])

        if i[1] and value:
            tp+=1
        elif i[1] and not value:
            fn+=1
        elif  not i[1] and value:
            fp+=1
        elif not i[1] and not value:
            tn+=1

    return (tn+tp)/(tp+fp+tn+fn)


#def learn_parameter_free2(problem, data, seed):
 #   feat_x, feat_y = problem.domain.real_vars
  #  print("L")

   # def learn_inc(_data, i, _k, _h):
    #    learner = KCnfSmtLearner(_k, _h, RandomViolationsStrategy(10))
     #   dir_name = "../output/{}".format(problem.name)
      #  img_name = "{}_{}_{}_{}_{}_{}_{}".format("oldversion",len(problem.domain.variables), i, _k, _h, len(data), seed)
       # learner.add_observer(plotting.PlottingObserver(problem.domain, data, dir_name, img_name, feat_x, feat_y))

        #initial_indices = random.sample(list(range(len(data))), 20)

        #learned_theory = learner.learn(problem.domain, data, initial_indices)
        #print("Learned theory:\n{}".format(pretty_print(learned_theory)))
        #return learned_theory

    #return learn_bottom_up(data, learn_inc, 1, 1)
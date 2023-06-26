import os 

n_estimator=[10,20,30,40,50]
max_depth=[1,2,3,4,5]

for i in n_estimator:
    for j in max_depth:
        os.system(f"python basic_ml_model.py -n{i} -m{j}")
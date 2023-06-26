import mlflow

def calculate_sum(x,y):
    return x+y

if __name__=='__main__':
    #starting the server of the mlflow
    with mlflow.start_run():
        x,y=10,20
        z=calculate_sum(x,y)
        #tracking the experiment with mlflow
         mlflow.log_param("X",x)
        mlflow.log_param("Y",y)
        mlflow.log_metric("Z",z)

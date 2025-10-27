from logging_utils.decorators import LoggingFunctionIdentification
from time import sleep
import environment.environment as environment



@LoggingFunctionIdentification("CONTROLLER")
def main():
    print("starting program")

    print(dir(environment))
    print(environment.__file__)

    env = environment.Simulation('./configs/simulation.yaml')
    
    while True:
        env.step()
        sleep(2)



if __name__ == "__main__":
    main()
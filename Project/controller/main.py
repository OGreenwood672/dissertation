from logging.decorators import LoggingFunctionIdentification
import environment


@LoggingFunctionIdentification("CONTROLLER")
def main():
    print(environment.__file__)
    print("starting program")
    environment.environment.run_sim_blocking("/mnt/c/Users/green/OneDrive/Desktop/Dissertation/Project/configs/simulation.yaml")

    environment.environment.print_test()



if __name__ == "__main__":
    main()
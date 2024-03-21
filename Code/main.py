from gradientFunction import*
from descentAlgorythme import*
from methodComparaison import*

def main():
    f1= lambda X: X[0]**2
    f2=lambda X: X[0]**2+X[1]**2+X[0]*X[1]
    f3=lambda X: X[0]*X[1]
    
    convergenceGraph(2, f2, [firstDescent, heavyBallDescent], ["first", "heavy"])


if __name__ == "__main__":
    main()
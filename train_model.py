import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve
import json
import os

# regression lineaire par la methode de descente du gradient :
# on calcule l'erreur quadratique moyenne 
# e = 1/2m * somme pour i allant de 0 a m-1 de (estimatedPrice(mileage[i]) - price[i])**2
# ou
# e(a, b) = 1/2m *somme des (a*xi + b - yi)**2 
# pour diminuer cette erreur, on calcule ses derivees partielles pour avoir le gradiant
# de(a, b)/ da = 1/m * somme des (a*xi + b - yi) * xi
# de(a, b)/ db = 1/m * somme des (a*xi + b - yi) * 1
# de / da : variation de l'erreur quand varie a ie tetha1
# de / db : variation de l'erreur quand varie b ie tetha0
# le gradient indique le sens de la pente/vairation de l'erreur
# on ajuste:
# a = a - rate * de / da 
# b = b - rate * de / db
# on recommence jusqu'a ce que l'erreur soit proche de 0 

def ft_normalize(xi):
    """normalize the km data so that it will be between 0 and 1"""
    try:
        xmax = xi.max()
        xmin = xi.min()
        ai = (xi - xmin) / (xmax - xmin)
        return ai
    except Exception as e:
        print(f"ft_normalize error: {e}")

def ft_denormalize(a, b, xi):
    """denormalize a and b (tetha1 and tetha2)"""
    try:
        xmax = xi.max()
        xmin = xi.min()
        a_origin = a / (xmax - xmin)
        b_origin = b - (a*xmin)/(xmax - xmin)
        return a_origin, b_origin
    except Exception as e:
        print(f"ft_denormalize error: {e}")

def ft_extract_data():
    """extracting data from 'data.csv' file"""
    try :
        xi = []
        yi = []
        with open("data.csv", mode='r', newline='') as file:
            dict_csv = csv.DictReader(file)
            for line in dict_csv:
                xi.append(int(line["km"]))
                yi.append(int(line["price"]))
        return np.array(xi), np.array(yi)
    except ValueError:
        print("Error: invalid data in file 'data.csv'")
    except Exception as e:
        print(f"Extracting data error: {e}") #a verifier plus tard si fonctionne bien

def ft_plot(a, b, xi, yi):
    """plot the data and the line resultating from the linear regression into a graph"""
    try:

        x = np.array(range(0, 300000, 1000))
        y = a* x + b
        plt.plot(x, y, 'r-', label="estimated price")

        # abc = np.polyfit(xi, yi, 1)  # pour test uniquement
        # y2 = abc[1] + abc[0] * x  # pour test uniquement 
        # plt.plot(x, y2, 'g-', label="estimated price by polyset") # pour test uniquement
        # print("poylfyt resultat", abc[1], abc[0])

        plt.plot(xi, yi, 'ob', label="data")
        plt.grid()
        plt.legend()
        plt.xlabel("mileage (in kilometers)")
        plt.ylabel("price (in euros)")
        plt.title("Linear regression of car price")
        plt.show()
        # plt.savefig("grap.png")

        return a, b

    except Exception as e:
        print(f"plot error : {e}")

def ft_get_rmse(xi, yi, a_origin, b_origin):
    """calculate the precision (the root mean squared error) of the algorithm"""
    try:
        n = len(xi)
        rmse = 0
        for i in range(n):
            rmse += (a_origin * xi[i] + b_origin - yi[i])**2
        rmse = rmse *(1/n)
        rmse = np.sqrt(rmse)
        print(f"precision : the linear regression line is mistaken by {rmse:0.1f} euros in mean")
    except Exception as e:
        print(f"rmse error : {e}")


def ft_load_parameters():
    """get parameters a and b from the json file if it exists"""
    a, b = 0.0, 0.0
    try:
        if os.path.exists("parameters.json"):
            with open("parameters.json", "r") as file:
                data = json.load(file)

            a = data["a"]
            b = data["b"]
        return a, b
    except Exception as e:
        print(f"ft_load_parameters error: {e}")
        return 0.0, 0.0

def ft_save_parameters(a, b):
    """"save the parameters a and b (tetha1 and theta0) into a json file"""
    data = {
        "a": a,
        "b": b
    }

    with open("parameters.json", "w") as file:
        json.dump(data, file)

def ft_linear_regression():
    try:

        xi, yi = ft_extract_data()

        ai = ft_normalize(xi)

        rate = 0.1 #learning rate 
        a = 0  #tetha1 
        b = 0  #tetha0 

        Niter = 10000 
        e = 1.e-6 
        m = len(xi)
        n = 0

        err2 = (1 / (2 * m)) * ((a * ai + b - yi)**2).sum()
        err1 = 1 
        while abs(err2 - err1) > e and n < Niter:
            tmpa = rate * ((a * ai + b - yi) * ai).sum() * (1/m)
            tmpb = rate * (a * ai + b - yi).sum() * (1/m)
            a = a - tmpa  
            b = b - tmpb
            err1 = err2
            err2 = (1 / (2 * m)) * ((a * ai + b - yi)**2).sum()
            n += 1

        a, b = ft_denormalize(a, b, xi)
        a, b = ft_plot(a, b, xi, yi)
        print("a, b =", a , b)

        ft_get_rmse(ai, yi, a, b)

        ft_save_parameters(a, b)
    except Exception as e:
        print(f"ft_linear_regression error : {e}")


if __name__ == '__main__':
    ft_linear_regression()

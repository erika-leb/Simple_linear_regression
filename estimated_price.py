import numpy as np
import sys
from train_model import ft_load_parameters

def main():
    """predict car price for a given mileage"""
    try:
        txt = ""
        while (txt != "q"):
            print("Please enter the mileage (or press 'q' to quit): ")
            txt = sys.stdin.readline()
            if txt == "q\n":
                return
            try:
                mil = float(txt)
            except Exception as e:
                print(f"Error: mileage should be an int or a float")
                continue
            if mil <= 0:
                raise ValueError("Error: mileage should be above 0")
            try:
                teta1, teta0 = ft_load_parameters()
            except Exception as e:
                print(f"loading of tetha0 and tetha1 failed: {e}")
                return
            estim_price = teta0 + (teta1 * mil)
            if (estim_price < 0):
                estim_price = 0
            print(f"\nEstimated price: {estim_price:.0f} euros\n" )
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
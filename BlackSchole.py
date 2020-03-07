import numpy as np
from scipy.stats import norm


class GBSGreeks():
    def blackScholesOptionPrice(S, K, T, r, sigma, option):
        # S is current stock price
        # K is strike price
        # T is tenor to expire on the option maturity
        # r is annualized rate of interest
        # sigma is the annualized volatulity of the underlying security
        # option is either call or put

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        if option == 'call':
            result = (S * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))

        if option == 'put':
            result = (K * np.exp(-r * T) * norm.cdf(-d2, 0.0, 1.0) - S * norm.cdf(-d1, 0.0, 1.0))

        return result

    # Greek letter Delta method
    def Delta(S, K, T, r, sigma, optionType="c"):  # # default is optionType="c"
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if optionType == "c":

            return norm.cdf(d1)

        elif optionType == "p":

            return norm.cdf(d1) - 1

    # Greek letter Gamma method
    def Gamma(S, K, T, r, sigma, optionType="c"):

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if optionType == "c" or optionType == "p":
            return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Greek letter Vega method
    def Vega(S, K, T, r, sigma, optionType="c"):

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if optionType == "c" or optionType == "p":
            return S * np.sqrt(T) * norm.pdf(d1)

    # Greek letter Theta method
    def Theta(S, K, T, r, sigma, optionType="c"):  # # default is optionType="c"
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if optionType == "c":

            return -S * sigma * norm.pdf(d1) / (2 * np.sqrt(T)) - K * r * np.exp(-r * T) * norm.cdf(d2)

        elif optionType == "p":

            return -S * sigma * norm.pdf(d1) / (2 * np.sqrt(T)) + K * r * np.exp(-r * T) * norm.cdf(d2)

    # Greek letter Rho method
    def Rho(S, K, T, r, sigma, optionType="c"):  # # default is optionType="c"
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if optionType == "c":

            return K * T * np.exp(-r * T) * norm.cdf(d2)

        elif optionType == "p":

            return -K * T * np.exp(-r * T) * norm.cdf(-d2)

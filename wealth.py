import numpy as np
import math
import random
from sympy import *
from scipy.stats import norm
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm

class stochastic_wealth_matrix():

    def __init__(self):
        np.set_printoptions(suppress=True, precision=2)
        self.start_capital =  0 #3.5*10**6
        self.wage_income = 45000
        self.wage_CAGR = 0.03
        self.tax_rate = 0.2
        self.expenses = 2000
        self.investment_CAGR = 0.07
        self.inflation_CAGR = 0.03
        self.debt = 0
        self.debt_interest_rate = 0.05
        self.start_date = 0
        self.months_simulated = NUM_MONTHS

        self.wealth_matrix = np.zeros((self.months_simulated,11))
        self.wealth_matrix[1:,1] = self.wage_income/12
        self.wealth_matrix[0,8] = self.start_capital
        self.wealth_matrix[0:,6] = self.debt


        # x,m,s = symbols("x m s")
        # # self.cdf = erf(sqrt(2)*(-m + x)/(2*s))/2
        # self.cdf = 1/2*(1+erf((x-m)/(s*sqrt(2))))

    def norm_dist(self, mean, std_dev):
        # y,x,m,s = symbols("y x m s")
        # expr = self.cdf.subs(m,mean).subs(s,std_dev)
        # return solve(Eq(expr,y),x)[0].subs(y,random.randrange(-5*10**9,5*10**9)/10**10).evalf()
        return norm.ppf(random.randrange(0,10**10)/10**10,mean,std_dev)

    def run_simulation(self):
        for month in range(self.months_simulated):
            # print("Simulating Month:{}".format(month))
            if month==0:
                pass
            else:
                self.wealth_matrix[month][0] = month
                if month%12==0:
                    self.wealth_matrix[month:,1]=self.wealth_matrix[month][1]*(1+self.norm_dist(self.wage_CAGR,0.02))
                self.wealth_matrix[month][2] = self.norm_dist(self.expenses*((1+self.wealth_matrix[month-1][4])**(month)),\
                                                400*((1+self.wealth_matrix[month-1][4])**(month)))
                self.wealth_matrix[month][3] = self.wealth_matrix[month][1]*self.tax_rate
                self.wealth_matrix[month][4] = self.norm_dist(self.inflation_CAGR,0.01)/12
                self.wealth_matrix[month][5] = self.norm_dist(self.investment_CAGR,0.10)/12
                # Residual
                self.wealth_matrix[month][7] = self.wealth_matrix[month][1] - self.wealth_matrix[month][2] - self.wealth_matrix[month][3]
                if self.wealth_matrix[month][6] > 0:
                    if self.wealth_matrix[month][7] > self.wealth_matrix[month][6]:
                        self.wealth_matrix[month][7] -= self.wealth_matrix[month][6]
                        self.wealth_matrix[month:,6] = 0
                    else:
                        self.wealth_matrix[month:,6] = (self.wealth_matrix[month,6]-self.wealth_matrix[month][7])*(1+self.debt_interest_rate/12)
                        self.wealth_matrix[month][7] = 0
                # Investeted Capital
                self.wealth_matrix[month][8] = self.wealth_matrix[month][7] + self.wealth_matrix[month-1][8] * (1 + self.wealth_matrix[month][5])
                # Net Worth
                self.wealth_matrix[month][9] = self.wealth_matrix[month][8] - self.wealth_matrix[month][6]
                # Inflation Adujusted
                self.wealth_matrix[month][10] = self.wealth_matrix[month][9]/(1+self.inflation_CAGR/12)**month # This should grow with Gaussian Inflation CAGR Value

        # for i in self.wealth_matrix:
        #     print(i[6],i[1]-i[2]-i[3])
        #
        # plt.plot(self.wealth_matrix[1:,0],self.wealth_matrix[1:,6])
        # plt.plot(self.wealth_matrix[1:,0],self.wealth_matrix[1:,8:11])
        # plt.show()

NUM_MONTHS = 30*12
AGE = 25
num_sims = 250
target = 1.0*10**8 #5.0*10**8
monte_carlo_matrix = np.zeros((NUM_MONTHS,num_sims))
for sim in tqdm(range(num_sims)):
    # print("Simulation: {}".format(sim+1))
    simulation = stochastic_wealth_matrix()
    simulation.run_simulation()
    monte_carlo_matrix[:,sim]=simulation.wealth_matrix[:,9]

print("Mean: ${:,.2f} Standard Deviation: ${:,.2f}".format(round(monte_carlo_matrix[-1,:].mean(),2),round(monte_carlo_matrix[-1,:].std(),2)))
print("Your purchasing power in {} dollars will be ${:,.2f}".format(datetime.date.today().year, round(monte_carlo_matrix[-1,:].mean())/((1+0.03)**(NUM_MONTHS/12))))
print("You have a {}% Chance of Accumulating \u2265 ${:,.2f} by the age of {}".format(round(1-norm.cdf(target,monte_carlo_matrix[-1,:].mean(),monte_carlo_matrix[-1,:].std()),2)*100,target,AGE+(NUM_MONTHS/12)))
plt.plot([x for x in range(len(monte_carlo_matrix[1:,0]))],monte_carlo_matrix[1:])
plt.show()

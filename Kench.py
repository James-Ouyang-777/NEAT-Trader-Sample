'''
General settings and implementation of the single-pole cart system dynamics.
'''

from math import cos, pi, sin
import random



class Game(object):
    # gravity = 9.8  # acceleration due to gravity, positive is downward, m/sec^2
    # mcart = 1.0  # cart mass in kg
    # mpole = 0.1  # pole mass in kg
    # lpole = 0.5  # half the pole length in meters
    time_step = 0.01  # time step in seconds

    def __init__(self, price=100, position=0, cash=10000, equity=10000, indicator1=0,
                 indicator2=0, t=0):

        self.price = 0
        self.position = 0
        self.cash = 10000
        self.equity = 10000

        self.indicator1 = 0
        self.indicator2 = 0

        self.t = 0


    def step(self, decision, nextprice):
        if decision == 1:
            self.position += 1
            self.cash -= self.price
        elif decision == -1:
            self.position -= 1
            self.cash += self.price

        self.equity = self.price*self.position+self.cash

        #### go to next day

        self.price = nextprice
        self.t += 1



        

    def get_scaled_state(self): ##########need to figure out what the scaled state needs to be.
        '''Get full state, scaled into (approximately) [0, 1].'''
        return [self.price, self.position, self.equity, self.indicator1, self.indicator2]
        # return [0.5 * (self.x + self.position_limit) / self.position_limit,
        #         (self.dx + 0.75) / 1.5,
        #         0.5 * (self.theta + self.angle_limit_radians) / self.angle_limit_radians,
        #         (self.dtheta + 1.0) / 2.0]


def buy_sell_action(action):
    if action[0] > 0.7:
        return 1
    elif action[0] < 0.3:
        return 0
    else:
        return -1

def continuous_actuator_force(action):
    return -10.0 + 2.0 * action[0]


def noisy_continuous_actuator_force(action):
    a = action[0] + random.gauss(0, 0.2)
    return 10.0 if a > 0.5 else -10.0


def discrete_actuator_force(action):
    return 10.0 if action[0] > 0.5 else -10.0


def noisy_discrete_actuator_force(action):
    a = action[0] + random.gauss(0, 0.2)
    return 10.0 if a > 0.5 else -10.0

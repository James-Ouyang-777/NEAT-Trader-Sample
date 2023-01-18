'''
Single-pole balancing experiment using a continuous-time recurrent neural network (CTRNN).
'''

from __future__ import print_function

import os
import pickle

import Kench

import neat
from neat.math_util import mean
import visualize
import pandas as pd

df = pd.read_csv('TahmKenchNEAT\pricedata\Gemini_LUNAUSD_1h.csv')
LUNA_openPrice = df['open'].tolist()
LUNA_openPrice.reverse()

df = pd.read_csv('TahmKenchNEAT\pricedata\BTCUSD2020.csv')
BTC_openPrice = df['open'].tolist()
BTC_openPrice.reverse()


runs_per_net = 5
simulation_length = 25
time_const = 0.01

# Use the CTRNN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.ctrnn.CTRNN.create(genome, config, time_const)

    fitnesses = []
    for runs in range(runs_per_net):
        sim = Kench.Game()
        net.reset()

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.t < simulation_length:
            inputs = sim.get_scaled_state()
            action = net.advance(inputs, time_const, time_const)
           # print("ACTION: ")
            #print(action)

            # Apply action 
            decision = Kench.buy_sell_action(action) #WTF IS THIS
            nextprice = BTC_openPrice[sim.t]
            sim.step(decision, nextprice)

            # Stop if the network fails to keep the cart within the position or angle limits.
            # The per-run fitness is the number of time steps the network can balance the pole
            # without exceeding these limits.
            # if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
            #     break

            fitness = sim.equity  ## <<<< tab over?

        fitnesses.append(fitness)

        #print("{0} fitness {1}".format(net, fitness))


    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-ctrnn')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    if 0:
        winner = pop.run(eval_genomes)
    else:
        pe = neat.ParallelEvaluator(5, eval_genome)
        winner = pop.run(pe.evaluate)

    # Save the winner.
    with open('winner-ctrnn', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    visualize.plot_stats(stats, ylog=True, view=True, filename="ctrnn-fitness.svg")
    visualize.plot_species(stats, view=True, filename="ctrnn-speciation.svg")

    node_names = {-1: 'var1', -2: 'var2', -3: 'var3', -4: 'var4', -5: 'var5', 0: 'Output'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-ctrnn.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-ctrnn-enabled.gv", show_disabled=False)
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-ctrnn-enabled-pruned.gv", show_disabled=False, prune_unused=True)


if __name__ == '__main__':
    run()

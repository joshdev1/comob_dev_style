from generating_stimuli.graph import graph_stimuli
from generating_stimuli.random_ipd_input import RandomIpdInput


random_ipd_input = RandomIpdInput(8)
ipd, spikes = random_ipd_input.generate()
graph_stimuli(ipd, spikes)

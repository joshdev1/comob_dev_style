from generating_stimuli.generate_stimuli import random_ipd_input_signal
from generating_stimuli.graph import graph_stimuli

ipd, spikes = random_ipd_input_signal(8)
graph_stimuli(ipd, spikes)

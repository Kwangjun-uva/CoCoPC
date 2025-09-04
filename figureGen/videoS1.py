from figureGen.training_data import sim_params, weights, dataset
from script.animation import animation_creator


# dir
model_dir = '../results/'

# load animation creator
animator = animation_creator(simParams=sim_params, pretrained_weights=weights)
# simulate network responses to a random image
animator.simulate(dataset=dataset)
# generate animation
animator.generate_animation(filename='../models/pyr_response_animation.mp4', frame_step=2, fps=60)
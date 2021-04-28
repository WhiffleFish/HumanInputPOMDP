include("../src/tree_trajectories.jl")
include("../src/display.jl")
include("../src/ChooseTheBetterMDP.jl")
# using StaticArrays
# game = Game(max_steps=50, reward_variance=10)

# play(game)

##
game = IMGame(reward_variance=10.0, s0=SA[5,5])

play(game)

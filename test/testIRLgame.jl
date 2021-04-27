include("../src/tree_trajectories.jl")
include("../src/display.jl")
include("../src/2irl4me.jl")
# include("../src/ChooseTheBetterMDP.jl")
using StaticArrays
# game = Game(max_steps=50, reward_variance=10)

# play(game)

##
game = UpdatingGame()

play(game)
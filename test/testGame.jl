include("../src/tree_trajectories.jl")
include("../src/display.jl")
include("../src/ChooseTheBetterMDP.jl")

game = Game(max_steps=50, reward_variance=10)

play(game)

##

#=
if tprob is too low, lots of random walking causes planner to
not be able to plan past wall of negative reward
=#

mdp = SimpleGridWorld(tprob=0.80)
solver = MCTSSolver(n_iterations=2000, depth=60, exploration_constant=10.0, enable_tree_vis=true)
for i = 1:7; mdp.rewards[SA[6,i]] = -5.0;end

game = UpdatingGame(true_mdp=mdp,reward_variance=2.0, s0=SA[1,1], max_steps=70, solver = solver)

play(game, show_true=false)

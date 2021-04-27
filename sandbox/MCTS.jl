using MCTS
using POMDPs
using POMDPModels
using StaticArrays
using D3Trees
using Random
include("../src/tree_trajectories.jl")
include("../src/display.jl")

mdp = SimpleGridWorld(tprob=0.70)

solver = MCTSSolver(n_iterations=1000, depth=30, exploration_constant=5.0,enable_tree_vis=true)
planner = solve(solver, mdp)
a, info = action_info(planner, SA[5,5])

tree = info[:tree]

# inchrome(D3Tree(tree))

##

trajectories = get_trajectories(mdp,tree, 100, 70)

render(rewards, [5,5], trajectories)

for x in trajectories[2]
    println(x)
end

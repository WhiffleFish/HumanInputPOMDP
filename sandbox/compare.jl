using MCTS
using POMDPs
using POMDPModels
using StaticArrays
using D3Trees
using Random
include("../src/tree_trajectories.jl")
include("../src/display.jl")

true_mdp = SimpleGridWorld()
mdp1 = SimpleGridWorld()
mdp2 = SimpleGridWorld()
mdp1.rewards[SA[9,3]] = 2.0
mdp1.rewards[SA[4,3]] = 0.0

mdp2.rewards[SA[9,3]] = 0.0
mdp2.rewards[SA[8,8]] = 10.0

function solved_tree(mdp::SimpleGridWorld, solver::MCTSSolver, s0::SVector{2, Int})
    planner = solve(solver, mdp)
    a, info = action_info(planner, s0)
    return info[:tree]
end

true_rewards = reward_grid(true_mdp)
rewards1 = reward_grid(mdp1)
rewards2 = reward_grid(mdp2)

start = SA[5,5]
solver = MCTSSolver(n_iterations=1000, depth=30, exploration_constant=5.0,enable_tree_vis=true)
planner_true = solve(solver, true_mdp)
planner1 = solve(solver, mdp1)
planner2 = solve(solver, mdp2)
a, true_info = action_info(planner_true, start)
a, info1 = action_info(planner1, start)
a, info2 = action_info(planner2, start)

true_tree = true_info[:tree]
tree1 = info1[:tree]
tree2 = info2[:tree]

# inchrome(D3Tree(tree))

##


for i = 1:1
    Drawing(1000,1000)
    origin()
    background("black")
    sethue("red")
    fontsize(50)
end

t_paths = get_trajectories(true_mdp,true_tree, 100, 50)
paths1 = get_trajectories(mdp1,tree1, 100, 50)
paths2 = get_trajectories(mdp2,tree2, 100, 50)

render(rewards, [5,5], t_paths, paths2)

for x in trajectories[2]
    println(x)
end

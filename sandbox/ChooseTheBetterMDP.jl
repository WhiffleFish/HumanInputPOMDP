using MCTS
using POMDPs
using POMDPModels
using StaticArrays
using D3Trees
using Random
include("../src/tree_trajectories.jl")
include("../src/display.jl")

true_mdp = SimpleGridWorld(tprob=0.95)
true_rewards = reward_grid(true_mdp)


steps = 50
s = SA[5,5]
total_reward = 0.0
solver = MCTSSolver(n_iterations=5000, depth=30, exploration_constant=5.0,enable_tree_vis=true)
for i = 1:steps
    println("\nStep: $i")

    mdp_blue, mdp_orange = genMDP(), genMDP()
    a_blue, tree_blue = solved_mdp(mdp_blue, solver, s)
    a_orange, tree_orange = solved_mdp(mdp_orange, solver, s)
    paths_b = get_trajectories(mdp_blue,tree_blue, 100, 50)
    paths_o = get_trajectories(mdp_orange,tree_orange, 100, 50)

    if isterminal(mdp, s)
        println("Game over")
        break
    end

    display(render(true_rewards, s, paths_b, paths_o))

    println("[B]lue or [O]range?")
    input = readline()
    if input == "B"
        a = a_blue
    elseif input == "O"
        a = a_orange
    else
        println("Invalid Input")
        break
    end
    s,r = @gen(:sp,:r)(true_mdp,s,a)
    total_reward += r
end

##
rewards = reward_grid(true_mdp)

t_paths = get_trajectories(true_mdp,true_tree, 100, 50)
paths1 = get_trajectories(mdp1,tree1, 100, 50)
paths2 = get_trajectories(mdp2,tree2, 100, 50)

render(rewards, [5,5], t_paths, paths2)

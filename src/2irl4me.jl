using MCTS
using POMDPs
using POMDPModels
include("mdp.jl")
using StaticArrays
using Parameters
include("birl.jl")

@with_kw mutable struct MCMCGame
    solver::MCTSSolver = MCTSSolver(n_iterations=1000, depth=40, exploration_constant=10.0, enable_tree_vis=true)
    s0::SVector{2,Int} = SA[5,5]
    max_steps::Int = 30
    true_mdp::SimpleGridWorld = OneGoodSimpleGridWorld()
    reward_ranges::Array{Tuple{Float64, Float64}, 1} = [(-15.0, 15.0), (-15.0, 15.0), (-15.0, 15.0), (-15.0, 15.0)]
    reward_belief::Array{Array{Float64,1},1}= [rand(length(true_mdp.terminate_from))]
end

function play(game::MCMCGame; show_true=false)
    solver = game.solver
    s = game.s0
    steps = game.max_steps
    true_rewards = reward_grid(game.true_mdp)
    total_reward = 0.0

    phi_As = []
    phi_Bs = []
    prefs = []
    confidences = []
    aut_steps = 0
    for i = 1:steps
        println("\nStep: $i")

        if isterminal(game.true_mdp, s)
            println("Game over")
            println("Final Score: $total_reward")
            break
        end

        #User Input
        if aut_steps <= 0
            mdp_main, phi_A = meanMDP(game.reward_belief, game.reward_ranges)
            # mdp_main, phi_A = genVarMDP(game.reward_belief ,game.reward_ranges)
            # mdp_alt, phi_B = genMDP(game.reward_belief, game.reward_ranges)
            mdp_alt, phi_B = genVarMDP(game.reward_belief , game.reward_ranges)

            @show phi_A, phi_B
            println("a")
            @show mdp_main.rewards
            a_blue, tree_blue = solved_mdp(mdp_main, solver, s)
            println("b")
            @show mdp_alt.rewards
            a_orange, tree_orange = solved_mdp(mdp_alt, solver, s)
            while (a_orange == a_blue && !(s in game.true_mdp.terminate_from))
                mdp_alt, phi_B = genVarMDP(game.reward_belief , game.reward_ranges)
                a_orange, tree_orange = solved_mdp(mdp_alt, solver, s)
            end
            paths_b = get_trajectories(mdp_main,tree_blue, 100, 50)
            paths_o = get_trajectories(mdp_alt,tree_orange, 100, 50)
            if show_true
                display(render(true_rewards, s, paths_b, paths_o))
            else
                display(render(reward_grid(mdp_main), s, paths_b, paths_o))
            end
            println("[B]lue or [O]range?")
            input = split(readline())
            aut_steps = 1
            if length(input) == 0
                confidence = 1.0
                choice = "s"
            elseif length(input) == 1
                confidence = 1.0
                choice = lowercase(input[1])
            elseif length(input) == 3
                choice = lowercase(input[1])
                confidence = tryparse(Int, input[2])
                aut_steps = tryparse(Int, input[3])
            elseif length(input) == 2
                choice = lowercase(input[1])
                confidence = tryparse(Int, input[2])
            else
                println("Invalid Input")
                break
            end

            if choice == "b"
                pref = 1
                a = a_blue
            elseif choice == "o"
                pref = -1
                a = a_orange
            elseif choice == "s"
                pref = 0
                a = a_blue
            else
                println("Invalid Input")
                break
            end
            push!(phi_As, phi_A)
            push!(phi_Bs, phi_B)
            push!(prefs, pref)
            push!(confidences, confidence)
            update_rewards!(game, phi_As, phi_Bs, prefs, confidences)
            # mdp_main, phi_A = meanMDP(game.reward_belief, game.reward_ranges)
            # a, tree_blue = solved_mdp(mdp_main, solver, s)
        else
            a, tree_blue = solved_mdp(mdp_main, solver, s)
            paths_b = get_trajectories(mdp_main,tree_blue, 100, 50)
            if show_true
                display(render(true_rewards, s, paths_b))
            else
                display(render(reward_grid(mdp_main), s, paths_b))
            end
        end

        s,r = @gen(:sp,:r)(game.true_mdp,s,a)
        total_reward += r
        aut_steps -= 1
    end
end

function update_rewards!(game::MCMCGame, phi_As, phi_Bs, prefs, confidences)::Nothing

    # @show mean(game.reward_belief)
    # @show phi_A
    # @show phi_B
    # @show phi_As[end], phi_Bs[end], prefs[end], confidences[end]  
    # reward_values = policy_walk(mean(game.reward_belief), phi_As, phi_Bs, prefs, confidences)
    reward_values = policy_walk(mean(game.reward_belief), phi_As[end], phi_Bs[end], prefs[end], confidences[end])
    sort!(reward_values, lt=my_compare)
    # @show reward_values
    game.reward_belief = reward_values
    @show mean(reward_values)
    nothing
end

using MCTS
using POMDPs
using POMDPModels
using StaticArrays
using Parameters

@with_kw struct Game
    solver::MCTSSolver = MCTSSolver(n_iterations=1000, depth=40, exploration_constant=5.0, enable_tree_vis=true)
    s0::SVector{2,Int} = SA[1,1]
    max_steps::Int = 30
    true_mdp::SimpleGridWorld = SimpleGridWorld()
    reward_variance::Float64 = 5.0
end

@with_kw mutable struct IMGame
    solver::MCTSSolver = MCTSSolver(n_iterations=2000, depth=50, exploration_constant=20.0, enable_tree_vis=true)
    s0::SVector{2,Int} = SA[1,1]
    max_steps::Int = 30
    true_mdp::SimpleGridWorld = SimpleGridWorld()
    reward_variance::Float64 = 5.0
    pred_mdp::SimpleGridWorld = initMDP(true_mdp)
    C::Int = 1 # Start at 1 to prevent NaN if first input is 0 confidence
end

"""
Inputs
- `game::IMGame`
- `mdp_alt::SimpleGridWorld`

Outputs

"""
function query(game::IMGame, mdp_alt::SimpleGridWorld, a_blue::Symbol, a_orange::Symbol)
    println("[B]lue or [O]range?")
    input = readline()
    if lowercase(input) == "exit"
        return (nothing, nothing, nothing)
    end

    input = split(input)
    aut_steps = 1
    confidence = 0
    if length(input) == 0
        choice = "b"
    elseif length(input) == 1
        choice = lowercase(input[1])
    elseif length(input) == 2
        choice = lowercase(input[1])
        confidence = tryparse(Int, input[2])
    elseif length(input) == 3
        choice = lowercase(input[1])
        confidence = tryparse(Int, input[2])
        aut_steps = tryparse(Int, input[3])
    else
        println("Invalid Input")
        (a, confidence, aut_steps) = query(game, mdp_alt, a_blue, a_orange)
    end

    if choice == "b"
        a = a_blue
        update_rewards!(game, game.pred_mdp, confidence)
    elseif choice == "o"
        a = a_orange
        update_rewards!(game, mdp_alt, confidence)
    else
        println("Invalid Input")
        (a, confidence, aut_steps) = query(game, mdp_alt, a_blue, a_orange)
    end

    return (a, confidence, aut_steps)
end

function play(game::Game)
    solver = game.solver
    s = game.s0
    steps = game.max_steps
    true_rewards = reward_grid(game.true_mdp)
    var = game.reward_variance
    total_reward = 0.0
    for i = 1:steps
        println("\nStep: $i")

        mdp_blue, mdp_orange = genMDP(game.true_mdp, game.reward_variance), genMDP(game.true_mdp, game.reward_variance)

        a_blue, tree_blue = solved_mdp(mdp_blue, solver, s)
        a_orange, tree_orange = solved_mdp(mdp_orange, solver, s)
        paths_b = get_trajectories(mdp_blue,tree_blue, 100, 50)
        paths_o = get_trajectories(mdp_orange,tree_orange, 100, 50)

        if isterminal(mdp, s)
            println("Game over")
            println("Final Score: $total_reward")
            break
        end

        display(render(true_rewards, s, paths_b, paths_o))

        println("[B]lue or [O]range?")
        input = readline()
        choice = lowercase(input)

        if choice == "b"
            a = a_blue
        elseif choice == "o"
            a = a_orange
        else

            println("Invalid Input")
            break
        end
        s,r = @gen(:sp,:r)(true_mdp,s,a)
        total_reward += r
    end
end

function play(game::IMGame; show_true=false)::Float64
    solver = game.solver
    s = game.s0
    steps = game.max_steps
    true_rewards = reward_grid(game.true_mdp)
    var = game.reward_variance
    total_reward = 0.0
    γ = discount(game.true_mdp)

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

            mdp_alt = genBetaMDP(game.pred_mdp, game.reward_variance)
            a_blue, tree_blue = solved_mdp(game.pred_mdp, solver, s)
            a_orange, tree_orange = solved_mdp(mdp_alt, solver, s)
            paths_b = get_trajectories(game.pred_mdp,tree_blue, 100, 50)
            paths_o = get_trajectories(mdp_alt,tree_orange, 100, 50)
            if show_true
                display(render(true_rewards, s, paths_b, paths_o))
            else
                display(render(reward_grid(game.pred_mdp), s, paths_b, paths_o))
            end
            (a,confidence,aut_steps) = query(game, mdp_alt, a_blue, a_orange)
            if a == nothing; break; end
        else
            a, tree_blue = solved_mdp(game.pred_mdp, solver, s)
            paths_b = get_trajectories(game.pred_mdp,tree_blue, 100, 50)
            if show_true
                display(render(true_rewards, s, paths_b))
            else
                display(render(reward_grid(game.pred_mdp), s, paths_b))
            end
        end

        s,r = @gen(:sp,:r)(game.true_mdp,s,a)
        total_reward += r*γ^(i-1)
        aut_steps -= 1
    end
    return total_reward
end

function update_rewards!(game::IMGame, mdp::SimpleGridWorld, confidence::Int)::Nothing
    # println("Pre-update")
    # @show game.pred_mdp.rewards
    game.C += confidence
    if (mdp != game.pred_mdp)
        # println("Orange Chosen")
        for (k,v) in game.pred_mdp.rewards # Incremental Avg Update
            game.pred_mdp.rewards[k] = game.pred_mdp.rewards[k] + (confidence/game.C)*(mdp.rewards[k] - game.pred_mdp.rewards[k])
        end
        # @show mdp.rewards
    else
        # println("Blue Chosen")
    end
    # @show game.pred_mdp.rewards
    nothing
end

struct ScriptedTeacher
    confidence::Float64
    true_mdp::SimpleGridWorld
end

function ScriptedTeacher(confidence::Float64, game::IMGame)
    return ScriptedTeacher(confidence, game.true_mdp)
end

confidence2variance(c::Float64) = -10log(c)

function tree2Qvec(tree::MCTS.MCTSTree{SVector{2, Int64}})::Vector{Float64}
    return tree.q[1:4]
end

# normed(v::Vector{Float64}) = (v .- minimum(v))./(maximum(v)-minimum(v))
normed(v::Vector{Float64}) = v./sqrt(sum(abs2,v))

function normdiff(v1::Vector{Float64},v2::Vector{Float64})::Float64
    sum(abs2, normed(v1) .- normed(v2))/4
end


function query(teacher::ScriptedTeacher, game::IMGame, alt_tree::MCTS.MCTSTree{SVector{2, Int64}}, pred_tree::MCTS.MCTSTree{SVector{2, Int64}}, s::SVector{2,Int})
    teach_mdp = genMDP(teacher.true_mdp, confidence2variance(teacher.confidence))
    a_teach, teach_tree = solved_mdp(teach_mdp, game.solver, s)
    teach_Q = tree2Qvec(teach_tree)
    alt_Q = tree2Qvec(alt_tree)
    pred_Q = tree2Qvec(pred_tree)

    @show teach_Q
    @show alt_Q
    @show pred_Q

    alt_diff = normdiff(teach_Q, alt_Q)
    pred_diff = normdiff(teach_Q, pred_Q)

    @show alt_diff
    @show pred_diff

    #=
    Confidence in a trajectory is inversely proportional to how much it differs
    from your given optimal trajectory
    =#
    if alt_diff < pred_diff
        choice_confidence = round(Int,10*(1.0 - alt_diff)^2)
        @show choice_confidence
        return "o",choice_confidence
    else
        choice_confidence = min(3,round(Int,10*(1.0 - pred_diff)^2))
        @show choice_confidence
        return "b", choice_confidence
    end
    # If difference between predicted Q values are
end

function play(game::IMGame, teacher::ScriptedTeacher; show_true=false)
    solver = game.solver
    s = game.s0
    steps = game.max_steps
    true_rewards = reward_grid(game.true_mdp)
    var = game.reward_variance
    total_reward = 0.0
    γ = discount(game.true_mdp)
    game.C = 1
    for i = 1:steps
        println("\nStep: $i")

        if isterminal(game.true_mdp, s)
            println("Game over")
            println("Final Score: $total_reward")
            break
        end
        a_true, tree_true = solved_mdp(game.true_mdp, solver, s) # REMOVE
        mdp_alt = genMDP(game.pred_mdp, game.reward_variance)
        a_blue, tree_blue = solved_mdp(game.pred_mdp, solver, s)
        a_orange, tree_orange = solved_mdp(mdp_alt, solver, s)
        paths_b = get_trajectories(game.pred_mdp,tree_blue, 100, 50)
        paths_o = get_trajectories(mdp_alt,tree_orange, 100, 50)
        paths_true = get_trajectories(game.true_mdp,tree_true, 100, 50) # REMOVE

        choice, confidence = query(teacher, game, tree_orange, tree_blue, s)
        if choice == "b"
            a = a_blue
            update_rewards!(game, game.pred_mdp, confidence)
        else
            a = a_orange
            update_rewards!(game, mdp_alt, confidence)
            # a, _ = solved_mdp(game.pred_mdp, solver, s)
        end

        if show_true
            display(render(true_rewards, s, paths_b, paths_o))
        else
            display(render(reward_grid(game.pred_mdp), s, paths_b, paths_o, paths_true))
        end

        s,r = @gen(:sp,:r)(game.true_mdp,s,a)
        total_reward += r*γ^(i-1)
    end
    return total_reward
end

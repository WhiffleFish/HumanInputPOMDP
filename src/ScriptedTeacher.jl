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


function query(teacher::ScriptedTeacher,
        game::IMGame,
        alt_tree::MCTS.MCTSTree{SVector{2, Int64}},
        pred_tree::MCTS.MCTSTree{SVector{2, Int64}},
        s::SVector{2,Int},
        deltaR::Float64,
        alt_mdp::SimpleGridWorld,
        alpha::Float64 = 0.5
    )

    teach_mdp = genMDP(teacher.true_mdp, confidence2variance(teacher.confidence))
    a_teach, teach_tree = solved_mdp(teach_mdp, game.solver, s)
    teach_Q = tree2Qvec(teach_tree)
    alt_Q = tree2Qvec(alt_tree)
    pred_Q = tree2Qvec(pred_tree)

    # @show teach_Q
    # @show alt_Q
    # @show pred_Q

    # alt_diff = normdiff(teach_Q, alt_Q)
    # pred_diff = normdiff(teach_Q, pred_Q)
    # altpred_diff = normdiff(alt_Q, pred_Q)

    teach_r = normalize(collect(values(teach_mdp.rewards)))
    r_max_idx = argmax(teach_r)

    pred_r = normalize(collect(values(game.pred_mdp.rewards)))
    alt_r = normalize(collect(values(alt_mdp.rewards)))

    alt_diff_r = 1.0 - alt_r[r_max_idx]
    pred_diff_r = 1.0 - pred_r[r_max_idx]

    alt_diff = normdiff(teach_r, alt_r)
    pred_diff = normdiff(teach_r, pred_r)

    altpred_diff = normdiff(alt_r, pred_r)
    @show teach_r
    @show pred_r
    @show alt_r
    @show alt_diff_r
    @show pred_diff_r

    # Stop querying if diff between ideal R and predicted R is sufficiently low
    done = normdiff(teach_r,pred_r) < deltaR

    #=
    Confidence in a trajectory is inversely proportional to how much it differs
    from your given optimal trajectory
    =#
    if alt_diff < pred_diff
        choice_confidence = alpha*(altpred_diff) + (1-alpha)*(alt_diff)
        choice_confidence = round(Int,10*choice_confidence)
        return "o",choice_confidence, done
    else
        choice_confidence = alpha*(altpred_diff) + (1-alpha)*(pred_diff)
        choice_confidence = round(Int,10*choice_confidence)
        return "b", choice_confidence, done
    end
    # If difference between predicted Q values are
end


"""
Returns tuple
- Total discounted reward
- Point at which solver is confident and stops updating
- total time steps
"""
function play(game::IMGame, teacher::ScriptedTeacher; show_true=false, deltaR::Float64=0.1, alpha=0.5)
    solver = game.solver
    s = game.s0
    steps = game.max_steps
    true_rewards = reward_grid(game.true_mdp)
    var = game.reward_variance
    total_reward = 0.0
    γ = discount(game.true_mdp)
    game.C = 1

    done_querying = false
    confident_point = nothing
    i = 0
    while i < steps
        i += 1
        println("\nStep: $i")

        if isterminal(game.true_mdp, s)
            println("Game over")
            println("Final Score: $total_reward")
            break
        end

        a_blue, tree_blue = solved_mdp(game.pred_mdp, solver, s)
        paths_b = get_trajectories(game.pred_mdp,tree_blue, 100, 50)

        if !done_querying
            mdp_alt = genBetaMDP(game.pred_mdp, game.reward_variance*2)
            a_orange, tree_orange = solved_mdp(mdp_alt, solver, s)
            paths_o = get_trajectories(mdp_alt,tree_orange, 100, 50)

            choice, confidence, done_querying = query(teacher, game, tree_orange, tree_blue, s, deltaR, mdp_alt, alpha)
            # confidence = 10
            @show choice
            @show confidence
            @show done_querying

            if done_querying; confident_point = i end

            if show_true
                display(render(true_rewards, s, paths_b, paths_o))
            else
                display(render(reward_grid(game.pred_mdp), s, paths_b, paths_o))
            end

            if choice == "b"
                a = a_blue
                update_rewards!(game, game.pred_mdp, confidence)
            else
                # a = a_orange
                update_rewards!(game, mdp_alt, confidence)
                a, _ = solved_mdp(game.pred_mdp, solver, s)
            end
        else
            a = a_blue

            if show_true
                display(render(true_rewards, s, paths_b))
            else
                display(render(reward_grid(game.pred_mdp), s, paths_b))
            end
        end
        @show a
        readline()
        s,r = @gen(:sp,:r)(game.true_mdp,s,a)
        total_reward += r*γ^(i-1)
    end
    if confident_point == nothing; confident_point = i end
    return total_reward, confident_point, i
end

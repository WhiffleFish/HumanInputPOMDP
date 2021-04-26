function get_trajectory(mdp::SimpleGridWorld, tree::MCTS.MCTSTree{SVector{2, Int64}, Symbol}, d::Int)::Vector{SVector{2, Int64}}
    root = first(tree.s_labels)
    state_hist = SVector{2,Int64}[root]
    s_idx = 1
    for i in 1:d
        s = tree.s_labels[s_idx]
        if length(tree.child_ids[s_idx]) == 0 || isterminal(mdp,s)
            break
        end
        q_max = -Inf
        a_idx_max = first(tree.child_ids[s_idx])
        for a_idx in tree.child_ids[s_idx]
            q = tree.q[a_idx]
            if q > q_max
                q_max = q
                a_idx_max = a_idx
            end
        end
        a = tree.a_labels[a_idx_max]

        sp = @gen(:sp)(mdp,s,a)
        if haskey(tree.state_map,sp)
            s_idx = tree.state_map[sp]
            push!(state_hist, sp)
        else
            break
        end
    end
    return state_hist
end

function get_trajectories(mdp::SimpleGridWorld, tree::MCTS.MCTSTree{SVector{2, Int64}, Symbol}, N::Int, d::Int)::Vector{Vector{SVector{2, Int64}}}
    return [get_trajectory(mdp, tree, d) for _ in 1:N]
end

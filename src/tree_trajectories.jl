using POMDPs
using MCTS
using StaticArrays
using Distributions
using POMDPModels
using Random

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

function genMDP(mdp::SimpleGridWorld = SimpleGridWorld(), std::Float64=10.0)::SimpleGridWorld
    new_mdp = SimpleGridWorld(rewards=copy(mdp.rewards),tprob = mdp.tprob)
    for (k,v) in mdp.rewards
        new_mdp.rewards[k] = v + randn()*std
    end
    return new_mdp
end

function genPermMDP(mdp::SimpleGridWorld, std::Float64=10.0)::SimpleGridWorld
    new_mdp = SimpleGridWorld(rewards=copy(mdp.rewards))
    r = collect(values(mdp.rewards))
    r = shuffle(r).*rand([-1,1], length(r))
    for (i,k) in enumerate(keys(mdp.rewards))
        new_mdp.rewards[k] = clamp(rand(Normal(r[i],std)),-15.0, 15.0)
    end
    return new_mdp
end

function genVarMDP(belief::Array{Array{Float64,1},1}, reward_ranges::Array{Tuple{Float64, Float64}, 1}, std::Float64=10.0)
    new_mdp = SimpleGridWorld()
    reward = mean(belief)
    i = 1
    for (k,v) in new_mdp.rewards
        new_mdp.rewards[k] = (reward[i]*15.0 + sum(reward_ranges[i])/2.0) + randn()*std
        reward[i] = new_mdp.rewards[k]
        i += 1
    end
    return new_mdp, reward
end

function genMDP(belief::Array{Array{Float64,1},1}, reward_ranges::Array{Tuple{Float64, Float64}, 1})
    new_mdp = SimpleGridWorld()
    reward = rand(belief)
    i = 1
    for (k,v) in new_mdp.rewards
        new_mdp.rewards[k] = reward[i]*15 + sum(reward_ranges[i])/2.0
        i += 1
    end
    return new_mdp, reward
end

function meanMDP(belief::Array{Array{Float64,1},1}, reward_ranges::Array{Tuple{Float64, Float64}, 1})
    new_mdp = SimpleGridWorld()
    reward = mean(belief)
    i = 1
    @show reward, new_mdp.rewards
    for (k,v) in new_mdp.rewards
        new_mdp.rewards[k] = reward[i]*15.0 + sum(reward_ranges[i])/2.0
        i += 1
    end
    return new_mdp, reward
end

function genBetaMDP(belief::Array{Array{Float64,1},1}, reward_ranges::Array{Tuple{Float64, Float64}, 1}, std::Float64=10.0)
    new_mdp = SimpleGridWorld()
    reward = mean(belief)
    i = 1
    for (k,v) in new_mdp.rewards
        new_mdp.rewards[k] = (reward[i]*15.0 + sum(reward_ranges[i])/2.0) + rand_from_beta(std)
        reward[i] = new_mdp.rewards[k]
        i += 1
    end
    return new_mdp, reward
end

function genBetaMDP(mdp::SimpleGridWorld, std::Float64=10.0)::SimpleGridWorld
    new_mdp = SimpleGridWorld(rewards=copy(mdp.rewards))
    r = collect(values(mdp.rewards))
    shuffle!(r)
    for (i,k) in enumerate(keys(mdp.rewards))
        new_mdp.rewards[k] = r[i] + rand_from_beta(std)
    end
    return new_mdp
end

function initMDP(mdp::SimpleGridWorld)
    d = Dict(k=>(rand()-0.75)*5 for k in keys(mdp.rewards))
    return SimpleGridWorld(rewards=d, tprob=mdp.tprob)
end

function genBetafromPostMDP(belief::Array{Array{Float64,1},1})
    new_mdp = SimpleGridWorld()
    index = floor(Int, rand((Beta(0.5,0.5)))*length(belief))
    return belief[index]
end

function solved_mdp(mdp::SimpleGridWorld, solver::MCTSSolver, s0::SVector{2, Int})
    planner = solve(solver, mdp)
    a, info = action_info(planner, s0)
    return a,info[:tree]
end

function rand_from_beta(std::Float64)
    temp = rand(Beta(0.5,0.5))*2 - 1
    return temp*std
end

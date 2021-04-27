using POMDPs
using StaticArrays
using Parameters
using Random
using Distributions
# using POMDPModelTools
# using DiscreteValueIteration
# using POMDPSimulators
# using POMDPPolicies: FunctionPolicy
using LinearAlgebra
using Statistics
using MCTS

const GWPos = SVector{2,Int}
const DEFAULT_SIZE = (100,100)
const RD = 30

@with_kw struct RewardBelief
    size::Tuple{Int, Int} = DEFAULT_SIZE
    b::Dict{GWPos, Any} = Dict(GWPos(x,y) => Distributions.Uniform(-10.0,10.0) for x in RD:RD:size[1]-RD, y in RD:RD:size[2]-RD)
end

struct RewardCollection
    b::Array{Float64, 1}
end

Base.rand(rng::AbstractRNG, b::RewardCollection) =rand(b.b)
Statistics.mean(b::RewardCollection) = mean(b.b)

my_compare(x, y) = sum(x) < sum(y)

function sample_reward(b)
    r = Array{Float64, 1}(undef, length(b.b))

    rewards = Dict{GWPos, Float64}()
    i = 1
    for (key, value) in b.b
        rewards[key] = rand(value)
        r[i] = rewards[key]/10.0
        i += 1
    end
    return r
end

function convert_dict_to_vector(dict)
    r = Array{Float64, 1}(undef, length(b.b))
    i = 1
    for (key, value) in dict
        r[i] = value
    end
end


function logp(i, w, phi_As, phi_Bs, prefs, delta, alphas)
    # @show phi_As
    # @show phi_Bs
    # @show prefs
    # @show alphas
    phi_A = phi_As[i]
    phi_B = phi_Bs[i]
    a = prefs[i]
    alpha = alphas[i]

    if a != 0
        if a < 0
            # psi = phi_A - phi_B
            psi = phi_B
            psi2 = phi_A
        else
            # psi = phi_B - phi_A
            psi = phi_A
            psi2 = phi_B
        end
        # return log(1/ ( 1 + exp(delta + alpha*(dot(psi, w)))))
        # return log(1/ (1 + exp(norm(psi - w) - norm(psi2 - w))))
        # @show psi2, psi
        # @show norm(psi2 - w), norm(psi-w)
        # @show norm(psi2 - w)/(norm(psi-w) + norm(psi2-w))
        return log(norm(psi2 - w)/(norm(psi-w) + norm(psi2-w)))
    else
        psi = phi_B - phi_A
        # return log((exp(2*delta)-1)/( 1 + exp(delta + dot(psi, w)) + exp(delta - dot(psi, w)) + exp(2*delta)))
        return (0)
    end
end

function logp(w, phi_A, phi_B, a, delta, alpha)
    if a != 0
        if a < 0
            psi = phi_A - phi_B
            # psi = phi_B
            # psi2 = phi_A
        else
            psi = phi_B - phi_A
            # psi = phi_A
            # psi2 = phi_B
        end
        # @show 1/ (1 + exp(delta + alpha*(dot(psi, w))))
        return log(1/ (1 + exp(delta + alpha*(dot(psi, w)))))
        # return log(1/ (1 + exp(norm(psi - w) - norm(psi2 - w))))
        #return log(alpha*(-2+sign(dot(psi,w))))
    else
        psi = phi_B - phi_A
        return log((exp(2*delta)-1)/( 1 + exp(delta + dot(psi, w)) + exp(delta - dot(psi, w)) + exp(2*delta)))
    end
end

function compute_log_posterior_total(w, phi_As, phi_Bs, prefs, delta, alphas)
    return sum([logp(i, w, phi_As, phi_Bs, prefs, delta, alphas) for i in 1:length(prefs)])
end

function compute_log_posterior(w, phi_As, phi_Bs, prefs, delta, alpha)
    return sum([logp(w, phi_As, phi_Bs, prefs, delta, alpha)])
end
function mcmc_reward_step(rewards, step_size::Float64, r_min::Float64, r_max::Float64)
    new_rewards = deepcopy(rewards)
    # i = rand(1:length(new_rewards))
    # new_rewards[i] += sign((rand()*2)-1)*rand()*step_size
    # new_rewards[i] = clamp(new_rewards[i], r_min, r_max)
    # if new_rewards[i] == rewards[i]
    #     new_rewards[i] -= rand()*step_size
    # end
    for (i, val) in enumerate(new_rewards)
        new_rewards[i] += sign((rand()*2)-1)*step_size
        new_rewards[i] = clamp(new_rewards[i], r_min, r_max)
        if new_rewards == rewards
            new_rewards[i] -= rand()*step_size
        end
    end
    return new_rewards
end

function policy_walk(prior, phi_As, phi_Bs, prefs, alphas, iterations=100050, burn_in = 10000)
    delta = 1.0
    rewards = prior
    reward_samples = (Array{Float64, 1})[]
    post_old = nothing
    num_change = 0
    step_size = 0.2
    r_min = -1.0
    r_max = 1.0
    for i in 1:iterations
        proposed_rewards = mcmc_reward_step(rewards, step_size, r_min, r_max)

        if post_old == nothing
            post_old = compute_log_posterior_total(proposed_rewards, phi_As, phi_Bs, prefs, delta, alphas)
        end

        post_new =  compute_log_posterior_total(proposed_rewards, phi_As, phi_Bs, prefs, delta, alphas)
        # @show exp(post_new)
        # if exp(post_new) > 0.7
            # return [proposed_rewards]
        # end
        # if exp(post_new) < 0.3
        #     println("LESS")
        # end
        fraction = exp(post_new - post_old)
        # @show fraction
        # @show fraction, exp(post_old), exp(post_new), rewards, proposed_rewards
        if (rand() < min(1, fraction))
            num_change += 1
            if i > burn_in
                reward_samples = push!(reward_samples,proposed_rewards)
            end
            post_old = post_new
            rewards = copy(proposed_rewards)
        else
            if i > burn_in
                reward_samples = push!(reward_samples,rewards)
            end
        end
    end
    R = mean(reward_samples)
    @show R
    return reward_samples
end


# function policy_walk(prior, phi_A, phi_B, pref, alpha, iterations=100050, burn_in = 10000)
#     delta = 1.0
#     rewards = prior
#     reward_samples = (Array{Float64, 1})[]
#     post_old = nothing
#     num_change = 0
#     step_size = 0.5
#     r_min = -1.0
#     r_max = 1.0
#     for i in 1:iterations
#         proposed_rewards = mcmc_reward_step(rewards, step_size, r_min, r_max)

#         if post_old == nothing
#             post_old = compute_log_posterior(proposed_rewards, phi_A, phi_B, pref, delta, alpha)
#         end

#         post_new =  compute_log_posterior(proposed_rewards, phi_A, phi_B, pref, delta, alpha)
#         # @show exp(post_new)
#         # if (exp(post_new) > 0.75)
#         #     @show post_new, proposed_rewards
#         #     break
#         # end
#         fraction = exp(post_new - post_old)/2
#         # @show fraction
#         # @show fraction, exp(post_old), exp(post_new), rewards, proposed_rewards
#         if (rand() < min(1, fraction))
#             num_change += 1
#             if i > burn_in
#                 reward_samples = push!(reward_samples,proposed_rewards)
#             end
#             post_old = post_new
#             rewards = copy(proposed_rewards)
#         else
#             if i > burn_in
#                 reward_samples = push!(reward_samples,rewards)
#             end
#         end
#     end
#     R = mean(reward_samples)
#     @show R
#     return reward_samples[end]
# end

# prior = rand(4)
# for i in 1:10000
#     reward_samples = policy_walk(prior, phi_A, phi_B, pref, confidence)
#     prior = mean(reward_samples)
# end

# mean(reward_samples[:,1])

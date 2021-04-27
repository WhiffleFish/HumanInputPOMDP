using POMDPs
using StaticArrays
using Parameters
using Random
using Distributions
using POMDPModelTools
using DiscreteValueIteration
using POMDPSimulators
using POMDPPolicies: FunctionPolicy
include("birl.jl")


const GWPos = SVector{2,Int}
const DEFAULT_SIZE = (100,100)
const RD = 20

struct MDPPlus{M<:MDP} 
    m::M
    b::Updater
end

@with_kw struct UncertainSimpleGridWorld <: MDP{GWPos, Symbol}
    size::Tuple{Int, Int}           = (10,10)
    rewards::Dict{GWPos, Tuple{Float64, Float64}}   = Dict(GWPos(4,3)=>(-10, 10), GWPos(4,6)=>(-10, 10), GWPos(9,3)=>(-10,10), GWPos(8,8)=>(-10,10))
    true_rewards::Dict{GWPos, Float64} = Dict(GWPos(4,3)=>-10.0, GWPos(4,6)=>-5.0, GWPos(9,3)=>10.0, GWPos(8,8)=>-3.0)
    terminate_from::Set{GWPos}     = Set(keys(rewards))
    tprob::Float64                  = 0.7
    discount::Float64               = 0.95
    reward_belief::RewardCollection = nothing
end

struct SimpleGridWorld <: MDP{GWPos, Symbol}
    size::Tuple{Int, Int}           
    rewards::Dict{GWPos, Float64}   
    terminate_from::Set{GWPos}     
    tprob::Float64                  
    discount::Float64               
end

function SimpleGridWorld(Dict)
    size = (10,10)
    rewards = Dict
    terminate_from = Set(keys(rewards))
    tprob = 0.8
    discount = 0.95

    return SimpleGridWorld(size, rewards, terminate_from, tprob, discount)
end

function SimpleGridWorld()
    size = (10,10)
    rewards = Dict(GWPos(4,3)=>-10.0, GWPos(4,6)=>-5.0, GWPos(9,3)=>10.0, GWPos(8,8)=>-3.0)
    terminate_from = Set(keys(rewards))
    tprob = 0.8
    discount = 0.95
    return SimpleGridWorld(size, rewards, terminate_from, tprob, discount)
end


# States

function POMDPs.states(mdp::SimpleGridWorld)
    ss = vec(GWPos[GWPos(x, y) for x in 1:mdp.size[1], y in 1:mdp.size[2]])
    push!(ss, GWPos(-1,-1))
    return ss
end

function POMDPs.stateindex(mdp::SimpleGridWorld, s::AbstractVector{Int})
    if all(s.>0)
        return LinearIndices(mdp.size)[s...]
    else
        return prod(mdp.size) + 1 # TODO: Change
    end
end

struct GWUniform
    size::Tuple{Int, Int}
end
Base.rand(rng::AbstractRNG, d::GWUniform) = GWPos(rand(rng, 1:d.size[1]), rand(rng, 1:d.size[2]))
function POMDPs.pdf(d::GWUniform, s::GWPos)
    if all(1 .<= s[1] .<= d.size)
        return 1/prod(d.size)
    else
        return 0.0
    end
end
POMDPs.support(d::GWUniform) = (GWPos(x, y) for x in 1:d.size[1], y in 1:d.size[2])

POMDPs.initialstate(mdp::SimpleGridWorld) = GWUniform(mdp.size)

# Actions

POMDPs.actions(mdp::SimpleGridWorld) = (:up, :down, :left, :right)
Base.rand(rng::AbstractRNG, t::NTuple{L,Symbol}) where L = t[rand(rng, 1:length(t))] # don't know why this doesn't work out of the box


const dir = Dict(:up=>GWPos(0,1), :down=>GWPos(0,-1), :left=>GWPos(-1,0), :right=>GWPos(1,0))
const aind = Dict(:up=>1, :down=>2, :left=>3, :right=>4)

POMDPs.actionindex(mdp::SimpleGridWorld, a::Symbol) = aind[a]


# Transitions

POMDPs.isterminal(m::SimpleGridWorld, s::AbstractVector{Int}) = any(s.<0)

function POMDPs.transition(mdp::SimpleGridWorld, s::AbstractVector{Int}, a::Symbol)
    if s in mdp.terminate_from || isterminal(mdp, s)
        return Deterministic(GWPos(-1,-1))
    end

    destinations = MVector{length(actions(mdp))+1, GWPos}(undef)
    destinations[1] = s

    probs = @MVector(zeros(length(actions(mdp))+1))
    for (i, act) in enumerate(actions(mdp))
        if act == a
            prob = mdp.tprob # probability of transitioning to the desired cell
        else
            prob = (1.0 - mdp.tprob)/(length(actions(mdp)) - 1) # probability of transitioning to another cell
        end

        dest = s + dir[act]
        destinations[i+1] = dest

        if !inbounds(mdp, dest) # hit an edge and come back
            probs[1] += prob
            destinations[i+1] = GWPos(-1, -1) # dest was out of bounds - this will have probability zero, but it should be a valid state
        else
            probs[i+1] += prob
        end
    end

    return SparseCat(destinations, probs)
end

function inbounds(m::SimpleGridWorld, s::AbstractVector{Int})
    return 1 <= s[1] <= m.size[1] && 1 <= s[2] <= m.size[2]
end

# Rewards

POMDPs.reward(mdp::SimpleGridWorld, s::AbstractVector{Int}) = get(mdp.rewards, s, 0.0)
POMDPs.reward(mdp::SimpleGridWorld, s::AbstractVector{Int}, a::Symbol) = reward(mdp, s)


# discount

POMDPs.discount(mdp::SimpleGridWorld) = mdp.discount

# Conversion
function POMDPs.convert_a(::Type{V}, a::Symbol, m::SimpleGridWorld) where {V<:AbstractArray}
    convert(V, [aind[a]])
end
function POMDPs.convert_a(::Type{Symbol}, vec::V, m::SimpleGridWorld) where {V<:AbstractArray}
    actions(m)[convert(Int, first(vec))]
end

# deprecated in POMDPs v0.9
POMDPs.initialstate_distribution(mdp::SimpleGridWorld) = GWUniform(mdp.size)
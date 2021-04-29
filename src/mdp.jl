using POMDPs
using StaticArrays
using Parameters
using Random
using Distributions
using POMDPModels

const GWPos = SVector{2,Int}

function SimpleGridWorld(Dict)
    rewards = Dict
    tprob = 0.8
    return SimpleGridWorld(rewards=rewards, tprob=tprob)
end


function OneGoodSimpleGridWorld()
    rewards = Dict(GWPos(2,2)=> -6.0, GWPos(2,9)=> -5.0, GWPos(9,2)=>10.0, GWPos(9,9)=>-7.0)
    tprob = 0.8
    return SimpleGridWorld(rewards=rewards, tprob=tprob)
end

function TwoGoodSimpleGridWorld()
    rewards = Dict(GWPos(2,2)=>-10.0, GWPos(2,9)=> 5.0, GWPos(9,2)=>10.0, GWPos(9,9)=> -5.0)
    tprob = 0.8
    return SimpleGridWorld(rewards=rewards, tprob=tprob)
end

function RandomSimpleGridWorld()
    rewards = Dict(GWPos(4,3)=>((rand()*2)-1)*15, GWPos(4,6)=>((rand()*2)-1)*15, GWPos(9,3)=>((rand()*2)-1)*15, GWPos(8,8)=>((rand()*2)-1)*15)
    tprob = 0.8
    return SimpleGridWorld(rewards=rewards, tprob=tprob)
end

function TwoTerminalSimpleGridWorld()
    rewards = Dict(GWPos(8,7)=>5.0, GWPos(8,4)=>1.0)
    tprob = 0.8
    return SimpleGridWorld(rewards=rewards, tprob=tprob)
end
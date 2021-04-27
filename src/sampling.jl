using StatsBase

function weights(vecs, ref_vec)
    w = [sqrt(sum(abs2,ref_vec .- vec)) for vec in vecs]
    return StatsBase.Weights(w./sum(w))
end

StatsBase.sample(vecs::Vector{Vector{Float64}}, ref_vec::Vector{Float64}) = sample(vecs,weights(vecs,ref_vec))

sample([[1.,2.,3.],[4.,5.,6.]],[0.,0.,0.])

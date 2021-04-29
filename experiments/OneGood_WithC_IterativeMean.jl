include("../src/tree_trajectories.jl")
include("../src/display.jl")
include("../src/ChooseTheBetterMDP.jl")
include("../src/ScriptedTeacher.jl")
include("../src/mdp.jl")
using ProgressMeter
using JLD
mdp = OneGoodSimpleGridWorld()

rewards = []
cutoffs = []
n_iters = []
@showprogress for i = 1:50
    game = IMGame(true_mdp=mdp,reward_variance=5.0, s0=SA[6,6], max_steps=50)
    teacher = ScriptedTeacher(1.0, game)
    (r,cutoff,n_iter) = play(game, teacher; show_true=false, deltaR = 0.3, alpha=0.1)
    push!(rewards, r)
    push!(cutoffs, cutoff)
    push!(n_iters, n_iter)
end

histogram(rewards,label="", normalize=true, bins=10)
xlabel!("Rewards")
ylabel!("Frequency")
mean(rewards)

results = Dict(:rewards=>rewards, :cutoffs=>cutoffs, :n_iters=>n_iters)
# save("../results/OneGood_WithC_IterativeMean.jld", "results", results)
save("OneGood_NoC_IterativeMean.jld", "results", results)

maximum(rewards)

histogram(cutoffs./n_iters, bins=10, label="", normalize=true)

mean(cutoffs./n_iters)

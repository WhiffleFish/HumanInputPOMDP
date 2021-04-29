include("../src/tree_trajectories.jl")
include("../src/display.jl")
include("../src/ChooseTheBetterMDP.jl")
include("../src/ScriptedTeacher.jl")
include("../src/mdp.jl")

mdp = SimpleGridWorld(tprob = 0.70)
game = IMGame(true_mdp=mdp,reward_variance=10.0, s0=SA[6,6], max_steps=100)
teacher = ScriptedTeacher(1.0, game)

play(game, teacher; show_true=false, deltaR = 0.01, alpha=0.3)
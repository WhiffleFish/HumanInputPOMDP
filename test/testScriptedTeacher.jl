include("../src/tree_trajectories.jl")
include("../src/display.jl")
include("../src/ChooseTheBetterMDP.jl")
include("../src/ScriptedTeacher.jl")
include("../src/mdp.jl")

mdp = OneGoodSimpleGridWorld()
game = IMGame(true_mdp=mdp,reward_variance=5.0, s0=SA[6,6], max_steps=100)
teacher = ScriptedTeacher(1.0, game)

<<<<<<< HEAD
play(game, teacher; show_true=false, deltaR = 0.1, alpha=0.3)
=======
play(game, teacher; show_true=false, deltaR = 0.01, alpha=0.3)
>>>>>>> 5edd36511ace415d94850ef688b21dbc4a2ec2e4

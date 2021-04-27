include("../src/display.jl")

mdp = SimpleGridWorld()
rewards = reward_grid(mdp)

render(rewards, show_vals=true)

render(rewards,[2,3], show_vals=true)

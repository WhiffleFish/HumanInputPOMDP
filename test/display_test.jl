include("../src/display.jl")

rewards = reward_grid(mdp)

render(rewards, show_vals=true)

render(rewards,[2,3], show_vals=true)

path = [SA[2,2],SA[3,2], SA[3,3]]
render(rewards, [2,2], path)

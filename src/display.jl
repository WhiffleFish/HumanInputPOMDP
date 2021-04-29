using Plots
using Luxor
using StaticArrays

function render(grid::Matrix{Float64}; show_vals::Bool=true)
    s = [500,500]
    offset = Point(s/2...)
    Drawing(s...)
    background("black")

    grid = transpose(grid)

    tiles = Tiler(s..., 10, 10, margin=0)
    g = [:red, :white, :green]
    colors = cgrad(g)
    grid_max = max(grid...)
    grid_min = min(grid...)
    for (pos, n) in tiles
        box(pos+offset, tiles.tilewidth*0.98, tiles.tileheight*0.98, :clip)
        intensity = (grid[n]-grid_min)/(grid_max-grid_min)
        background(colors[intensity])

        sethue("black")
        if show_vals
            textcentred(string(round(grid[n], digits=2)), pos+offset)
        end
        clipreset()
    end
    finish()
    preview()
end

function render(grid::Matrix{Float64}, pos::Vector{Int}; show_vals::Bool=true)
    s = [500,500]
    ds = 500/10
    offset = Point(s/2...)
    Drawing(s...)
    background("black")

    grid = transpose(grid)

    pos_n = (pos[1]-1)*10 + pos[2]

    tiles = Tiler(s..., 10, 10, margin=0)
    g = [:red, :white, :green]
    colors = cgrad(g)
    grid_max = max(grid...)
    grid_min = min(grid...)
    for (pos, n) in tiles
        box(pos+offset, tiles.tilewidth*0.98, tiles.tileheight*0.98, :clip)
        intensity = (grid[n]-grid_min)/(grid_max-grid_min)
        background(colors[intensity])

        sethue("black")
        if show_vals
            textcentred(string(round(grid[n], digits=2)), pos+offset)
        end

        if n == pos_n
            sethue("purple")
            setmode("darken")
            circle(pos+offset,20,:fill)
            setmode("over")
        end

        clipreset()

    end

    finish()
    preview()
end

function render(grid::Matrix{Float64}, pos::Union{Vector{Int},SVector{2,Int}}, paths::Vector{Vector{SVector{2,Int64}}}; show_vals::Bool=true)
    s = [500,500]
    term_state = SA[-1,-1]
    ds = 500/10
    offset = Point(s/2...)
    Drawing(s...)
    background("black")

    grid = transpose(grid)

    pos_n = (pos[1]-1)*10 + pos[2]

    tiles = Tiler(s..., 10, 10, margin=0)
    g = [:red, :white, :green]
    colors = cgrad(g)
    grid_max = max(grid...)
    grid_min = min(grid...)
    for (pos, n) in tiles
        box(pos+offset, tiles.tilewidth*0.98, tiles.tileheight*0.98, :clip)

        if grid[n] == 0.0
            background("white")
        else
            intensity = (grid[n]-grid_min)/(grid_max-grid_min)
            background(colors[intensity])
        end

        sethue("black")
        if show_vals
            textcentred(string(round(grid[n], digits=2)), pos+offset)
        end

        if n == pos_n
            sethue("purple")
            setmode("darken")
            circle(pos+offset,20,:fill)
            setmode("over")
        end

        clipreset()

    end
    setline(20)
    op = 1/(0.5*length(paths))
    setopacity(min(1, op))
    sethue("blue")
    for path in paths
        filter!(pt -> pt != term_state, path)
        pts = map(x->noisypos2px(x,ds), path)
        poly(pts, :stroke)
    end

    finish()
    preview()
end

function render(grid::Matrix{Float64}, pos::Union{Vector{Int},SVector{2,Int}}, path1::Vector{Vector{SVector{2,Int64}}}, path2::Vector{Vector{SVector{2,Int64}}}, path3::Vector{Vector{SVector{2,Int64}}}; show_vals::Bool=true)
    s = [500,500]
    term_state = SA[-1,-1]
    ds = 500/10
    offset = Point(s/2...)
    Drawing(s...)
    background("black")

    grid = transpose(grid)

    pos_n = (pos[1]-1)*10 + pos[2]

    tiles = Tiler(s..., 10, 10, margin=0)
    g = [:red, :white, :green]
    colors = cgrad(g)
    grid_max = max(grid...)
    grid_min = min(grid...)
    for (pos, n) in tiles
        box(pos+offset, tiles.tilewidth*0.98, tiles.tileheight*0.98, :clip)

        if grid[n] == 0.0
            background("white")
        else
            intensity = (grid[n]-grid_min)/(grid_max-grid_min)
            background(colors[intensity])
        end

        sethue("black")
        if show_vals
            textcentred(string(round(grid[n], digits=2)), pos+offset)
        end

        if n == pos_n
            sethue("purple")
            setmode("darken")
            circle(pos+offset,20,:fill)
            setmode("over")
        end

        clipreset()

    end
    setline(20)
    op = 1/(0.5*length(path1))
    setopacity(min(1, op))
    sethue("blue")
    for path in path1
        filter!(pt -> pt != term_state, path)
        pts = map(x->noisypos2px(x,ds), path)
        poly(pts, :stroke)
    end

    sethue("orange")
    for path in path2
        filter!(pt -> pt != term_state, path)
        pts = map(x->noisypos2px(x,ds), path)
        poly(pts, :stroke)
    end

    finish()
    preview()
end

function reward_grid(gw::SimpleGridWorld)::Matrix{Float64}
    r = zeros(Float64, 10, 10)
    for (k,v) in gw.rewards
        r[k...] = v
    end
    return r
end


"""
Input: Grid world position
Output: pixel center of corresponding board block
"""
function pos2px(pos, dx::Float64)::Point
    return Point(((reverse(pos) .- 1)*dx .+ dx/2)...)
end

function noisypos2px(pos, dx::Float64, intensity = 10)::Point
    return Point(((reverse(pos) .- 1)*dx .+ dx/2 + rand(-intensity:intensity,2))...)
end

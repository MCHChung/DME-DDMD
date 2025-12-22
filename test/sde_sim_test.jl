using DrWatson, Plots, StatsBase, Random
@quickactivate "TMI&MD"

include(srcdir("get_ens_data.jl"))

rng = Random.default_rng()
Random.seed!(rng, 0)

β = 0.4
a = 0.1
μ = 0.5
ω = 0.1
σ =2e-1 #2e-1
a,b,c= 1,0,-1
u0 = 0.0
f(u,p,t) = @. β*(μ*(1-a*sin(2*π*ω*t))-u)
#f(u,p,t) = @. a*u + b*u^2 + c*u^3
g(u,p,t) = σ
dt = 0.01
tspan = (0.0, 7.0) 

X, ts = Get1DEnsData(f,g,u0,dt,tspan, rng; num_traj=3000, noise_mag=0.1)

xs = -1.5:0.01:1.5
Xd = EstDist(X,xs)
Xd = Xd ./ sum(Xd, dims=1)
#print(size(Xd))
#plot(ts, mean(X, dims=1)[:], ribbon = std(X, dims=1)[:], color=:black, lw=3, label=false)
heatmap(ts, xs, Xd)

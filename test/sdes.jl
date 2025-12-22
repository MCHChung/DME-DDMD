using DrWatson, Random
@quickactivate "TMI&MD"

# Here you may include files from the source directory
include(srcdir("tmi.jl"))
include(srcdir("build_weight_mats.jl"))
include(srcdir("vis_results.jl"))
include(srcdir("get_ens_data.jl"))
include(srcdir("pred_lats.jl"))

# generate data 
# Params
β = 0.4
μ = 1.
a = 0.1
ω = 0.1
σ =2e-1 #2e-1
a,b,c=1,0,-1
u0 = 0.0
f(u,p,t) = @. β*(μ*(1-a*sin(2*π*ω*t))-u)
#f(u,p,t) = @. a*u + b*u^2 + c*u^3
g(u,p,t) = σ
dt = 0.01
tspan = (0.0, 7.0) 

rng = Random.default_rng()
Random.seed!(rng, 0)

X, ts = Get1DEnsData(f,g,u0,dt,tspan, rng; num_traj=3000, noise_mag=0.1)

# estimate distribution using KDE
xb = 1.5
xs = -xb:0.01:xb
Xd = EstDist(X,xs)
Xd = Xd ./ sum(Xd, dims=1)

wind = 11
W = BuildWeightMat(ts, wind, 0)
DW = BuildWeightMat(ts, wind, 1)

K = 2 # latent space dim 
#N = [1 ; 2 ; 3 ; 4 ; 5] # polynomial orders 
N = [1 0 ; 0 1 ; 2 0 ; 1 1 ; 0 2]#; 3 0; 2 1 ; 1 2 ; 0 3]
θy = [xs xs.^2 xs.^3 xs.^4 xs.^5] # Model terms for Y = θy*Ξy

@time (z, Y, q), (Ξz, Ξy), errs = tmi(Xd, K, W, DW, N, θy, maxiters=150000, η=1e-3, tol=1e-4, λz=0, λy=1e1, 
use_E=false, cz=1e-6, cy=1e1, use_rand_init=true, sparse_iter=50000, track_lats=false, ptol=0.7, ρ=0, 
νz_sm=0, νz=0, νy=0, print_err=false)

p_lats, p_pred = PlotResults((ts,xs,Xd), (z,Y,q))  

zpred = PredTraj(N, Ξz, z[1,:], ts)

display(p_lats)
display(p_pred)

display(VisModel(Ξz, "Ξz"))
display(VisModel(Ξy, "Ξy"))
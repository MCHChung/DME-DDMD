using DrWatson, Random, CSV, DataFrames
@quickactivate "TMI&MD"

# load tmi functions 
include(srcdir("tmi.jl"))
include(srcdir("build_weight_mats.jl"))
include(srcdir("vis_results.jl"))

# data 
pend_data = wload(datadir("sims", "pend_data.jld2"))
ts = pend_data["ts"]
X = pend_data["X"]

#=
Ks = collect(1:5)
qs_cp, klds_cp = latent_dim_scan(X, Ks, normalize=true ,tol =1e-5, η=1e-3, maxiters=100000, use_E=true)
scatter(klds_cp)
=# 

# warm start 
#= 
K = 1
(E, z , Y, q) , _ = tmi(X, K; 
    normalize=true,
    maxiters=150000, # max number of iterations to update params 
    tol=1e-5, # termination condition if total L2 error of derivatives across iter < tol
    η=1e-3, # learning rate 
    use_rand_init=false,
    use_E=true, 
    track_lats = false,
    print_err = false
)
=# 

wind = 21
W = BuildWeightMat(ts, wind, 0)
DW = BuildWeightMat(ts, wind, 2)
#N = [1 0 ; 0 1; 2 0; 1 1; 0 2]
N = [1; 2; 3; 4; 5]
# learn dynamics w/ warm start  
(Ep, zp , Yp, qp) , Ξz, _ = tmi(X, K, W, DW, N; 
    normalize=true,
    maxiters=10000, # max number of iterations to update params 
    tol=1e-5, # termination condition if total L2 error of derivatives across iter < tol
    η=1e-6, # learning rate 
    use_rand_init=false,
    use_E=true, 
    z = z,
    Y = Y, 
    E = E, 
    sparse_iter = 5000, 
    λz=1e2, # how strongly to enforce dzdt = θ(z)*Ξz
    cz = 1e-1, # adds small amount of noise to prevent overfitting, essentially ,
    track_lats = false,
    print_err = false
)


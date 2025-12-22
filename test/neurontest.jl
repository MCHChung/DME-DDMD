using DrWatson, Random, CSV, DataFrames
@quickactivate "TMI&MD"

# load tmi functions 
include(srcdir("tmi.jl"))
include(srcdir("build_weight_mats.jl"))
include(srcdir("vis_results.jl"))
include(srcdir("pred_lats.jl"))

# data 
data = matread(datadir("exp_raw", "ChurchlandRotNeuronData.mat"))["Data"]
ts = vec(data["times"][1])
ts .+= abs(min(ts...))
P = data["A"][1]'
P = P ./ sum(P,dims=1)

#=
Ks = collect(1:10)
qs_cp, klds_cp = latent_dim_scan(P, Ks, normalize=true ,tol =1e-5, η=1e-3, maxiters=150000, use_E=true)
scatter(klds_cp)
=# 

# warm start 
#=
K = 4
(E, z , Y, q) , _ = tmi(P, K; 
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
#=
# warm start 
K = 4
(z , Y, q) , _ = tmi(P, K; 
    normalize=true,
    maxiters=150000, # max number of iterations to update params 
    tol=1e-5, # termination condition if total L2 error of derivatives across iter < tol
    η=1e-3, # learning rate 
    use_rand_init=false,
    use_E=false, 
    track_lats = false,
    print_err = false
)
=# 

K = 4
wind = 11
W = BuildWeightMat(ts, wind, 0)
DW = BuildWeightMat(ts, wind, 1)

#DW, W = BuildDiscWeightMat(ts)
#N = [1 0 ; 0 1; 2 0; 1 1; 0 2]
N = [1 0 0 0 ; 0 1 0 0 ; 0 0 1 0 ; 0 0 0 1 ; 2 0 0 0; 1 1 0 0 ; 1 0 1 0; 1 0 0 1; 0 2 0 0; 0 1 1 0; 0 1 0 1;
    0 0 2 0; 0 0 1 1; 0 0 0 2]

# learn dynamics w/ warm start  
(Ep, zp , Yp, qp) , Ξz, _ = tmi(P, K, W, DW, N; 
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
    λz = 1e-1, # how strongly to enforce dzdt = θ(z)*Ξz
    cz = 1e-6, # adds small amount of noise to prevent overfitting, essentially ,
    track_lats = false,
    print_err = false
)

zpred = PredTraj(N, Ξz, z[1,:], ts)
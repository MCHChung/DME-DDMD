using DrWatson, Random, CSV, DataFrames
@quickactivate "TMI&MD"

# load tmi functions 
include(srcdir("tmi.jl"))
include(srcdir("build_weight_mats.jl"))
include(srcdir("vis_results.jl"))
include(srcdir("smooth.jl"))

# data 
cpdata = CSV.File(datadir("exp_raw", "hungarian+chickenpox+cases", "hungary_chickenpox.csv")) |> DataFrame
P = Matrix(Float64.(Array(cpdata[:, 2:end])'))
P = abs.(smooth_ode(P))
Pn = P ./ sum(P, dims=1) 

Ks = collect(1:5)
qs_cp, klds_cp = latent_dim_scan(Pn, Ks, normalize=true ,tol =1e-5, η=1e-3, maxiters=150000, use_E=true)
scatter(klds_cp)

#= 
K = 4
(E, z , Y, q) , _ = tmi(Pn, K; 
    normalize=true,
    maxiters=150000, # max number of iterations to update params 
    tol=1e-3, # termination condition if total L2 error of derivatives across iter < tol
    η=1e-3, # learning rate 
    use_rand_init=false,
    use_E=true, 
    sparse_iter = 50000, 
    λz=0., # how strongly to enforce dzdt = θ(z)*Ξz
    cz = 0., # adds small amount of noise to prevent overfitting, essentially ,
    track_lats = false,
    print_err = false
)
=# 


#=
wind = 21
ts = 0:size(P,2)-1
W = BuildWeightMat(ts, wind, 0)
DW = BuildWeightMat(ts, wind, 1)

Pn = P ./ sum(P, dims=1)
K = 2
N = [1 0 ; 0 1; 2 0; 1 1; 0 2]

# warm start 
(E, z , Y, q) , Ξz, _ = tmi(P, K, W, DW, N; 
    normalize=true,
    maxiters=150000, # max number of iterations to update params 
    tol=1e-3, # termination condition if total L2 error of derivatives across iter < tol
    η=1e-3, # learning rate 
    use_rand_init=false,
    use_E=true, 
    sparse_iter = 50000, 
    λz=0., # how strongly to enforce dzdt = θ(z)*Ξz
    cz = 0., # adds small amount of noise to prevent overfitting, essentially ,
    track_lats = false,
    print_err = false
)
=# 


function EstDist(data, xs, ys)
    X = zeros((length(xs),length(ys), size(data,3)))
    for t=1:size(data,3)
        # kde est of spatial dist at time t
        k = kde(data[:,:,t])
        X[:,:,t] = pdf(k, xs,ys)
    end

    return X
end
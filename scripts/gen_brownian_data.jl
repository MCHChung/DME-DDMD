using DrWatson, Random
@quickactivate "TMI&MD"

# Here you may include files from the source directory
include(srcdir("get_ens_data.jl"))

# ============= generate train data =============
# Params
rng = Random.default_rng()
Random.seed!(rng, 0)

u0 = [0., 0.] 
tspan = (0. , 10.)
dt = 0.01
ps = [1e-1, 1e-1]
trajs = 3000
sig_init = 0.1
xs = -1:0.05:1
ys = -1:0.05:1

# get data 
ts, X, Xd = BrownianData(u0, tspan, dt, ps, trajs, rng, sig_init=sig_init, xs=xs, ys=ys) ;

# create and save data 
brownian_data = @strdict ts xs ys X Xd u0
#wsave(datadir("sims", "brownian_data.jld2"), brownian_data)


# ============ generate test data ===============

#u0s = rand(rng, -0.5:0.1:0.5, 2, 10) # random ICs
ps_new = rand(rng, 0.005:0.001:0.3, 10)
ps_new = vcat(ps_new',ps_new')

Xs = []
Xds = []
for i=1:size(ps_new,2)
    #u0_new = u0s[:,i]
    ps_new_i = ps_new[:,i]
    _, X_new, Xd_new = BrownianData(u0, tspan, dt, ps_new_i, trajs, rng, sig_init=sig_init, xs=xs, ys=ys) ;
    push!(Xs, X_new)
    push!(Xds, Xd_new)
end

#brownian_test_data = @strdict ts xs Xs Xds u0s ps_new
brownian_test_data = @strdict ts xs Xs Xds ps_new u0
wsave(datadir("sims", "brownian_test_data.jld2"), brownian_test_data)
 
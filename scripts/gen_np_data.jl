using DrWatson, Random
@quickactivate "TMI&MD"

# Simulates NP system
include(srcdir("sim_np_data.jl"))

# ========== generate TRAIN data ==============
# simulation parameters
dt = 0.05 # time step
tspan = (0.,100.) # sim time
Kr = 1e-5 # strength of constant interaction kernel 
nmax = 70 # largest allowed cluster size, determines number of odes
M = 50000
sys = 2 # Brownian kernel 

# Run sims 
# Train 
# initial data 
u0 = zeros(nmax)
u0[1] = M # initial state : only monomers
#ts, Ns = SimAgg(nmax, u0, tspan, dt, Kr; i=sys)
#Ns_n = Ns ./ sum(Ns, dims=1)

#np_data = @strdict nmax ts Ns Ns_new nmax_new
#wsave(datadir("sims", "np_data2.jld2"), np_data)

# ========== generate TEST data ==============
rng = Random.default_rng()
Random.seed!(rng, 0)

Ns_new = []
nmaxs_new = []
for i=1:10
    println("i = $(i)")
    nmax_new = rand(rng, 80:2:100)
    u0_new = zeros(nmax_new)
    u0_new[1] = M 
    K = rand(rng, 0.3:0.1:1.2)*Kr
    _, N_new = SimAgg(nmax_new, u0_new, tspan, dt, K; i=sys)
    push!(Ns_new, N_new)
    push!(nmaxs_new, nmax_new)
end

np_test_data = @strdict ts Ns_new nmaxs_new
wsave(datadir("sims", "np_test_data.jld2"), np_test_data)
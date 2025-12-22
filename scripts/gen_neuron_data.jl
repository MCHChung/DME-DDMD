using DrWatson, Plots, DifferentialEquations

@quickactivate "TMI&MD"

include(srcdir("sim_spiking.jl"))
include(srcdir("smooth.jl"))

# ============ Linear Systems ===============

seed = 0
rng = Random.default_rng()
Random.seed!(rng, seed)

# generate test datasets 
As = []
push!(As, [0 rand(rng, 0.5:0.1:3) ; -rand(rng, 0.5:0.1:3) 0.]) # this is rotational

# generates linear system with decaying component 
for i=1:1
    while true
        A = randn(rng, 2,2)
        W, _ = eigen(A)
        if all(real(W) .<= -0.1) # make sure real part of eigen is negative 
            push!(As, A)
            break
        end
    end
end

# defining simulation of linear system 
function LinearData2d(A, u0, ts)
    prob = ODEProblem((u,p,t) -> vec(A*u), u0, (ts[1], ts[end]), [])
    soln = solve(prob, Tsit5(), saveat=ts)
    return Array(soln)
end

# define number of trials and simulation parameters 
trials = 15
dt = 0.01 
ts = 0:dt:800*dt
Xss_tr = []
Xss_test = []

# holder arrays 
Cs = []
bs = []
Pss = []
Pavgs_tr = []
Pavgs_test = []
Psms_tr = []
Psms_test = []
trind = 10
for i = 1:length(As)
    Xs = zeros(2, length(ts), trials)
    for j=1:trials 
        X2d = LinearData2d(As[i], rand(rng, -5:0.1:5,2), ts)
        Xs[:,:,j] = X2d
    end
    Xs_tr = Xs[:,:,1:trind]
    Xs_test = Xs[:,:,trind:end]

    push!(Xss_tr, Xs_tr)
    push!(Xss_test, Xs_test)

    # gen data for specific low-dim dynamical system over 100 trials 
    (C, b), Ps, (Pavg_tr, Pavg_test), (Psm_tr, Psm_test) = GenerateSpikingDataWithTrials(Xs, trind, dt; N=100, 
    trials=100, seed=seed, σ=1/(5*dt))
    
    # smooth 
    Psm_tr = copy(Pavg_tr)
    Psm_test = copy(Pavg_test)

    for i=1:2
        Psm_tr_i = Psm_tr[i]
        Psm_test_i = Psm_test[i]

        for j=1:trind
            Psm_tr[:,:,j] = smooth_ode(Psm_tr[:,:,j])
        end

        for j=1:trials-trind
            Psm_test[:,:,j] = smooth_ode(Psm_test[:,:,j])
        end
    end
    
    push!(Cs,C)
    push!(bs,b)
    push!(Pss, Ps)
    push!(Pavgs_tr, Pavg_tr)
    push!(Pavgs_test, Pavg_test)
    push!(Psms_tr, Psm_tr)
    push!(Psms_test, Psm_test)
end

# save 
neuron_data = @strdict ts Xss_tr Xss_test Cs bs Pss Pavgs_tr Pavgs_test Psms_tr Psms_test As
# wsave(datadir("sims", "neuron_data3.jld2"), neuron_data)

# uncomment below to look at the data of one of the systems 
#=
p1= plot_zs(permutedims(neuron_data["Xss_tr"][1], (2,1,3)))
p2 = plot_zs(permutedims(neuron_data["Xss_tr"][2], (2,1,3)))
display(p1)
display(p2)

p = heatmap(Psms_tr[1][:,:,1])
=# 
using KernelDensity , DifferentialEquations.StochasticDiffEq

# estimating distribution from kde
function EstDist(data, xs)
    X = zeros((length(xs), size(data,2)))
    for t=1:size(data,2)
        # kde est of spatial dist at time t
        k = kde(data[:, t])
        X[:, t] = pdf(k, xs)
    end

    return X
end

function EstDist(data, xs, ys)
    X = zeros((length(xs),length(ys), size(data,3)))
    for t=1:size(data,3)
        # kde est of spatial dist at time t
        k = kde(data[:,:,t])
        X[:,:,t] = pdf(k, xs,ys)
    end

    return X
end

# get ensemble data for 1d OU
function Get1DEnsData(f,g, u0, dt, tspan, rng;
        num_traj=1000, 
        noise_mag = 0.1
    )
    # define SDE problem
    prob = SDEProblem(f,g, u0, tspan)
    # remakes
    function prob_func(prob, i, repeat)
        remake(prob, u0 = noise_mag*randn(rng) + prob.u0)
    end

    ensembleprob = EnsembleProblem(prob, prob_func=prob_func)
    sol = solve(ensembleprob, EnsembleThreads(), trajectories = num_traj, saveat=dt)

    # make an array of the solns
    X = zeros((num_traj , length(sol[1].t) ))

    for i=1:num_traj
        X[i, :] = sol[i].u
    end

    return X, sol[1].t 
end

# get ensemble data for 2-d brownian 
function BrownianData(u0::Vector{Float64}, tspan::Tuple{Float64, Float64}, dt::Float64, ps::Vector{Float64}, trajs::Int, rng ; 
    sig_init = 0.1 ,
    xs = -2:0.01:2,
    ys = -2:0.01:2 
    )
    # simulate SDE
    #σ=ps[1]

    function f(du, u, p, t)
        du[1] = 0.
        du[2] = 0.
    end
    
    function g(du, u, p, t)
        du[1] = p[1]
        du[2] = p[2]
    end

    prob = SDEProblem(f, g, u0, tspan,ps)

    # run many trajectories, collect data
    prob_func(prob, i, repeat) = remake(prob, u0 = sig_init*randn(rng, length(u0)) + prob.u0) # this 
    ensembleprob = EnsembleProblem(prob, prob_func=prob_func)
    sol = solve(ensembleprob, EnsembleThreads(), trajectories = trajs, saveat=dt)

    # make an array of the solns
    X = zeros((trajs , length(u0), length(sol[1].t)))

    for i=1:trajs
        X[i,:,:] = Array(sol[i])
    end
    
    Xd = EstDist(X,xs,ys)
    #Xd = Xd ./ sum(Xd, dims=1)

    return sol[1].t, X, Xd  
end

using Catalyst, DifferentialEquations

# function serves to build odes to estimate the kernel of the RNA-NP system. It: 
# 1. Uses Catalyst.jl to build the equations with a kernel we want to estimate 
# 2. Uses DiffEqParamEstim.jl to determine the kernel values that minimize the L2 loss with data 
# 3. Returns the estimated kernel, the indices used, and other quantities 
function SimAgg(N::Int , u0vec::AbstractVector, tspan, dt , Kr; 
        i = 2,
        df = 1.5 ,
        γ = 1  , # exponent for product kernel 
        odealgo = Tsit5() , # solver for ode system
    )
    #=
    =========== INPUTS ==============
    N: number of particles to simulate 
    Xn: 
    u0vec: initial particle count 
    k0: initial guess of kernel 
    =# 
    # ========================= GENERATE ODE =====================================


    integ(x) = Int(floor(x))
    n        = integ(N/2)
    nr       = N%2 == 0 ? (n*(n + 1) - n) : (n*(n + 1)) # No. of forward reactions

    # POSSIBLE PAIRS OF REACTING MULTIMERS
    pair = []
    for i = 2:N
        push!(pair,[1:integ(i/2)  i .- (1:integ(i/2))])
    end
    pair = vcat(pair...) ;
    vᵢ = @view pair[:,1] ;  # Reactant 1 indices
    vⱼ = @view pair[:,2] ; # Reactant 2 indices
    sum_vᵢvⱼ = @. vᵢ + vⱼ ; # Product index

    # BUILD KERNEL VALUES
    if i==1
        B = 1.53e03                # s⁻¹
        kv = @. Kr*(vᵢ + vⱼ)  ; # dividing by volume as its a bi-molecular reaction chain
        #kv = @. 1e-2 * (vᵢ + vⱼ)
    elseif i==2           
        kv = fill(Kr, nr) ; # constant kernel 
    elseif i==3
        kv = @. Kr  * (vᵢ * vⱼ)^γ ; # product kernel 
    elseif i==4
        kv = @. Kr * (2 + (vᵢ/vⱼ)^(1/df)+ (vⱼ/vᵢ)^(1/df)) ; # Full brownian kernel 
    end

    # USE ModelingToolkit TO BUILD PARAMS & VARS
    # state variables are X, pars stores rate parameters for each rx
    @parameters t k[1:nr] # ks had to be added to parameters
    @species X(t)[1:N] # defining the reactions and species 
    pars = Pair.(collect(k), kv)

    u₀map = Pair.(collect(X), u0vec)   # map variable to its initial value
    # BUILD Catalyst REACTION SYSTEM 
    # vector to store the Reactions in
    rx = []
    for n = 1:nr
        # for clusters of the same size, double the rate
        if (vᵢ[n] == vⱼ[n])
            # form of the Reaction arg ==> Reaction(rate, [species in], [species out], [stoic coef in], [stoic coef out])
            push!(rx, Reaction(k[n], [X[vᵢ[n]]], [X[sum_vᵢvⱼ[n]]], [2], [1])) 
        else
            push!(rx, Reaction(k[n], [X[vᵢ[n]], X[vⱼ[n]]], [X[sum_vᵢvⱼ[n]]],
                            [1, 1], [1]))
        end
    end

    @named rs = ReactionSystem(rx, t, collect(X), collect(k))
    rs_c = complete(rs)
    # BUILD AND SOLVE ODE PROBLEM
    # trying conversion to an ODE system as well
    #odesys = convert(ODESystem, rs_c)
    odeprob = ODEProblem(rs_c,  u₀map, tspan, pars)
    soln = solve(odeprob, Tsit5(), saveat=dt)

    return soln.t, Array(soln)
end
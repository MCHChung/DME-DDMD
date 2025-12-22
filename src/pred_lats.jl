using DifferentialEquations.OrdinaryDiffEq

function PredTraj(Lib::Function, Ξ, u0, ts; m=1)
    prob = ODEProblem((u,p,t) -> vec(Lib(u')*p), u0, (ts[1], m*ts[end]), Ξ)
    soln = solve(prob, Tsit5(), saveat=ts[2]-ts[1])
    return Int(soln.retcode) != 1 ? throw("Did not converge") : Array(soln)'
end

function PredTraj(N, Ξ, u0, ts; m=1, disc=false)
    function Lib(z)
        @assert size(z,2) == size(N,2)
        T, L = size(z,1), size(N,1)
        θ = zeros(T,L)
        for l=1:L
            θ[:,l] = ones(T)
            for s=1:size(N,2)
                θ[:,l] = θ[:,l].*z[:,s].^N[l,s]
            end
        end

        return θ
    end
    if !disc 
        prob = ODEProblem((u,p,t) -> vec(Lib(u')*p), u0, (ts[1], m*ts[end]), Ξ)
        soln = solve(prob, Tsit5(), saveat=ts[2]-ts[1])
        return Int(soln.retcode) != 1 ? throw("Did not converge") : Array(soln)'
    else
        zpred = zeros(length(ts), length(u0))
        zpred[1, :] = u0 
        for t=2:length(ts)
            zpred[t, :] = zpred[t-1, :] + vec(Lib(zpred[t-1, :]')*Ξ)
        end

        return zpred 
    end
end


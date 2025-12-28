using LinearAlgebra

function PolyLib(x; order=2, use_cons=true)
    @assert 0 < order <= 5 && typeof(order) <: Int
    n = size(x,1)
    θ = ones(size(x,2))
    if order==1
        for i=1:n
            θ = hcat(θ, x[i, :])
        end
    elseif order == 2
        for i=1:n
            θ = hcat(θ, x[i, :])
        end

        for i=1:n
            for j=i:n
                θ = hcat(θ, x[i, :].*x[j,:])
            end
        end
    elseif order==3
        for i=1:n
            θ = hcat(θ, x[i, :])
        end

        for i=1:n
            for j=i:n
                θ = hcat(θ, x[i, :].*x[j,:])
            end
        end

        for i=1:n
            for j=i:n
                for k=j:n
                    θ = hcat(θ, x[i, :].*x[j,:].*x[k,:])
                end
            end
        end
    elseif order==4
        for i=1:n
            θ = hcat(θ, x[i, :])
        end

        for i=1:n
            for j=i:n
                θ = hcat(θ, x[i, :].*x[j,:])
            end
        end

        for i=1:n
            for j=i:n
                for k=j:n
                    θ = hcat(θ, x[i, :].*x[j,:].*x[k,:])
                end
            end
        end

        for i=1:n
            for j=i:n
                for k=j:n
                    for l=k:n
                        θ = hcat(θ, x[i, :].*x[j,:].*x[k,:].*x[l,:])
                    end
                end
            end
        end
    elseif order==5 

        for i=1:n
            θ = hcat(θ, x[i, :])
        end

        for i=1:n
            for j=i:n
                θ = hcat(θ, x[i, :].*x[j,:])
            end
        end

        for i=1:n
            for j=i:n
                for k=j:n
                    θ = hcat(θ, x[i, :].*x[j,:].*x[k,:])
                end
            end
        end

        for i=1:n
            for j=i:n
                for k=j:n
                    for l=k:n
                        θ = hcat(θ, x[i, :].*x[j,:].*x[k,:].*x[l,:])
                    end
                end
            end
        end

        for i=1:n
            for j=i:n
                for k=j:n
                    for l=k:n
                        for m=l:n
                            θ = hcat(θ, x[i, :].*x[j,:].*x[k,:].*x[l,:].*x[m,:])
                        end
                    end
                end
            end
        end
    end

    if use_cons
        return θ'
    else
        return θ[:,2:end]'
    end

end

function Linear(x)
    return PolyLib(x, order=1, use_cons=false)
end

function Cons(x)
    return ones(size(x))
end

function TrigLib(x; use_ident=true)
    n = size(x,1)
    θ = ones(size(x,2))
    for i=1:n
        θ = hcat(θ, sin.(x[i, :]))
        θ = hcat(θ, cos.(x[i, :]))
    end

    if use_ident
        θ = hcat(θ, x')
    end

    return θ'
end

function ExpLib(x; use_ident=true, use_hyp=false)
    n = size(x,1)
    θ = ones(size(x,2))
    if !use_hyp
        for i=1:n
            θ = hcat(θ, exp.(x[i, :]))
        end
    else
        for i=1:n
            θ = hcat(θ, sinh.(x[i, :]))
            θ = hcat(θ, cosh.(x[i, :]))
        end
    end

    if use_ident
        θ = hcat(θ, x')
    end

    return θ'
end
#=
function RationalLib(x; use_ident=true)
    n = size(x,1)
    θ = ones(size(x,2))
    for i=1:n
        θ = hcat(θ, x[i, :] ./ (1 .+ x[i, :]))
    end

    if use_ident
        θ = hcat(θ, x')
    end

    return θ'
end
=#

function RationalLib(x; use_ident=true, tol=1e-6)
    n = size(x,1)
    θ = ones(size(x,2))
    for i=1:n
        for j=i:n
            mean(abs, x[j, :]) < tol ? continue : nothing
            θ = hcat(θ, x[i, :] ./ x[j, :])
        end
    end

    if use_ident
        θ = hcat(θ, x')
    end

    return θ'
end

function SqrtLib(x)
    return abs.(PolyLib(x, order=1, use_cons=false)).^(1/2)
end
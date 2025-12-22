using LinearAlgebra 

function elPOD(X::Array{Float64,2}, K::Int; eps=1e-8)
    # log of data 
    C = log.(X .+ eps)
    # svd 
    U, S, V = svd(C)
    # truncate and reconstruct
    Cest = U[:,1:K]*diagm(S[1:K])*V[:,1:K]'
    # exponentiate and normalize
    Xest = exp.(Cest)
    Xest = Xest ./ sum(Xest, dims=1)
    return Xest
end

function elPOD(X::Array{Float64,3}, K::Int; eps=1e-8)
    Xsvd = copy(X)
    for i=1:size(X,3)
        Xsvd[:,:,i] = elPOD(X[:,:,i], K; eps=eps)
    end
    return Xsvd
end
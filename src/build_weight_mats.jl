using SparseArrays , ForwardDiff

# define weight functions for Galerkin-proj
w(t,p,a,b) = @. (1/(p)^(2*p))*((2p/(b-a))^(2p))*((t-a)*(b-t))^p 
dw(t,p,a,b) = @. ForwardDiff.derivative(x -> w(x,p,a,b), t)
d2w(t,p,a,b) = @. ForwardDiff.derivative(x -> dw(x,p,a,b), t)

# Builds weight matrices for Galerkin-projection of dynamics 
function BuildWeightMat(ts, wind::Int, d::Int; p=10)
    @assert 0 <= d <= 2
    dt = ts[2] - ts[1]
    n = length(ts) 
    W = zeros(n-wind,n)
    if d == 0
        for i=1:n-wind
            ts_i = @view ts[i:i+wind]
            sc_ts_i = ts_i*(2/(ts_i[end]-ts_i[1])) .- (ts_i[end]+ts_i[1])/(ts_i[end]-ts_i[1])
            W[i, i:i+wind] = 2*w(sc_ts_i, p, -1, 1) / (2/(ts_i[end] - ts_i[1]))
        end
    elseif d == 1  
        for i=1:n-wind
            ts_i = @view ts[i:i+wind]
            sc_ts_i = ts_i*(2/(ts_i[end]-ts_i[1])) .- (ts_i[end]+ts_i[1])/(ts_i[end]-ts_i[1])
            W[i, i:i+wind] = -2*dw(sc_ts_i, p, -1, 1) 
        end
    elseif d == 2  
        for i=1:n-wind
            ts_i = @view ts[i:i+wind]
            sc_ts_i = ts_i*(2/(ts_i[end]-ts_i[1])) .- (ts_i[end]+ts_i[1])/(ts_i[end]-ts_i[1])
            W[i, i:i+wind] = -2*d2w(sc_ts_i, p, -1, 1)*(2/(ts_i[end] - ts_i[1])) 
        end
    end

    return sparse(W)*dt/2
end

function BuildDiscWeightMat(ts)
    n = length(ts) 
    DW = zeros(n,n) # discrete 
    for i=1:n-1 
        DW[i,i:i+1] = [-1, 1]
    end


    return sparse(DW), I
end

# to-do here: automate the generation of the polynomial degree matrix
function BuildPolyDegreeMat(K::Int, order::Int)
    @assert 0 < order <= 3 

end
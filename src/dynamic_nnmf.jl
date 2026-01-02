using LinearAlgebra, StatsBase, Random


const EPS = 1e-12

function normalize_vector!(v)
    s = sum(v)
    if s > 0
        v ./= s
    end
    v
end

function normalize_columns!(M)
    for j in axes(M,2)
        normalize_vector!(view(M,:,j))
    end
    M
end

function responsibilities(W, h)
    K, I = size(W)
    Γ = W .* h'
    denom = sum(Γ, dims=2) .+ EPS
    Γ ./ denom
end

function update_W(X, Γ)
    K, T = size(X)
    I = size(Γ,2)

    W = zeros(K,I)
    for t in 1:T
        W .+= X[:,t] .* Γ[:,:,t]
    end

    normalize_columns!(W)
end

function predict_h(A, H, t)
    I = size(H,1)
    η = zeros(I)

    for j in eachindex(A)
        t - j ≥ 1 && (η .+= A[j] * H[:,t-j])
    end

    η .+ EPS
end


function update_h(x, W, h, η; inner_iters=2, newton_iters=25)
    I = length(h)

    for _ in 1:inner_iters
        Γ = responsibilities(W, h)
        num = vec(sum(x .* Γ, dims=1))

        β = 0.0
        βmin = -minimum(1.0 ./ η) + 1e-6

        for _ in 1:newton_iters
            denom = β .+ 1.0 ./ η
            denom .= max.(denom, EPS)

            htmp = num ./ denom
            f = sum(htmp) - 1
            df = -sum(num ./ denom.^2)

            β -= f / (df + EPS)
            β = max(β, βmin)
        end

        h .= num ./ (β .+ 1.0 ./ η)
        h .= max.(h, EPS)
        normalize_vector!(h)
    end

    h
end

function update_A!(Acat, H, V; niter=1)
    for _ in 1:niter
        AV = Acat * V .+ EPS
        Acat .*= ((H ./ AV.^2) * V') ./ ((1 ./ AV) * V')
        Acat .= max.(Acat, EPS)
    end
end

function dynamic_nmf(
    X,
    I;
    J=1,
    niter=100,
    M=50,
    q=0.15,
    seed=0
)
    Random.seed!(seed)

    K, T = size(X)

    # ---- PLCA scaling (CRITICAL) ----
    X = copy(X)
    X ./= sum(X, dims=1) .+ EPS
    X .*= 1000.0

    # ---- Initialization ----
    W = rand(K,I)
    normalize_columns!(W)

    H = rand(I,T)
    normalize_columns!(H)

    A = [rand(I,I) .+ 0.1 for _ in 1:J]

    for r in 1:niter
        r % 100 == 0 ? println("Iteration $r") : nothing

        # ===== Update W =====
        Γ = Array{Float64,3}(undef, K, I, T)
        for t in 1:T
            Γ[:,:,t] = responsibilities(W, H[:,t])
        end
        W = update_W(X, Γ)

        # ===== Update H (filtering) =====
        for t in 1:T
            if r > M
                η = predict_h(A, H, t)
                η .= η .^ q
            else
                η = ones(I)
            end

            update_h(X[:,t], W, view(H,:,t), η)
        end

        # ===== Update A =====
        if r ≥ M
            V = zeros(I*J, T)
            for t in 1:T, j in 1:J
                t-j ≥ 1 && (V[(j-1)*I+1:j*I, t] .= H[:,t-j])
            end

            Acat = hcat(A...)
            update_A!(Acat, H, V)

            for j in 1:J
                A[j] .= Acat[:, (j-1)*I+1:j*I]
            end
        end
    end

    return W, H, A
end

function forecast_states(H, A, steps)
    I, T = size(H)
    J = length(A)

    Hf = zeros(I, T + steps)
    Hf[:,1:T] .= H

    for t in T+1:T+steps
        for j in 1:J
            Hf[:,t] .+= A[j] * Hf[:,t-j]
        end
        Hf[:,t] .= max.(Hf[:,t], 0)
        Hf[:,t] ./= sum(Hf[:,t]) + 1e-12
    end

    return Hf[:,T+1:end]
end

function forecast_observations(W, Hf; gamma=1.0)
    Xf = gamma .* (W * Hf)
    return Xf
end
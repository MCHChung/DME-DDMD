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
        X::Matrix,
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


# =================== TENSOR =================


#const EPS = 1e-12

function normalize_vector!(v)
    s = sum(v)
    s > 0 && (v ./= s)
    v
end

function normalize_columns!(M)
    for j in axes(M,2)
        normalize_vector!(view(M,:,j))
    end
    M
end

function responsibilities(W, h)
    Γ = W .* h'
    Γ ./ (sum(Γ, dims=2) .+ EPS)
end

function update_W_tensor(X, Hs, W)
    K, T, E = size(X)
    I = size(W,2)

    Wnew = zeros(K,I)

    for e in 1:E
        for t in 1:T
            Γ = responsibilities(W, Hs[e][:,t])
            Wnew .+= X[:,t,e] .* Γ
        end
    end

    normalize_columns!(Wnew)
end

function predict_h(A, H, t)
    I = size(H,1)
    η = zeros(I)
    for j in eachindex(A)
        t - j ≥ 1 && (η .+= A[j] * H[:,t-j])
    end
    η .+ EPS
end

function update_h!(x, W, h, η; inner_iters=2)
    I = length(h)

    for _ in 1:inner_iters
        Γ = responsibilities(W, h)
        num = vec(sum(x .* Γ, dims=1))

        β = 0.0
        βmin = -minimum(1.0 ./ η) + 1e-6

        for _ in 1:25
            denom = max.(β .+ 1.0 ./ η, EPS)
            f = sum(num ./ denom) - 1
            df = -sum(num ./ denom.^2)
            β -= f / (df + EPS)
            β = max(β, βmin)
        end

        h .= num ./ (β .+ 1.0 ./ η)
        h .= max.(h, EPS)
        normalize_vector!(h)
    end
end

function update_A_shared!(A, Hs)
    I = size(Hs[1],1)
    J = length(A)
    E = length(Hs)
    T = size(Hs[1],2)

    V = zeros(I*J, E*T)
    Hcat = zeros(I, E*T)

    col = 1
    for e in 1:E
        H = Hs[e]
        for t in 1:T
            Hcat[:,col] .= H[:,t]
            for j in 1:J
                t-j ≥ 1 && (V[(j-1)*I+1:j*I, col] .= H[:,t-j])
            end
            col += 1
        end
    end

    Acat = hcat(A...)
    for _ in 1:1
        AV = Acat * V .+ EPS
        Acat .*= ((Hcat ./ AV.^2) * V') ./ ((1 ./ AV) * V')
        Acat .= max.(Acat, EPS)
    end

    for j in 1:J
        A[j] .= Acat[:, (j-1)*I+1:j*I]
    end
end

function dynamic_nmf(
    X::Array{<:Real, 3},            # K × T × E
    I;
    J=1,
    niter=100,
    M=50,
    q=0.15,
    seed=0,
    iter = 100
    )
    Random.seed!(seed)

    K, T, E = size(X)

    # --- PLCA scaling ---
    X = copy(X)
    for e in 1:E, t in 1:T
        X[:,t,e] ./= sum(X[:,t,e]) + EPS
        X[:,t,e] .*= 1000
    end

    # --- Init ---
    W = rand(K,I)
    normalize_columns!(W)

    Hs = [normalize_columns!(rand(I,T)) for _ in 1:E]
    A  = [rand(I,I) .+ 0.1 for _ in 1:J]

    for r in 1:niter
        r % iter == 0 ? println("Iteration $r") : nothing

        # ---- Update W ----
        W = update_W_tensor(X, Hs, W)

        # ---- Update H (per experiment) ----
        for e in 1:E
            H = Hs[e]
            for t in 1:T
                η = r > M ? predict_h(A, H, t).^q : ones(I)
                update_h!(X[:,t,e], W, view(H,:,t), η)
            end
        end

        # ---- Update shared A ----
        r ≥ M && update_A_shared!(A, Hs)
    end

    return W, Hs, A
end

# ======== UPDATE MATRICES =========

function infer_H_fixedW(
    Xnew::Matrix,
    W,
    A;
    niter=50,
    M=0,
    q=0.15
    )
    K, T = size(Xnew)
    I = size(W,2)
    J = length(A)

    # --- PLCA scaling ---
    X = copy(Xnew)
    X ./= sum(X, dims=1) .+ EPS
    X .*= 1000.0

    # --- Initialize H ---
    H = rand(I,T)
    normalize_columns!(H)

    for r in 1:niter
        for t in 1:T
            η = r > M ? predict_h(A, H, t).^q : ones(I)
            update_h(X[:,t], W, view(H,:,t), η)
        end
    end

    return H
end

function update_A_from_H!(A, H)
    I, T = size(H)
    J = length(A)

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

function update_model_matrix!(
    Xnew,
    W,
    A;
    infer_iters=50,
    update_A=true
    )

    Hnew = infer_H_fixedW(Xnew, W, A; niter=infer_iters)

    if update_A
        update_A_from_H!(A, Hnew)
    end

    return Hnew, A
end

function infer_Hs_fixedW(
    Xnew::Array{<:Real,3},
    W,
    A;
    niter=50,
    M=0,
    q=0.15
    )
    K, T, E = size(Xnew)
    I = size(W,2)

    # --- PLCA scaling ---
    X = copy(Xnew)
    for e in 1:E, t in 1:T
        X[:,t,e] ./= sum(X[:,t,e]) + EPS
        X[:,t,e] .*= 1000
    end

    Hs = [normalize_columns!(rand(I,T)) for _ in 1:E]

    for r in 1:niter
        for e in 1:E
            H = Hs[e]
            for t in 1:T
                η = r > M ? predict_h(A, H, t).^q : ones(I)
                update_h!(X[:,t,e], W, view(H,:,t), η)
            end
        end
    end

    return Hs
end

function update_model_tensor!(
    Xnew,
    W,
    A;
    infer_iters=50,
    update_A=true
    )
    Hs_new = infer_Hs_fixedW(Xnew, W, A; niter=infer_iters)

    if update_A
        update_A_shared!(A, Hs_new)
    end

    return Hs_new, A
end

function estimate_W_from_H(
    X::Matrix,
    H;
    niter=50,
    seed=0
    )
    Random.seed!(seed)

    K, T = size(X)
    I = size(H,1)

    # --- PLCA scaling ---
    X = copy(X)
    X ./= sum(X, dims=1) .+ EPS
    X .*= 1000.0

    # --- Initialize W ---
    W = rand(K,I)
    normalize_columns!(W)

    for _ in 1:niter
        Γ = Array{Float64,3}(undef, K, I, T)

        for t in 1:T
            Γ[:,:,t] = responsibilities(W, H[:,t])
        end

        W = update_W(X, Γ)
    end

    return W
end

function estimate_W_from_Hs(
    X::Array{<:Real,3},
    Hs;
    niter=50,
    seed=0
    )
    Random.seed!(seed)

    K, T, E = size(X)
    I = size(Hs[1],1)

    # --- PLCA scaling ---
    X = copy(X)
    for e in 1:E, t in 1:T
        X[:,t,e] ./= sum(X[:,t,e]) + EPS
        X[:,t,e] .*= 1000
    end

    # --- Initialize W ---
    W = rand(K,I)
    normalize_columns!(W)

    for _ in 1:niter
        W = update_W_tensor(X, Hs, W)
    end

    return W
end
using DrWatson
@quickactivate "TMI&MD"

# axes , originally: (xb=1, tend=2)
xb = 1.5
xs = -xb:0.01:xb
ts = 0:0.01:3

# ========== generate TRAIN data =============
P = zeros(length(xs), length(ts)) # train
for i=1:length(ts)
    p = @. exp(-xs^2/(ts[i] + 1e-2)) # simple diffusing gaussian 
    p = p / sum(p) # normalize
    P[:,i] = p
end

diff_data = @strdict xs ts P 
#wsave(datadir("sims", "diff_data2.jld2"), diff_data)

# ========== generate TEST data =============
rng = Random.default_rng()
Random.seed!(rng, 0)

Ps_new = []
x0s_new = []
for j=1:10
    P = zeros(length(xs), length(ts)) # train
    Pnew = similar(P) # test 
    D = rand(rng, 0.5:0.3:10)
    x0_new = rand(rng, -0.5:0.1:0.5)
    for i=1:length(ts)
        
        p = exp.(-D*(xs .- x0_new).^2 / (ts[i] + 1e-2)) # simple diffusing gaussian 
        p = p / sum(p) # normalize
        Pnew[:,i] = p
        
    end
    push!(Ps_new,Pnew)
    push!(x0s_new, x0_new)
end

diff_test_data = @strdict xs ts Ps_new x0s_new
wsave(datadir("sims", "diff_test_data.jld2"), diff_test_data)

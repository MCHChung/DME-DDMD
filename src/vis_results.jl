using Plots, Measures, LinearAlgebra 

default(dpi=300, grid=false, fontfamily="computer modern")
#scalefontsizes(1/1.2^2)

# latent visualization
function PlotResults(data, pred, flabel::String; logtrans=true, eps=1e-3)
    if length(pred) == 3 
        ts, xs, P = data
        z, Y, q = pred 

        p1 = plot(ts, z, label=false, lw=3)
        xlabel!("Time")
        ylabel!("Z")
        p2 = plot(xs, Y, label=false,  lw=3)
        xlabel!(flabel)
        ylabel!("Y")

        p12 = plot(p1,p2, layout=(1,2), size=(800,400), margins=5*Measures.mm)

        # data 
        Pp = logtrans ? log10.(P .+ eps) : P
        clims = (min(Pp...) , max(Pp...))
        p3 = heatmap(ts, xs, Pp, label=false, clims=clims)
        ylabel!(flabel)
        xlabel!("Time")
        title!("Data")

        # prediction
        K = size(z,2)
        qp = logtrans ? log10.(q .+ eps) : q
        p4 = heatmap(ts,xs, qp, label=false, clims=clims)
        ylabel!(flabel)
        xlabel!("Time")
        title!("TMI, K=$(K)")

        # SVD
        U,S,V = svd(log.(P .+ 1e-6))
        Psvd = exp.(U[:, 1:K]*diagm(S[1:K])*V[:,1:K]')
        Psvd = Psvd ./ sum(Psvd, dims=1)

        Psvd_p = logtrans ? log10.(Psvd .+ eps) : Psvd
        p5 = heatmap(ts,xs, Psvd_p, label=false, clims=clims)
        ylabel!(flabel)
        xlabel!("Time")
        title!("SVD")

        p34 = plot(p3,p4,p5, layout=(1,3), size=(1800,400), margins=10*Measures.mm)

        return p12, p34 
    else 
        # unpack data 
        ts, xs, P = data
        E, z, Y, q = pred 

        p1 = plot(ts, z, label=false,  lw=3)
        xlabel!("Time")
        ylabel!("Z")

        p2 = plot(xs, Y, label=false,  lw=3)
        xlabel!("Feature")
        ylabel!("Y")

        p3 = plot(xs, E, label=false,  lw=3)
        xlabel!("Feature")
        ylabel!("E")

        p123 = plot(p1,p2,p3, layout=(1,3), size=(1200,400), margins=5*Measures.mm)

        clims = (min(P...) , max(P...))
        p4 = heatmap(ts,xs, P, label=false, clims=clims)
        ylabel!("feature")
        xlabel!("Time")
        title!("Data")

        K = size(z,2)
        p5 = heatmap(ts, xs, q, label=false, clims=clims)
        ylabel!("feature")
        xlabel!("Time")
        title!("TMI, K=$(K)")

        p45 = plot(p4,p5, layout=(1,2), size=(1000,400), margins=5*Measures.mm)

        return p123, p45 
    end 
end

function PlotLatents(ts, Z::Array{Float64,2}, xs, Y::Array{Float64,2}; lw = 2, zlabel=false, ylabel=false)
    p1 = plot(ts, Z, lw=lw, label=zlabel)
    xlabel!("Time")
    ylabel!("Z")
    p2 = plot(xs, Y, lw=lw, label=ylabel)
    xlabel!("Feature")
    ylabel!("Y")
    p = plot(p1,p2, layout=(1,2), size=(800,400), margins=5*Measures.mm)
    return p
end

function PlotLatents(ts, Z::Array{Float64,2}, xs, ys, Y::Array{Float64,2}; lw = 2, zlabel=false, ylabel=false)
    p1 = plot(ts, Z, lw=lw, label=zlabel, size=(400,400))
    xlabel!("Time")
    ylabel!("Z")
    p2 = surface(xs, ys, Y, lw=lw, label=ylabel, size=(600,600), alpha=0.9)
    xlabel!("Feature (x)")
    ylabel!("Feature (y)")
    zlabel!("Y")
    #p = plot(p1,p2, layout=(1,2), size=(1200,800), margins=5*Measures.mm)
    return p1, p2
end

function PlotLatents(Z::Array{Float64,3}; color=:black, lw=3)
    p = plot(size=(400,400))
    for i=1:size(Z,3)
        plot!(Z[:,1,i], Z[:,2,i], color=color, label=false, lw=lw)
        #scatter!([Z[1,1,i]], [Z[2,2,i]], color=:magenta, label=false)
    end
    xlabel!("Z1")
    ylabel!("Z2")
    return p
end

function PlotLatents(Z::Array{Float64,3}, Zpred::Array{Float64,3}, lw=3)
    p = plot(size=(400,400))
    for i=1:size(Z,3)
        plot!(Z[:,1,i], Z[:,2,i], color=:black, label=false, lw=lw)
        plot!(Zpred[:,1,i], Zpred[:,2,i], color=:red, ls=:dash, label=false, lw=lw)

    end
    xlabel!("Z1")
    ylabel!("Z2")
    return p
end

# model visualization 
function VisModel(Ξtrue, Ξes)
    clims = (min(Ξtrue...), max(Ξtrue...))
    p1 = heatmap(replace(Ξtrue, 0=>NaN), yflip=true, aspect_ratio=1, color=:blues, axis=false, clims=clims, title="True/Exp. Model")
    p2 = heatmap(replace(Ξes, 0=>NaN), yflip=true, aspect_ratio=1, color=:blues, axis=false, clims=clims, title="Est. Model")
    plot(p1,p2,layout=(1,2), size=(800,800), margin=7*Measures.mm)
    xlabel!("Variables")
    ylabel!("Model terms")
end

function VisModel(Ξtrue::Vector, Ξes::Vector)
    Ξtrue = reshape(Ξtrue, (length(Ξtrue), 1))
    Ξes = reshape(Ξes, (length(Ξes), 1))

    clims = (min(Ξtrue...), max(Ξtrue...))
    p1 = heatmap(replace(Ξtrue, 0=>NaN), yflip=true, aspect_ratio=1, color=:blues, axis=false, clims=clims, title="True/Exp. Model")
    p2 = heatmap(replace(Ξes, 0=>NaN), yflip=true, aspect_ratio=1, color=:blues, axis=false, clims=clims, title="Est. Model")
    plot(p1,p2,layout=(1,2), size=(800,800), margin=7*Measures.mm)
    xlabel!("Variables")
    ylabel!("Model terms")
end

function VisModel(Ξtrue, Ξes, label::String, margin=15)
    clims = (min(Ξtrue...), max(Ξtrue...))
    p1 = heatmap(replace(Ξtrue, 0=>NaN), yflip=true, aspect_ratio=1, color=:blues, axis=false, clims=clims, title="True/Exp. Model")
    p2 = heatmap(replace(Ξes, 0=>NaN), yflip=true, aspect_ratio=1, color=:blues, axis=false, clims=clims, title="Est. Model : "*label)
    plot(p1,p2,layout=(1,2), size=(800,800), margin=margin*Measures.mm)
    xlabel!("Variables")
    ylabel!("Model terms")
end

function VisModel(Ξes, label::String, margin=15)
    heatmap(replace(Ξes, 0=>NaN), yflip=true, aspect_ratio=1, color=:blues, axis=false, title="Est. Model : "*label, margins=margin*Measures.mm)
    xlabel!("Variables")
    ylabel!("Model terms")
end
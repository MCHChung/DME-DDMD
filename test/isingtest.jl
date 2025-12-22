using Random, Plots

# Function to initialize a 2D lattice with random spins (+1 or -1)
function initialize_lattice(L)
    return 2 .* (rand(Bool, L, L) .- 0.5)  # Random +1 or -1 spins
end

# Compute energy of a given spin configuration
function compute_energy(lattice, J)
    L = size(lattice, 1)
    energy = 0.0
    for i in 1:L, j in 1:L
        s = lattice[i, j]
        neighbors = lattice[mod1(i+1, L), j] + lattice[mod1(i-1, L), j] +
                    lattice[i, mod1(j+1, L)] + lattice[i, mod1(j-1, L)]
        energy -= J * s * neighbors  # Interaction energy
    end
    return energy / 2  # Each pair counted twice
end

# Perform a Monte Carlo step using the Metropolis algorithm
function monte_carlo_step!(lattice, beta, J)
    L = size(lattice, 1)
    for _ in 1:(L^2)  # Sweep through L^2 updates
        i, j = rand(1:L), rand(1:L)
        s = lattice[i, j]
        neighbors = lattice[mod1(i+1, L), j] + lattice[mod1(i-1, L), j] +
                    lattice[i, mod1(j+1, L)] + lattice[i, mod1(j-1, L)]
        dE = 2 * J * s * neighbors  # Energy change if flipped
        if dE < 0 || rand() < exp(-beta * dE)
            lattice[i, j] *= -1  # Flip spin
        end
    end
end

# Run the simulation
function run_simulation(L, beta, J, eqsteps, mcsteps)
    lattice = initialize_lattice(L)
    energies = Float64[]
    magnetizations = Float64[]
    E1 = E2 = M1 = M2 = 0

    for step in 1:eqsteps
        monte_carlo_step!(lattice, beta, J)
    end

    for step in 1:mssteps
        monte_carlo_step!(lattice, beta, J)
        E = compute_energy(lattice, J) 
        M = sum(lattice) / (L^2)
        E1 += E
        M1 += M
        E2 += E^2 
        M2 += M^2
        #push!(energies, compute_energy(lattice, J))
        #push!(magnetizations, sum(lattice) / (L^2))  # Normalized magnetization
    end
    
    #return energies, magnetizations, lattice
    return E1 , E2 , M1 , M2
end

# Parameters
L = 20  # Lattice size
Js = [0.5, 1.0, 1.5]  # Interaction strengths
Ts = range(0.1,3, 100)
betas = 1 ./ Ts #[0.2, 0.5, 1.0]  # Inverse temperatures (1/kT)
eqsteps, mcsteps = 1000, 1000  # Monte Carlo steps
n1, n2 = 1/(mcsteps*L^2)
# Run simulations over multiple Js and betas
results = Dict()
for J in Js, beta in betas
    println("Running simulation for J=$J, beta=$beta")
    #energies, magnetizations, final_lattice = run_simulation(L, beta, J, eqsteps, mcsteps)
    E1, E2, M1, M2 = run_simulation(L, beta, J, eqsteps, mcsteps) 
    E = 
    results[(J, beta)] = (energies, magnetizations, final_lattice)
end

# Example plot for one J and beta
J_plot, beta_plot = 0.5, betas[50]
energies, magnetizations, _ = results[(J_plot, beta_plot)]

plot(energies, label="Energy", xlabel="Monte Carlo steps", ylabel="Energy", title="Energy vs. MC Steps for J=$J_plot, beta=$beta_plot")
plot(magnetizations, label="Magnetization", xlabel="Monte Carlo steps", ylabel="Magnetization", title="Magnetization vs. MC Steps for J=$J_plot, beta=$beta_plot")

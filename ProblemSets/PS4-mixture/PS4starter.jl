# Problem Set 4 Starter Code
# ECON 6343: Econometrics III
# Multinomial and Mixed Logit Estimation

using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions

# Include quadrature function (make sure lgwt.jl is in your working directory)
include("lgwt.jl")

#---------------------------------------------------
# Data Loading Function
#---------------------------------------------------
function load_data()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, 
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occ_code
    return df, X, Z, y
end

#---------------------------------------------------
# Question 1: Multinomial Logit with Alternative-Specific Covariates
#---------------------------------------------------

function mlogit_with_Z(theta, X, Z, y)
    # Extract parameters
    # theta = [alpha1, alpha2, ..., alpha21, gamma]
    # alpha has K*(J-1) = 3*7 = 21 elements  
    # gamma is the coefficient on Z
    alpha = theta[1:end-1]  # first 21 elements
    gamma = theta[end]      # last element
    
    K = size(X, 2)  # number of covariates in X (3)
    J = length(unique(y))  # number of choices (8)
    N = length(y)   # number of observations
    
    # Create choice indicator matrix
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end
    
    # Reshape alpha into K x (J-1) matrix, add zeros for normalized choice J
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
    
    # TODO: Compute choice probabilities
    # Hint: P_ij = exp(X_i*beta_j + gamma*(Z_ij - Z_iJ)) / denominator
    # where denominator sums over all choices
    
    # Initialize probability matrix  
    T = promote_type(eltype(X), eltype(theta))
    num = zeros(T, N, J)
    dem = zeros(T, N)
    
    # Fill in: compute numerator for each choice j
    for j = 1:J
        # num[:,j] = exp.(X * bigAlpha[:,j] .+ gamma .* (Z[:,j] .- Z[:,J]))
    end
    
    # Fill in: compute denominator (sum of numerators)
    # dem = sum(num, dims=2)
    
    # Fill in: compute probabilities
    # P = num ./ dem
    
    # Fill in: compute negative log-likelihood
    # loglike = -sum(bigY .* log.(P))
    
    return loglike
end

#---------------------------------------------------
# Question 3a: Quadrature Practice
#---------------------------------------------------

function practice_quadrature()
    println("=== Question 3a: Quadrature Practice ===")
    
    # Define standard normal distribution
    d = Normal(0, 1)
    
    # Get quadrature nodes and weights for 7 grid points
    nodes, weights = lgwt(7, -4, 4)
    
    # TODO: Verify integral of density equals 1
    # integral_density = sum(weights .* pdf.(d, nodes))
    println("∫φ(x)dx = [FILL IN] (should be ≈ 1)")
    
    # TODO: Verify expectation equals 0  
    # expectation = sum(weights .* nodes .* pdf.(d, nodes))
    println("∫xφ(x)dx = [FILL IN] (should be ≈ 0)")
end

#---------------------------------------------------
# Question 3b: More Quadrature Practice
#---------------------------------------------------

function variance_quadrature()
    println("\n=== Question 3b: Variance using Quadrature ===")
    
    # Define N(0,2) distribution
    d = Normal(0, 2)
    σ = 2
    
    # TODO: Use quadrature to compute ∫x²f(x)dx with 7 points
    nodes7, weights7 = lgwt(7, -5*σ, 5*σ)
    # variance_7pts = sum(weights7 .* (nodes7.^2) .* pdf.(d, nodes7))
    
    # TODO: Use quadrature to compute ∫x²f(x)dx with 10 points
    nodes10, weights10 = lgwt(10, -5*σ, 5*σ)  
    # variance_10pts = sum(weights10 .* (nodes10.^2) .* pdf.(d, nodes10))
    
    println("Variance with 7 quadrature points: [FILL IN]")
    println("Variance with 10 quadrature points: [FILL IN]")
    println("True variance: $(σ^2)")
    
    # TODO: Comment on approximation quality
end

#---------------------------------------------------
# Question 3c: Monte Carlo Practice  
#---------------------------------------------------

function practice_monte_carlo()
    println("\n=== Question 3c: Monte Carlo Integration ===")
    
    σ = 2
    d = Normal(0, σ)
    a, b = -5*σ, 5*σ
    
    # TODO: Implement Monte Carlo integration function
    function mc_integrate(f, a, b, D)
        # ∫f(x)dx ≈ (b-a) * (1/D) * Σf(X_i) where X_i ~ U[a,b]
        # draws = rand(D) * (b - a) .+ a  # uniform draws on [a,b]
        # return (b - a) * mean(f.(draws))
    end
    
    # Test with different numbers of draws
    for D in [1000, 1000000]
        println("\nWith D = $D draws:")
        
        # TODO: Variance: ∫x²f(x)dx  
        # variance_mc = mc_integrate(x -> x^2 * pdf(d, x), a, b, D)
        println("MC Variance: [FILL IN] (true: $(σ^2))")
        
        # TODO: Mean: ∫xf(x)dx
        # mean_mc = mc_integrate(x -> x * pdf(d, x), a, b, D)  
        println("MC Mean: [FILL IN] (true: 0)")
        
        # TODO: Density integral: ∫f(x)dx
        # density_mc = mc_integrate(x -> pdf(d, x), a, b, D)
        println("MC Density integral: [FILL IN] (true: 1)")
    end
end

#---------------------------------------------------
# Question 4: Mixed Logit with Quadrature (DO NOT RUN!)
#---------------------------------------------------

function mixed_logit_quad(theta, X, Z, y, nodes, weights)
    # Extract parameters
    # theta = [alpha1, ..., alpha21, mu_gamma, sigma_gamma]
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    
    alpha = theta[1:(K*(J-1))]  # coefficients on X
    mu_gamma = theta[end-1]     # mean of gamma distribution
    sigma_gamma = theta[end]    # std dev of gamma distribution
    
    # Create choice indicator matrix
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end
    
    # Reshape alpha 
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
    
    # TODO: Implement mixed logit with quadrature
    # This involves integrating over the distribution of gamma
    
    # Initialize integrated probabilities
    T = promote_type(eltype(X), eltype(theta))
    P_integrated = zeros(T, N, J)
    
    # TODO: For each quadrature point r:
    # 1. Transform node: gamma_r = mu_gamma + sigma_gamma * nodes[r]
    # 2. Compute choice probabilities for this gamma_r (like regular logit)
    # 3. Weight by quadrature weight and normal density
    # 4. Add to integrated probabilities
    
    for r in eachindex(nodes)
        # gamma_r = mu_gamma + sigma_gamma * nodes[r]
        
        # Compute probabilities for this gamma_r
        # num_r = zeros(T, N, J)
        # for j = 1:J
        #     num_r[:,j] = exp.(X * bigAlpha[:,j] .+ gamma_r .* (Z[:,j] .- Z[:,J]))
        # end
        # dem_r = sum(num_r, dims=2)
        # P_r = num_r ./ dem_r
        
        # Weight and add to integrated probabilities
        # density_weight = weights[r] * pdf(Normal(0,1), nodes[r])
        # P_integrated .+= P_r * density_weight
    end
    
    # TODO: Compute log-likelihood  
    # loglike = -sum(bigY .* log.(P_integrated))
    
    return loglike
end

#---------------------------------------------------
# Question 5: Mixed Logit with Monte Carlo (DO NOT RUN!)
#---------------------------------------------------

function mixed_logit_mc(theta, X, Z, y, D)
    # Extract parameters (same as quadrature version)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    
    alpha = theta[1:(K*(J-1))]
    mu_gamma = theta[end-1]
    sigma_gamma = theta[end]
    
    # Create choice indicator matrix
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end
    
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
    
    # TODO: Implement mixed logit with Monte Carlo
    # Similar to quadrature but with random draws instead of nodes/weights
    
    T = promote_type(eltype(X), eltype(theta))
    P_integrated = zeros(T, N, J)
    
    # TODO: For d = 1 to D:
    # 1. Draw gamma_d from N(mu_gamma, sigma_gamma^2)
    # 2. Compute choice probabilities for this draw
    # 3. Add to running average (P_integrated += P_d / D)
    
    gamma_dist = Normal(mu_gamma, sigma_gamma)
    
    for d = 1:D
        # gamma_d = rand(gamma_dist)
        
        # Compute probabilities for this draw (same as regular logit)
        # num_d = zeros(T, N, J)  
        # for j = 1:J
        #     num_d[:,j] = exp.(X * bigAlpha[:,j] .+ gamma_d .* (Z[:,j] .- Z[:,J]))
        # end
        # dem_d = sum(num_d, dims=2)
        # P_d = num_d ./ dem_d
        
        # Add to running average
        # P_integrated .+= P_d / D
    end
    
    # TODO: Compute log-likelihood
    # loglike = -sum(bigY .* log.(P_integrated))
    
    return loglike
end

#---------------------------------------------------
# Optimization Functions
#---------------------------------------------------

function optimize_mlogit(X, Z, y)
    K = size(X, 2)
    J = length(unique(y))
    
    # Starting values: K*(J-1) alphas + 1 gamma
    # TODO: You might want to use estimates from PS3 as starting values
    startvals = [2*rand(K*(J-1)).-1; 0.1]
    
    # TODO: Use optimize() function with automatic differentiation
    # Hint: Use LBFGS() algorithm with autodiff = :forward
    result = optimize(theta -> mlogit_with_Z(theta, X, Z, y),
                     startvals, LBFGS(), 
                     Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true);
                     autodiff = :forward)
    
    return result.minimizer
end

function optimize_mixed_logit_quad(X, Z, y)
    K = size(X, 2)  
    J = length(unique(y))
    
    # Get quadrature nodes and weights
    nodes, weights = lgwt(7, -4, 4)
    
    # Starting values: K*(J-1) alphas + mu_gamma + sigma_gamma
    # TODO: Use regular logit estimates as starting values for alpha and gamma
    startvals = [2*rand(K*(J-1)).-1; 0.1; 1.0]  # last element is sigma_gamma
    
    # TODO: Set up optimization (DON'T ACTUALLY RUN - TOO SLOW!)
    # result = optimize(theta -> mixed_logit_quad(theta, X, Z, y, nodes, weights),
    #                  startvals, LBFGS(),
    #                  Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true);
    #                  autodiff = :forward)
    
    println("Mixed logit quadrature optimization setup complete (not executed)")
    return startvals  # Return starting values instead of running
end

function optimize_mixed_logit_mc(X, Z, y)
    K = size(X, 2)
    J = length(unique(y))
    
    D = 1000  # Number of Monte Carlo draws
    
    # Starting values: same as quadrature version
    startvals = [2*rand(K*(J-1)).-1; 0.1; 1.0]
    
    # TODO: Set up optimization (DON'T ACTUALLY RUN - TOO SLOW!)
    # result = optimize(theta -> mixed_logit_mc(theta, X, Z, y, D),
    #                  startvals, LBFGS(),
    #                  Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true);
    #                  autodiff = :forward)
    
    println("Mixed logit Monte Carlo optimization setup complete (not executed)")
    return startvals  # Return starting values instead of running
end

#---------------------------------------------------
# Question 6: Main Function
#---------------------------------------------------

function allwrap()
    println("=== Problem Set 4: Multinomial and Mixed Logit ===")
    
    # Load data
    df, X, Z, y = load_data()
    
    println("Data loaded successfully!")
    println("Sample size: ", size(X, 1))
    println("Number of covariates in X: ", size(X, 2))
    println("Number of alternatives: ", length(unique(y)))
    
    # Question 1: Estimate multinomial logit
    println("\n=== QUESTION 1: MULTINOMIAL LOGIT RESULTS ===")
    # TODO: Uncomment when ready to estimate
    # theta_hat_mle = optimize_mlogit(X, Z, y)
    # println("Estimates: ", theta_hat_mle)
    # alpha_hat = theta_hat_mle[1:end-1]
    # gamma_hat = theta_hat_mle[end]
    # println("γ̂ = ", gamma_hat)
    
    # Question 2: Interpret gamma
    println("\n=== QUESTION 2: INTERPRETATION ===")
    println("TODO: Compare γ̂ to PS3 results and interpret")
    
    # Question 3: Practice with quadrature and Monte Carlo
    practice_quadrature()
    variance_quadrature() 
    practice_monte_carlo()
    
    # Question 4: Mixed logit with quadrature (setup only)
    println("\n=== QUESTION 4: MIXED LOGIT QUADRATURE (SETUP) ===")
    optimize_mixed_logit_quad(X, Z, y)
    
    # Question 5: Mixed logit with Monte Carlo (setup only)  
    println("\n=== QUESTION 5: MIXED LOGIT MONTE CARLO (SETUP) ===")
    optimize_mixed_logit_mc(X, Z, y)
    
    println("\n=== ALL ANALYSES COMPLETE ===")
end

# TODO: Uncomment to run
# allwrap()

println("Starter code loaded successfully!")
println("Remember to:")
println("1. Fill in all TODO sections")
println("2. Test functions step by step")  
println("3. Don't run mixed logit estimations (too computationally intensive)")
println("4. Use automatic differentiation in optimization")
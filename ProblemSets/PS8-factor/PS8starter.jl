#=
Problem Set 8 - Starter Code
ECON 6343: Econometrics III
Factor Models and Dimension Reduction

Instructions: This starter code provides the basic structure for completing
the problem set. Fill in the missing parts marked with TODO comments.
The key concepts you'll need to implement are indicated, but the actual
implementation is left for you to work through.
=#

using Random, LinearAlgebra, Statistics, Distributions
using Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM
using MultivariateStats, FreqTables, ForwardDiff

# Set working directory
cd(@__DIR__)

# TODO: You may need to include an external quadrature file
# include("lgwt.jl")

#==================================================================================
# Question 1: Load data and estimate base regression
==================================================================================#

"""
    load_data(url::String)

Load the NLSY dataset from the given URL and return as a DataFrame.

# Arguments
- `url::String`: URL to the CSV file

# Returns
- DataFrame containing the NLSY data
"""
function load_data(url::String)
    # TODO: Use CSV.read() and HTTP.get() to load the data
    # Hint: CSV.read(HTTP.get(url).body, DataFrame)
    
end

"""
    estimate_base_regression(df::DataFrame)

Estimate the baseline wage regression without ASVAB scores.
Model: logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr

# Arguments
- `df::DataFrame`: The data containing all variables

# Returns
- Regression results from GLM.lm()
"""
function estimate_base_regression(df::DataFrame)
    # TODO: Use the @formula macro and lm() function to estimate the regression
    # Formula: logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr
    
end

#==================================================================================
# Question 2: Compute correlations among ASVAB scores
==================================================================================#

"""
    compute_asvab_correlations(df::DataFrame)

Compute the correlation matrix for the six ASVAB test scores.

# Arguments
- `df::DataFrame`: Data containing ASVAB scores in the last 6 columns

# Returns
- DataFrame with correlation matrix (6x6)

# Hint
- The ASVAB scores are in columns: asvabAR, asvabCS, asvabMK, asvabNO, asvabPC, asvabWK
- These are the last 6 columns of the DataFrame
- Use cor() function on the matrix of ASVAB scores
"""
function compute_asvab_correlations(df::DataFrame)
    # TODO: Extract the last 6 columns as a matrix
    # TODO: Compute correlation matrix using cor()
    # TODO: Return as a formatted DataFrame
    
end

#==================================================================================
# Question 3: Regression with all ASVAB scores
==================================================================================#

"""
    estimate_full_regression(df::DataFrame)

Estimate wage regression including all six ASVAB scores.
Model: logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + 
       asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK

# Arguments
- `df::DataFrame`: The data containing all variables

# Returns
- Regression results from GLM.lm()

# Question to consider:
- Given the correlations you computed, what problems might arise?
"""
function estimate_full_regression(df::DataFrame)
    # TODO: Estimate regression with all ASVAB scores included
    
end

#==================================================================================
# Question 4: PCA regression
==================================================================================#

"""
    estimate_pca_regression(df::DataFrame)

Perform PCA on ASVAB scores and include first principal component in regression.

# Arguments
- `df::DataFrame`: The data containing all variables

# Returns
- Regression results including the first principal component

# Key steps:
1. Extract ASVAB scores as a matrix
2. IMPORTANT: Transpose to J×N (MultivariateStats requires features × observations)
3. Fit PCA model with maxoutdim=1
4. Transform the data to get principal component scores
5. Add PC scores to DataFrame and run regression

# Hints:
- asvabMat = Matrix(df[:, end-5:end])'  (note the transpose!)
- M = fit(PCA, asvabMat; maxoutdim=1)
- asvabPCA = MultivariateStats.transform(M, asvabMat)
- asvabPCA will be 1×N, need to reshape for regression
"""
function estimate_pca_regression(df::DataFrame)
    # TODO: Extract ASVAB matrix and transpose to J×N
    
    # TODO: Fit PCA model with maxoutdim=1
    
    # TODO: Transform data to get principal component
    
    # TODO: Add PC to dataframe (careful with dimensions!)
    
    # TODO: Run regression with PC included
    
end

#==================================================================================
# Question 5: Factor Analysis regression
==================================================================================#

"""
    estimate_factor_regression(df::DataFrame)

Perform Factor Analysis on ASVAB scores and include first factor in regression.

# Arguments
- `df::DataFrame`: The data containing all variables

# Returns
- Regression results including the first factor

# Note:
- Syntax is nearly identical to PCA, just use FactorAnalysis instead
"""
function estimate_factor_regression(df::DataFrame)
    # TODO: Follow same steps as PCA but use FactorAnalysis
    
end

#==================================================================================
# Question 6: Full factor model with MLE
==================================================================================#

"""
    prepare_factor_matrices(df::DataFrame)

Prepare the data matrices needed for the factor model estimation.

# Arguments
- `df::DataFrame`: The data containing all variables

# Returns
- `X`: Covariates for wage equation (N×7: black, hispanic, female, schoolt, gradHS, grad4yr, constant)
- `y`: Log wage outcomes (N×1)
- `Xfac`: Covariates for measurement equations (N×4: black, hispanic, female, constant)
- `asvabs`: Matrix of all 6 ASVAB scores (N×6)
"""
function prepare_factor_matrices(df::DataFrame)
    # TODO: Create X matrix for wage equation (include constant at end)
    
    # TODO: Extract y (logwage)
    
    # TODO: Create Xfac matrix for measurement equations
    
    # TODO: Create asvabs matrix
    
    return X, y, Xfac, asvabs
end

"""
    factor_model(θ::Vector{T}, X::Matrix, Xfac::Matrix, Meas::Matrix, 
                 y::Vector, R::Integer) where T<:Real

Compute the negative log-likelihood for the factor model.

# Arguments
- `θ`: Parameter vector containing:
  * γ parameters (L×J matrix, vectorized): coefficients in measurement equations
  * β parameters (K×1 vector): coefficients in wage equation  
  * α parameters (J+1 vector): factor loadings (J for measurements, 1 for wage)
  * σ parameters (J+1 vector): standard deviations (J for measurements, 1 for wage)
- `X`: Wage equation covariates (N×K)
- `Xfac`: Measurement equation covariates (N×L)
- `Meas`: ASVAB test scores (N×J)
- `y`: Log wages (N×1)
- `R`: Number of quadrature points

# Returns
- Negative log-likelihood value (scalar)

# Model Structure:
## Measurement equations (for each j=1,...,J):
   asvab_j = Xfac*γ_j + α_j*ξ + ε_j,  ε_j ~ N(0, σ_j²)

## Wage equation:
   logwage = X*β + α_{J+1}*ξ + ε,  ε ~ N(0, σ_{J+1}²)

## Latent factor:
   ξ ~ N(0,1)

# Likelihood for person i:
   L_i = ∫ [∏_j φ((M_ij - Xfac_i*γ_j - α_j*ξ)/σ_j) / σ_j] 
          × [φ((y_i - X_i*β - α_{J+1}*ξ)/σ_{J+1}) / σ_{J+1}]
          × φ(ξ) dξ

# Key Steps:
1. Unpack θ into γ, β, α, σ parameters
2. Set up Gauss-Legendre quadrature nodes and weights
3. For each quadrature point:
   a. Compute likelihood contribution from each measurement equation
   b. Compute likelihood contribution from wage equation
   c. Weight by quadrature weight and ξ density
4. Sum log-likelihoods across all observations
5. Return negative for minimization
"""
function factor_model(θ::Vector{T}, X::Matrix, Xfac::Matrix, Meas::Matrix, 
                     y::Vector, R::Integer) where T<:Real
    
    # Get dimensions
    K = size(X, 2)      # Number of covariates in wage equation
    L = size(Xfac, 2)   # Number of covariates in measurement equations
    J = size(Meas, 2)   # Number of ASVAB tests
    N = length(y)       # Number of observations
    
    # TODO: Unpack parameters from θ vector
    # γ should be L×J matrix (reshape from θ[1:J*L])
    # β should be K×1 vector
    # α should be (J+1)×1 vector (factor loadings)
    # σ should be (J+1)×1 vector (standard deviations)
    
    # TODO: Get quadrature nodes (ξ) and weights (ω) using lgwt()
    # Recommended: lgwt(R, -5, 5) for standard normal integration
    
    # Initialize likelihood storage
    like = zeros(T, N)
    
    # TODO: Loop over quadrature points
    for r in 1:R
        # TODO: For each ASVAB test j, compute:
        # - Residuals: M_ij - Xfac*γ_j - α_j*ξ_r
        # - Standardize by σ_j
        # - Evaluate normal PDF
        # - Store in Mlike[:,j]
        
        # TODO: For wage equation, compute:
        # - Residuals: y_i - X*β - α_{J+1}*ξ_r  
        # - Standardize by σ_{J+1}
        # - Evaluate normal PDF
        # - Store in Ylike
        
        # TODO: Update likelihood by:
        # like += ω[r] * (product of all Mlike) * Ylike * φ(ξ_r)
    end
    
    # TODO: Return negative sum of log-likelihoods
    return -sum(log.(like))
end

"""
    run_estimation(df::DataFrame, start_vals::Vector)

Run the full MLE estimation procedure for the factor model.

# Arguments
- `df::DataFrame`: The data
- `start_vals::Vector`: Starting values for optimization

# Returns
- `θ̂`: Estimated parameters
- `se`: Standard errors
- `loglike`: Log-likelihood at optimum

# Steps:
1. Prepare data matrices
2. Set up TwiceDifferentiable objective (for Hessian-based optimization)
3. Optimize using Newton method with line search
4. Compute standard errors from inverse Hessian
"""
function run_estimation(df::DataFrame, start_vals::Vector)
    # TODO: Prepare data matrices
    
    # TODO: Create TwiceDifferentiable objective
    # Use autodiff = :forward for ForwardDiff
    
    # TODO: Optimize using Newton method
    # Optim.Options: set g_tol, iterations, show_trace as appropriate
    
    # TODO: Compute Hessian and standard errors
    
    # TODO: Return estimates, standard errors, and log-likelihood
    
end

"""
    format_results(θ::Vector, se::Vector, loglike::Real, asvabs::Matrix)

Format estimation results into a readable DataFrame.

# Arguments
- `θ`: Estimated parameters
- `se`: Standard errors
- `loglike`: Log-likelihood value
- `asvabs`: ASVAB matrix (used to get J)

# Returns
- DataFrame with columns: equation, variable, coefficient, std_error
"""
function format_results(θ::Vector, se::Vector, loglike::Real, asvabs::Matrix)
    # TODO: Create vectors for equation names and variable names
    # Structure should match the parameter ordering in θ
    
    # TODO: Combine into DataFrame with appropriate column names
    
end

#==================================================================================
# Main execution function
==================================================================================#

"""
    main()

Execute the complete analysis workflow for Problem Set 8.

This function runs through all questions sequentially:
1. Load data and run base regression
2. Compute ASVAB correlations
3. Run full regression with all ASVABs
4. Run PCA regression
5. Run Factor Analysis regression
6. Estimate full factor model via MLE
"""
function main()
    # Data URL
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
    
    println("="^80)
    println("Problem Set 8: Factor Models and Dimension Reduction")
    println("="^80)
    
    # Load data
    println("\nLoading data...")
    df = load_data(url)
    println("Data loaded successfully. Dimensions: ", size(df))
    
    # Question 1
    println("\n" * "="^80)
    println("Question 1: Base Regression (without ASVAB)")
    println("="^80)
    # TODO: Call estimate_base_regression() and display results
    
    # Question 2
    println("\n" * "="^80)
    println("Question 2: ASVAB Correlations")
    println("="^80)
    # TODO: Call compute_asvab_correlations() and display results
    println("Consider: Are these correlations high? What might this imply?")
    
    # Question 3
    println("\n" * "="^80)
    println("Question 3: Full Regression (with all ASVAB)")
    println("="^80)
    # TODO: Call estimate_full_regression() and display results
    println("Consider: How do results compare to Question 1? Any concerns?")
    
    # Question 4
    println("\n" * "="^80)
    println("Question 4: PCA Regression")
    println("="^80)
    # TODO: Call estimate_pca_regression() and display results
    
    # Question 5
    println("\n" * "="^80)
    println("Question 5: Factor Analysis Regression")
    println("="^80)
    # TODO: Call estimate_factor_regression() and display results
    
    # Question 6
    println("\n" * "="^80)
    println("Question 6: Full Factor Model (MLE)")
    println("="^80)
    
    # TODO: Prepare starting values
    # Hint: Can use OLS estimates as starting values:
    # - For γ: regress each ASVAB on Xfac
    # - For β: regress wage on X
    # - For α: start with random small values or zeros
    # - For σ: start with sample standard deviations or 0.5
    
    println("\nEstimating full factor model...")
    # TODO: Call run_estimation() with starting values
    
    # TODO: Format and display results
    
    println("\n" * "="^80)
    println("Analysis complete!")
    println("="^80)
end

#==================================================================================
# Execute main function
==================================================================================#

# Uncomment when ready to run:
# main()

#==================================================================================
# Question 7: Unit Tests
==================================================================================#

# TODO: Create unit tests for your functions
# Suggested tests:
# - Test data loading returns proper DataFrame
# - Test correlation matrix has correct dimensions
# - Test likelihood function returns finite value
# - Test parameter unpacking and dimensions
# - Test optimization converges

# Example structure:
# using Test
# @testset "Factor Model Tests" begin
#     @test <condition>
# end
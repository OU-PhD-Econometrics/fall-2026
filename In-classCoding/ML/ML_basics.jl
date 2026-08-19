# =============================================================================
# Machine Learning: Bias-Variance Tradeoff with Basis Function Approximation
# ECON 6343 - Structural Econometrics
# =============================================================================

using Random, LinearAlgebra, Statistics, Optim
using DataFrames, Plots, GLM, StatsPlots
using MLJ, MLJLinearModels, Tables

Random.seed!(1234)

cd(@__DIR__)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

"""
Generate Chebyshev polynomial basis functions up to degree `max_degree`
"""
function chebyshev_basis(x::Vector{Float64}, max_degree::Int)
    n = length(x)
    T = zeros(n, max_degree + 1)
    T[:, 1] .= 1.0  # T_0(x) = 1
    if max_degree >= 1
        T[:, 2] = x  # T_1(x) = x
    end
    for j in 3:(max_degree + 1)
        T[:, j] = 2 .* x .* T[:, j-1] .- T[:, j-2]  # T_n(x) = 2xT_{n-1}(x) - T_{n-2}(x)
    end
    return T
end

"""
Generate polynomial basis for multivariate input
"""
function polynomial_basis(X::Matrix{Float64}, max_degree::Int)
    n, d = size(X)
    bases = []
    
    # Univariate polynomials for each feature
    for j in 1:d
        for deg in 1:max_degree
            push!(bases, X[:, j].^deg)
        end
    end
    
    # Interactions (if multiple features)
    if d > 1
        for j1 in 1:d, j2 in (j1+1):d
            for deg1 in 1:max_degree, deg2 in 1:max_degree
                if deg1 + deg2 <= max_degree
                    push!(bases, X[:, j1].^deg1 .* X[:, j2].^deg2)
                end
            end
        end
    end
    
    return hcat(bases...)
end

"""
Generate data with sparse nonlinear structure
"""
function generate_data(n::Int, n_features::Int, n_basis::Int, n_active::Int; noise_std=0.5)
    # Generate features
    X = randn(n, n_features)
    
    # Create basis expansion
    Φ = polynomial_basis(X, 4)  # Up to 4th degree polynomials
    
    # Truncate/pad to exactly n_basis functions
    if size(Φ, 2) > n_basis
        Φ = Φ[:, 1:n_basis]
    elseif size(Φ, 2) < n_basis
        # Pad with random linear combinations
        extra = n_basis - size(Φ, 2)
        Φ = hcat(Φ, randn(n, extra))
    end
    
    # True coefficients: only first n_active are non-zero
    β_true = zeros(n_basis)
    β_true[1:n_active] = randn(n_active) .* 6.0
    
    # Generate outcome
    y = Φ * β_true + randn(n) * noise_std
    
    return X, Φ, y, β_true
end

"""
Split data into train/val/test
"""
function split_data(Φ, y; train_frac=0.6, val_frac=0.2)
    n = length(y)
    n_train = Int(floor(train_frac * n))
    n_val = Int(floor(val_frac * n))
    
    idx = Random.shuffle(1:n)
    train_idx = idx[1:n_train]
    val_idx = idx[n_train+1:n_train+n_val]
    test_idx = idx[n_train+n_val+1:end]
    
    return (
        train = (Φ[train_idx, :], y[train_idx]),
        val = (Φ[val_idx, :], y[val_idx]),
        test = (Φ[test_idx, :], y[test_idx]),
        idx = (train=train_idx, val=val_idx, test=test_idx)
    )
end

"""
Fit OLS regression
"""
function fit_ols(X::Matrix{Float64}, y::Vector{Float64})
    X_design = hcat(ones(size(X, 1)), X)
    β = (X_design' * X_design) \ (X_design' * y)
    return β
end

"""
Predict with OLS coefficients
"""
function predict_ols(X::Matrix{Float64}, β::Vector{Float64})
    X_design = hcat(ones(size(X, 1)), X)
    return X_design * β
end

"""
Fit Ridge regression (L2 penalty)
"""
function fit_ridge(X::Matrix{Float64}, y::Vector{Float64}, λ::Float64)
    n, p = size(X)
    X_design = hcat(ones(n), X)
    # Don't penalize intercept
    penalty = diagm([0.0; fill(λ, p)])
    β = (X_design' * X_design + penalty) \ (X_design' * y)
    return β
end

"""
Compute MSE
"""
function compute_mse(y_true::Vector{Float64}, y_pred::Vector{Float64})
    return mean((y_true .- y_pred).^2)
end

"""
Cross-validate to find optimal λ for Ridge
"""
function cv_ridge(Φ_train, y_train, Φ_val, y_val, λ_grid)
    mse_train = zeros(length(λ_grid))
    mse_val = zeros(length(λ_grid))
    
    for (i, λ) in enumerate(λ_grid)
        β = fit_ridge(Φ_train, y_train, λ)
        mse_train[i] = compute_mse(y_train, predict_ols(Φ_train, β))
        mse_val[i] = compute_mse(y_val, predict_ols(Φ_val, β))
    end
    
    best_idx = argmin(mse_val)
    return λ_grid[best_idx], mse_train, mse_val, best_idx
end

"""
Cross-validate LASSO
"""
function cv_lasso(Φ_train, y_train, Φ_val, y_val, λ_grid)
    mse_train = zeros(length(λ_grid))
    mse_val = zeros(length(λ_grid))
    n_nonzero = zeros(Int, length(λ_grid))
    
    Φ_train_tbl = Tables.table(Φ_train)
    Φ_val_tbl = Tables.table(Φ_val)
    
    for (i, λ) in enumerate(λ_grid)
        lasso = LassoRegressor(lambda=λ, solver=ProxGrad(max_iter=10_000))
        mach = machine(lasso, Φ_train_tbl, y_train)
        fit!(mach, verbosity=0)
        
        y_train_pred = MLJ.predict(mach, Φ_train_tbl)
        y_val_pred = MLJ.predict(mach, Φ_val_tbl)
        
        mse_train[i] = compute_mse(y_train, y_train_pred)
        mse_val[i] = compute_mse(y_val, y_val_pred)
        
        β = fitted_params(mach).coefs
        n_nonzero[i] = sum(abs.(last.(β)) .> 1e-6)
    end
    
    best_idx = argmin(mse_val)
    return λ_grid[best_idx], mse_train, mse_val, n_nonzero, best_idx
end

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

function main()
    println("="^70)
    println("BIAS-VARIANCE TRADEOFF: BASIS FUNCTION APPROXIMATION")
    println("="^70)
    
    # Parameters
    n_total = 300
    n_features = 2      # Raw features
    n_basis = 120       # Basis functions (p)
    n_active = 5       # True non-zero coefficients
    
    println("\n📊 Data Generation:")
    println("   Total observations: ", n_total)
    println("   Raw features: ", n_features)
    println("   Basis functions (p): ", n_basis)
    println("   True active coefficients: ", n_active)
    println("   True sparse model: Only ", n_active, " of ", n_basis, " basis terms matter!")
    
    # Generate data
    X, Φ, y, β_true = generate_data(n_total, n_features, n_basis, n_active, noise_std=23)
    println(β_true)
    
    # Split data
    data = split_data(Φ, y)
    Φ_train, y_train = data.train
    Φ_val, y_val = data.val
    Φ_test, y_test = data.test
    
    # Standardize based on training set
    μ = mean(Φ_train, dims=1)
    σ = std(Φ_train, dims=1)
    Φ_train = (Φ_train .- μ) ./ σ
    Φ_val = (Φ_val .- μ) ./ σ
    Φ_test = (Φ_test .- μ) ./ σ
    
    # Standardize y, too
    y_mean = mean(y_train)
    y_std = std(y_train)
    y_train = (y_train .- y_mean) ./ y_std
    y_val = (y_val .- y_mean) ./ y_std
    y_test = (y_test .- y_mean) ./ y_std
    
    n_train = length(y_train)
    println("\n📦 Data Split:")
    println("   Training: ", n_train, " obs")
    println("   Validation: ", length(y_val), " obs")
    println("   Test: ", length(y_test), " obs")
    println("   p/n ratio: ", round(n_basis/n_train, digits=2), 
            " (high-dimensional problem!)")
    
    # =========================================================================
    # OLS (will overfit!)
    # =========================================================================
    println("\n" * "="^70)
    println("ORDINARY LEAST SQUARES")
    println("="^70)
    
    β_ols = fit_ols(Φ_train, y_train)
    println(β_ols)
    
    mse_train_ols = compute_mse(y_train, predict_ols(Φ_train, β_ols))
    mse_val_ols = compute_mse(y_val, predict_ols(Φ_val, β_ols))
    
    println("\n   Training MSE: ", round(mse_train_ols, digits=4))
    println("   Validation MSE: ", round(mse_val_ols, digits=4))
    println("   Val/Train ratio: ", round(mse_val_ols/mse_train_ols, digits=2), "x")
    println("\n   ⚠️  Large ratio indicates OVERFITTING!")


    #β_ols_cheating = fit_ols(Φ_train[:,1:5], y_train)
    #println(β_ols_cheating)
    
    
    # =========================================================================
    # Ridge Regression
    # =========================================================================
    println("\n" * "="^70)
    println("RIDGE REGRESSION (L2 PENALTY)")
    println("="^70)
    
    λ_grid = exp.(range(log(1e-6), log(1000), length=500))
    λ_best_ridge, mse_train_ridge, mse_val_ridge, best_idx_ridge = 
        cv_ridge(Φ_train, y_train, Φ_val, y_val, λ_grid)
    #println([mse_val_ridge log.(λ_grid)])
    #println(λ_grid[1:5])
    test_ols = fit_ridge(Φ_train, y_train, 0.0)
    println("Ridge with λ=0 matches OLS: ", 
            all(abs.(test_ols .- β_ols) .< 1e-8) ? "✅" : "❌")
    
    println("\n   Best λ: ", round(λ_best_ridge, digits=4))
    println("   Training MSE: ", round(mse_train_ridge[best_idx_ridge], digits=4))
    println("   Validation MSE: ", round(mse_val_ridge[best_idx_ridge], digits=4))
    println("   Val/Train ratio: ", 
            round(mse_val_ridge[best_idx_ridge]/mse_train_ridge[best_idx_ridge], digits=2), "x")
    println("   Improvement over OLS: ", 
            round(100*(1 - mse_val_ridge[best_idx_ridge]/mse_val_ols), digits=1), "%")
    
    # =========================================================================
    # LASSO Regression
    # =========================================================================
    println("\n" * "="^70)
    println("LASSO REGRESSION (L1 PENALTY + SPARSITY)")
    println("="^70)
    
    λ_grid_lasso = exp.(range(log(1e-6), log(1000), length=500))
    λ_best_lasso, mse_train_lasso, mse_val_lasso, n_nonzero, best_idx_lasso = 
        cv_lasso(Φ_train, y_train, Φ_val, y_val, λ_grid_lasso)
    
    println("\n   Best λ: ", round(λ_best_lasso, digits=4))
    println("   Training MSE: ", round(mse_train_lasso[best_idx_lasso], digits=4))
    println("   Validation MSE: ", round(mse_val_lasso[best_idx_lasso], digits=4))
    println("   Val/Train ratio: ", 
            round(mse_val_lasso[best_idx_lasso]/mse_train_lasso[best_idx_lasso], digits=2), "x")
    println("   Selected features: ", n_nonzero[best_idx_lasso], " of ", n_basis)
    println("   True active features: ", n_active)
    println("   Improvement over OLS: ", 
            round(100*(1 - mse_val_lasso[best_idx_lasso]/mse_val_ols), digits=1), "%")
    
    # =========================================================================
    # Test Set Evaluation
    # =========================================================================
    println("\n" * "="^70)
    println("FINAL TEST SET EVALUATION")
    println("="^70)
    
    # Refit with best hyperparameters
    β_ridge_final = fit_ridge(Φ_train, y_train, λ_best_ridge)
    
    lasso_final = LassoRegressor(lambda=λ_best_lasso)
    mach_final = machine(lasso_final, Tables.table(Φ_train), y_train)
    fit!(mach_final, verbosity=0)
    
    # Test predictions
    mse_test_ols = compute_mse(y_test, predict_ols(Φ_test, β_ols))
    mse_test_ridge = compute_mse(y_test, predict_ols(Φ_test, β_ridge_final))
    mse_test_lasso = compute_mse(y_test, MLJ.predict(mach_final, Tables.table(Φ_test)))
    
    println("\n   OLS Test MSE:    ", round(mse_test_ols, digits=4))
    println("   Ridge Test MSE:  ", round(mse_test_ridge, digits=4))
    println("   LASSO Test MSE:  ", round(mse_test_lasso, digits=4))
    
    # Results table
    results = DataFrame(
        Method = ["OLS", "Ridge", "LASSO"],
        Train_MSE = round.([mse_train_ols, 
                           mse_train_ridge[best_idx_ridge], 
                           mse_train_lasso[best_idx_lasso]], digits=4),
        Val_MSE = round.([mse_val_ols, 
                         mse_val_ridge[best_idx_ridge], 
                         mse_val_lasso[best_idx_lasso]], digits=4),
        Test_MSE = round.([mse_test_ols, mse_test_ridge, mse_test_lasso], digits=4),
        Overfit_Ratio = round.([mse_val_ols/mse_train_ols,
                               mse_val_ridge[best_idx_ridge]/mse_train_ridge[best_idx_ridge],
                               mse_val_lasso[best_idx_lasso]/mse_train_lasso[best_idx_lasso]], 
                              digits=2)
    )
    
    println("\n", results)
    
    # =========================================================================
    # Visualizations
    # =========================================================================
    println("\n" * "="^70)
    println("GENERATING VISUALIZATIONS")
    println("="^70)
    
    # Plot 1: Ridge regularization path
    p1 = plot(log.(λ_grid), [mse_train_ridge mse_val_ridge],
         label=["Training MSE" "Validation MSE"],
         xlabel="log(λ)", ylabel="MSE",
         title="Ridge: Bias-Variance Tradeoff",
         linewidth=2, legend=:topleft,
         size=(800, 500))
    vline!(p1, [log(λ_best_ridge)], label="Optimal λ", 
           linestyle=:dash, linewidth=2, color=:red)
    savefig(p1, "ridge_regularization_path.png")
    println("   ✅ Saved: ridge_regularization_path.png")
    
    # Plot 2: LASSO sparsity path
    p2 = plot(n_nonzero, [mse_train_lasso mse_val_lasso],
         label=["Training MSE" "Validation MSE"],
         xlabel="Number of Selected Features", ylabel="MSE",
         title="LASSO: Model Complexity vs Performance",
         linewidth=2, legend=:topright,
         size=(800, 500))
    vline!(p2, [n_nonzero[best_idx_lasso]], label="Optimal", 
           linestyle=:dash, linewidth=2, color=:red)
    vline!(p2, [n_active], label="True # Active", 
           linestyle=:dot, linewidth=2, color=:green)
    savefig(p2, "lasso_sparsity_path.png")
    println("   ✅ Saved: lasso_sparsity_path.png")
    
    # Plot 3: Method comparison
    p3 = groupedbar([results.Train_MSE results.Val_MSE results.Test_MSE],
               bar_position=:dodge,
               label=["Train" "Validation" "Test"],
               xlabel="Method", ylabel="MSE",
               title="Performance Comparison Across Data Splits",
               xticks=(1:3, results.Method),
               legend=:topright,
               size=(800, 500))
    savefig(p3, "method_comparison.png")
    println("   ✅ Saved: method_comparison.png")
    
    # =========================================================================
    # Key Takeaways
    # =========================================================================
    println("\n" * "="^70)
    println("🎯 KEY TAKEAWAYS")
    println("="^70)
    
    println("\n1. HIGH-DIMENSIONAL PROBLEM (p ≈ n)")
    println("   • ", n_basis, " basis functions vs ", n_train, " training observations")
    println("   • OLS overfits severely (Val/Train ratio = ", 
            round(mse_val_ols/mse_train_ols, digits=2), ")")
    
    println("\n2. BIAS-VARIANCE TRADEOFF")
    println("   • No regularization (λ=0): Fits training noise → high variance")
    println("   • Too much regularization (λ→∞): Ignores signal → high bias")
    println("   • Optimal λ balances both")
    
    println("\n3. REGULARIZATION BENEFITS")
    println("   • Ridge: Shrinks all coefficients → ", 
            round(100*(1 - mse_val_ridge[best_idx_ridge]/mse_val_ols), digits=1), 
            "% improvement")
    println("   • LASSO: Sparse selection (", n_nonzero[best_idx_lasso], 
            " of ", n_basis, " features) → ",
            round(100*(1 - mse_val_lasso[best_idx_lasso]/mse_val_ols), digits=1),
            "% improvement")
    
    println("\n4. VALIDATION SET IMPORTANCE")
    println("   • Chooses λ without peeking at test set")
    println("   • Test MSE similar to validation MSE → good generalization")
    
    println("\n5. SPARSITY & INTERPRETABILITY")
    println("   • True model: ", n_active, " active terms")
    println("   • LASSO selected: ", n_nonzero[best_idx_lasso], " terms")
    println("   • Automatic feature selection aids interpretation")
    
    println("\n" * "="^70)
    println("✨ Analysis Complete!")
    println("="^70)
    
    return results
end

# Run analysis
results = main()

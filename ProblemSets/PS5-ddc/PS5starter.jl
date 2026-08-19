using Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

# Read in function to create state transitions for dynamic model
include("create_grids.jl")

################################################################################
# PART 1: DATA LOADING AND PREPARATION (Pre-implemented)
################################################################################

"""
    load_static_data()

Load and reshape data for static estimation (Questions 1-2).
Returns a long-format DataFrame ready for GLM estimation.

This is pre-implemented so you can focus on the modeling rather than 
data wrangling. The key is understanding that we're converting from:
  - Wide format: Y1, Y2, ..., Y20 (20 columns per bus)
  - Long format: One row per bus-time observation (N×T rows total)
"""
function load_static_data()
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    
    # Create bus id variable
    df = @transform(df, :bus_id = 1:size(df,1))
    
    #---------------------------------------------------
    # Reshape from wide to long (done twice because 
    # stack() requires doing one variable at a time)
    #---------------------------------------------------
    
    # First reshape the decision variable (Y1-Y20)
    dfy = @select(df, :bus_id, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10,
                      :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20,
                      :RouteUsage, :Branded)
    dfy_long = DataFrames.stack(dfy, Not([:bus_id, :RouteUsage, :Branded]))
    rename!(dfy_long, :value => :Y)
    dfy_long = @transform(dfy_long, :time = kron(collect(1:20), ones(size(df,1))))
    select!(dfy_long, Not(:variable))
    
    # Next reshape the odometer variable (Odo1-Odo20)
    dfx = @select(df, :bus_id, :Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, 
                      :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, :Odo14, :Odo15,
                      :Odo16, :Odo17, :Odo18, :Odo19, :Odo20)
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, :time = kron(collect(1:20), ones(size(df,1))))
    select!(dfx_long, Not(:variable))
    
    # Join reshaped dataframes back together
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id, :time])
    sort!(df_long, [:bus_id, :time])
    
    return df_long
end

"""
    load_dynamic_data()

Load and prepare data for dynamic estimation (Question 3+).
Returns a named tuple with all data structures needed for estimation.

The named tuple approach keeps our code clean by bundling related data together.
This is a professional coding practice that makes function signatures manageable.
"""
function load_dynamic_data()
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    
    # Extract matrices and vectors
    Y = Matrix(df[:, r"^Y\d+"])       # Decision variables (N × T)
    X = Matrix(df[:, r"^Odo\d+"])     # Odometer readings (N × T)
    Xstate = Matrix(df[:, r"^Xst"])   # Discretized mileage state (N × T)
    Zstate = Vector(df[:, :Zst])      # Discretized route usage state (N × 1)
    B = Vector(df[:, :Branded])       # Brand indicator (N × 1)
    
    N, T = size(Y)
    
    # Create state grids
    zval, zbin, xval, xbin, xtran = create_grids()
    
    # Bundle everything into a named tuple for clean code
    # This is much better than passing 10+ separate arguments!
    return (
        # Data
        Y = Y,
        X = X,
        B = B,
        Xstate = Xstate,
        Zstate = Zstate,
        # Dimensions
        N = N,
        T = T,
        # State space
        xval = xval,
        xbin = xbin,
        zbin = zbin,
        xtran = xtran,
        # Parameters
        β = 0.9
    )
end

################################################################################
# PART 2: STATIC ESTIMATION (Question 2)
################################################################################

function estimate_static_model(df_long)
    # TODO: Estimate the logit model
    # θ̂_static = glm(...)
    
    println("TODO: Implement static logit estimation")
    return nothing
end

################################################################################
# PART 3: DYNAMIC ESTIMATION - FUTURE VALUE COMPUTATION (Question 3c)
################################################################################

"""
    compute_future_value!(FV, θ, d)

Compute future value function for all states using backward recursion.
Uses the @views and @inbounds macros for performance (cuts runtime ~50%).

This is the heart of dynamic programming! We solve Bellman's equation:
  V(s,t) = max{v₀(s,t), v₁(s,t)} + ε
where v₁ - v₀ = θ₀ + θ₁·x + θ₂·b + β·E[V(s',t+1)|s,d=1] - β·E[V(s',t+1)|s,d=0]

Key insight: We compute V for ALL possible states (not just observed ones),
because we need E[V(s',t+1)] which requires knowing V at states we might 
transition to in the future.

Arguments:
- FV: Pre-allocated future value array (zbin×xbin, 2, T+1) - modified in place
- θ: Parameter vector [θ₀, θ₁, θ₂]  
- d: Named tuple with data and grids

Algorithm:
1. Initialize FV[:,:,T+1] = 0 (no future beyond last period)
2. Loop backwards from t=T to t=1:
   - For each state (z, x, b):
     * Compute v₁(z,x,b,t) = flow utility + β·E[V(s',t+1)|continue]
     * Compute v₀(z,x,b,t) = 0 + β·E[V(s',t+1)|replace]  
     * Store FV(z,x,b,t) = β·log(exp(v₀) + exp(v₁))
3. This FV array is then used in likelihood computation

TODO: Implement the four nested loops for backward recursion.
"""
@views @inbounds function compute_future_value!(FV, θ, d)
    # FV is already initialized to zeros in the calling function
    # FV[T+1] stays at zero (terminal condition)
    
    # TODO: Loop backward over time
    # for t in d.T:-1:1
    #     
    #     TODO: Loop over brand states
    #     for b in 0:1
    #         
    #         TODO: Loop over route usage states (permanent characteristic)
    #         for z in 1:d.zbin
    #             
    #             TODO: Loop over mileage states  
    #             for x in 1:d.xbin
    #                 
    #                 # Calculate row index in transition matrix
    #                 # This indexes the joint state (z,x)
    #                 row = # TODO: x + (z-1)*d.xbin
    #                 
    #                 # Compute v₁: value of continuing with current engine
    #                 # Flow utility: θ₀ + θ₁·mileage + θ₂·brand
    #                 # Future value: E[V_{t+1} | don't replace, current state (z,x)]
    #                 # 
    #                 # The expectation is: xtran[row,:]' · FV[z's rows, b+1, t+1]
    #                 # where "z's rows" means all x values for the given z
    #                 v1 = # TODO: θ[1] + θ[2]*d.xval[x] + θ[3]*b + d.xtran[row,:]⋅FV[(z-1)*d.xbin+1:z*d.xbin, b+1, t+1]
    #                 
    #                 # Compute v₀: value of replacing engine
    #                 # Flow utility: normalized to 0 (replacement is the reference)
    #                 # Future value: E[V_{t+1} | replace, current z]
    #                 # 
    #                 # Key difference: after replacement, mileage resets to 0
    #                 # So we use row = 1 + (z-1)*xbin (first mileage bin)
    #                 v0 = # TODO: d.xtran[1+(z-1)*d.xbin,:]⋅FV[(z-1)*d.xbin+1:z*d.xbin, b+1, t+1]
    #                 
    #                 # Store future value using log-sum-exp formula
    #                 # FV(t) = β·log(exp(v₀) + exp(v₁))
    #                 # This is the expected value before the agent observes ε
    #                 FV[row, b+1, t] = # TODO: d.β * log(exp(v1) + exp(v0))
    #                 
    #             end
    #         end
    #     end
    # end
    
    println("TODO: Implement backward recursion loops")
    return nothing
end

################################################################################
# PART 4: DYNAMIC ESTIMATION - LOG LIKELIHOOD (Question 3d)
################################################################################

"""
    log_likelihood_dynamic(θ, d)

Compute the log likelihood for the dynamic model.
Uses the @views and @inbounds macros for performance.

Now we use the pre-computed FV array, but only for the states we actually 
observe in the data. The likelihood is:

  ℓ(θ) = Σᵢ Σₜ [yᵢₜ·log(P₁ᵢₜ) + (1-yᵢₜ)·log(P₀ᵢₜ)]

where P₁ᵢₜ = exp(v₁-v₀)/(1+exp(v₁-v₀)) is the probability of continuing.

The conditional value difference is:
  v₁ - v₀ = θ₀ + θ₁·xᵢₜ + θ₂·bᵢ + β·[E[V|continue] - E[V|replace]]

Key insight: The future value difference can be computed as:
  E[V|continue] - E[V|replace] = (xtran[row1,:] - xtran[row0,:])' · FV[...]
This works because both use the same FV array for s', just different 
transition probabilities.

Arguments:
- θ: Parameter vector [θ₀, θ₁, θ₂]
- d: Named tuple with data and pre-computed FV

Returns: 
- Scalar log likelihood value (we return negative for minimization)

TODO: Implement the likelihood computation using observed states.
"""
@views @inbounds function log_likelihood_dynamic(θ, d)
    # First, compute future values for all states given current θ
    FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
    compute_future_value!(FV, θ, d)
    
    # Now compute likelihood using only observed states
    loglike = 0.0
    
    # TODO: Loop over individuals
    # for i in 1:d.N
    #     
    #     # Pre-compute the row index for replacement (mileage = 0)
    #     # This is constant across time for each individual
    #     row0 = # TODO: (d.Zstate[i] - 1) * d.xbin + 1
    #     
    #     TODO: Loop over time periods
    #     for t in 1:d.T
    #         
    #         # Get row index for current state (observed mileage and route usage)
    #         row1 = # TODO: d.Xstate[i,t] + (d.Zstate[i] - 1) * d.xbin
    #         
    #         # Compute conditional value difference: v₁ - v₀
    #         # 
    #         # Part 1: Flow utility difference
    #         #   v₁ has: θ₀ + θ₁·x + θ₂·b
    #         #   v₀ has: 0 (normalized)
    #         #   So difference is just the flow utility
    #         flow_diff = # TODO: θ[1] + θ[2]*d.X[i,t] + θ[3]*d.B[i]
    #         
    #         # Part 2: Expected future value difference
    #         #   E[V|continue] uses transition probs from current state (row1)
    #         #   E[V|replace] uses transition probs from mileage=0 (row0)
    #         #   Both look at same FV array, just different transition weights
    #         # 
    #         # This is the elegant part: we DIFFERENCE the transition matrices!
    #         ev_diff = # TODO: (d.xtran[row1,:] .- d.xtran[row0,:])⋅FV[row0:row0+d.xbin-1, d.B[i]+1, t+1]
    #         
    #         # Total conditional value difference
    #         v_diff = # TODO: flow_diff + d.β * ev_diff
    #         
    #         # Compute choice probabilities using logit formula
    #         # P(Y=1) = exp(v_diff) / (1 + exp(v_diff))
    #         # P(Y=0) = 1 / (1 + exp(v_diff))
    #         
    #         # Add to log likelihood
    #         # Efficient form: ℓ = Σ[Y·v_diff - log(1 + exp(v_diff))]
    #         # This is numerically stable and avoids computing P₀ and P₁ separately
    #         loglike += # TODO: (d.Y[i,t] == 1) * v_diff - log(1 + exp(v_diff))
    #         
    #     end
    # end
    
    println("TODO: Implement likelihood computation")
    
    # Return NEGATIVE log likelihood (Optim minimizes)
    return -loglike
end

################################################################################
# PART 5: OPTIMIZATION WRAPPER (Question 3e-h)
################################################################################

"""
    estimate_dynamic_model(d; θ_start=nothing)

Estimate the dynamic discrete choice model using MLE.

This function:
1. Sets up starting values (use static estimates if available)
2. Defines objective function (negative log likelihood)
3. Calls Optim to minimize
4. Returns results

Arguments:
- d: Named tuple with data
- θ_start: Starting values for optimization (if nothing, uses random)

Returns:
- Optimization result object from Optim

TODO: Set up and run the optimization.
"""
function estimate_dynamic_model(d; θ_start=nothing)
    println("="^70)
    println("DYNAMIC MODEL ESTIMATION")
    println("="^70)
    
    # Set starting values
    if isnothing(θ_start)
        θ_start = rand(3)  # Random start if nothing provided
        println("\nUsing random starting values: ", θ_start)
    else
        println("\nUsing provided starting values: ", θ_start)
    end
    
    # Define objective function (just passes θ to likelihood)
    # The data is "captured" by the closure
    objective = θ -> log_likelihood_dynamic(θ, d)
    
    # TODO: Time the likelihood evaluation (good practice!)
    # println("\nTiming likelihood evaluation...")
    # @time objective(θ_start)
    # @time objective(θ_start)
    
    # TODO: Run optimization
    # println("\nOptimizing (this may take several minutes)...")
    # result = optimize(objective, θ_start, LBFGS(), 
    #                   Optim.Options(g_tol=1e-5, 
    #                                iterations=100_000, 
    #                                show_trace=true,
    #                                show_every=10))
    
    println("\nTODO: Implement optimization call")
    
    # TODO: Display results
    # println("\n" * "="^70)
    # println("RESULTS")
    # println("="^70)
    # println("Parameter estimates:")
    # println("  θ₀ (constant):     ", round(result.minimizer[1], digits=4))
    # println("  θ₁ (mileage):      ", round(result.minimizer[2], digits=4))
    # println("  θ₂ (brand):        ", round(result.minimizer[3], digits=4))
    # println("\nLog likelihood:      ", round(-result.minimum, digits=2))
    # println("Converged:           ", Optim.converged(result))
    # println("Iterations:          ", Optim.iterations(result))
    # println("="^70)
    
    return nothing
end

################################################################################
# MAIN EXECUTION WRAPPER (Question 3f)
################################################################################

"""
    main()

Main wrapper function that runs all estimation procedures.
Wrapping everything in a function (rather than running in global scope) 
is a Julia best practice for performance.
"""
function main()
    println("\n" * "="^70)
    println("PROBLEM SET 5: BUS ENGINE REPLACEMENT MODEL")
    println("="^70)
    
    #---------------------------------------------------------------------------
    # Part 1: Static Estimation (Questions 1-2)
    #---------------------------------------------------------------------------
    println("\n" * "-"^70)
    println("PART 1: STATIC (MYOPIC) ESTIMATION")
    println("-"^70)
    
    # Load data
    println("\nLoading and reshaping data...")
    df_long = load_static_data()
    println("Observations: ", nrow(df_long))
    println("First few rows:")
    println(first(df_long, 5))
    
    # Estimate static model
    println("\nEstimating static logit model...")
    θ̂_static = estimate_static_model(df_long)
    
    # TODO: Uncomment when static estimation is implemented
    # if !isnothing(θ̂_static)
    #     println("\nStatic estimates:")
    #     println(θ̂_static)
    #     
    #     # Extract coefficients to use as starting values for dynamic model
    #     θ_start_dynamic = coef(θ̂_static)
    # else
    #     θ_start_dynamic = nothing
    # end
    
    θ_start_dynamic = [2.0, -0.15, 1.0]  # Placeholder starting values
    
    #---------------------------------------------------------------------------
    # Part 2: Dynamic Estimation (Question 3)
    #---------------------------------------------------------------------------
    println("\n" * "-"^70)
    println("PART 2: DYNAMIC ESTIMATION")  
    println("-"^70)
    
    # Load data
    println("\nLoading dynamic data...")
    d = load_dynamic_data()
    println("Buses (N): ", d.N)
    println("Time periods (T): ", d.T)
    println("Mileage bins: ", d.xbin)
    println("Route usage bins: ", d.zbin)
    println("Total state space size: ", d.xbin * d.zbin)
    println("Discount factor (β): ", d.β)
    
    # Estimate dynamic model
    println("\nSetting up dynamic estimation...")
    estimate_dynamic_model(d, θ_start=θ_start_dynamic)
    
    println("\n" * "="^70)
    println("END OF PROBLEM SET")
    println("="^70)
    
    return nothing
end

################################################################################
# NOTES FOR STUDENTS
################################################################################
#
# WHAT YOU NEED TO UNDERSTAND:
#
# 1. THE ECONOMIC MODEL:
#    - Static: Zurcher is myopic (β=0), only cares about today
#    - Dynamic: Zurcher is forward-looking (β=0.9), considers future costs
#    - Decision: Replace engine (d=0) vs. Keep running (d=1)
#    - State: (mileage, route usage, brand, time)
#
# 2. BACKWARD RECURSION (the key algorithm):
#    - Start at t=T: No future, so FV[T+1] = 0
#    - Work backward: FV[t] depends on FV[t+1]
#    - Compute for ALL states (not just observed ones)
#    - This gives us E[V_{t+1}] for any state we might visit
#
# 3. WHY WE COMPUTE FV FOR ALL STATES:
#    - When at state (z=5, x=10, t=5), we need E[V(s',6)]
#    - The expectation is over all possible s' we might transition to
#    - We weight each possible s' by transition probability xtran
#    - So we need V(s',6) for ALL s' in the state space
#
# 4. THE TRANSITION MATRIX TRICK:
#    - xtran[row,:] gives P(x'|x, z, don't replace)
#    - xtran[row0,:] gives P(x'|0, z, replace)  
#    - In likelihood: (xtran[row1,:] - xtran[row0,:])' · FV
#    - This efficiently computes E[V|continue] - E[V|replace]
#
# 5. PERFORMANCE TIPS:
#    - @views avoids array copies
#    - @inbounds skips bounds checking
#    - Named tuples avoid long argument lists
#    - Together these cut runtime ~50%
#
# 6. DEBUGGING TIPS:
#    - Start with small test: compute FV for one θ value
#    - Check FV dimensions: (zbin*xbin, 2, T+1)
#    - Verify FV[T+1] = 0 (terminal condition)
#    - Check that v1 > v0 when mileage is low (replacing is costly)
#    - Print likelihood value - should be large negative number
#    - If likelihood = 0, you have a bug in the loops!
#
# Good luck! The hard part is understanding the algorithm, not coding it.
################################################################################
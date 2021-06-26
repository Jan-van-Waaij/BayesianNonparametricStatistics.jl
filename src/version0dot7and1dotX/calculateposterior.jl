# calculateposterior.jl

"""
    calculatedependentfaberschauderfunctions(higestlevel::Int64)

Internal function, not exported.

Calculates which Faber-Schauder functions (ψ_i,ψ_j) have essential overlapping
support, i ≤ j. All other combinations (ψ_i, ψ_j), i ≤ j, have essential
nonoverlapping support and hence int_0^T ψ_i(X_t)ψ_j(X_t)dt=0, which we use in
calculategirsanovmatrix.

Returns a triple (lengthvectors, rowindices, columnindices) where
```julia
lengthvectors == length(rowindices) == length(columnindices),
```
and for i in 1:lengthvectors (ψ_rowindices[i],ψ_columnindices[i]) have
essentially overlapping support. For all i, rowindices[i] ≤ columnindices[i].
"""
function calculatedependentfaberschauderfunctions(higestlevel::Int64)
    lengthvectors = (higestlevel+1)*2^(higestlevel+1)+1 # Correct.
    rowindices = Vector{Int}(undef, lengthvectors)
    columnindices = Vector{Int}(undef, lengthvectors)
    # Ψ_1 is dependent with all basis functions, including itself. 
    rowindices[1:2^(higestlevel+1)] .= 1
    columnindices[1:2^(higestlevel+1)] = 1:2^(higestlevel+1)
    index = 2^(higestlevel+1) + 1 # point where we are in constructing rowindices and columnindices
    # psi_{j,k}=psi_{2^j+k} is dependent with
    # psi_{j+d,(k-1)2^d+1},...,psi_{j+d, k2^d}, d\ge0 (which includes itself).
    for jone in 0:higestlevel
        twotothepowerjone = 2^jone
        for kone in 1:2^jone
            ione = twotothepowerjone+kone # other index system
            for  jtwo in jone:higestlevel
                d = jtwo - jone
                twotothepowerjtwo = 2^jtwo
                for ktwo in (kone-1)*2^d+1:kone*2^d
                    rowindices[index] = ione
                    columnindices[index] = twotothepowerjtwo+ktwo
                    index += 1
                end
            end
        end
    end
    @assert lengthvectors + 1 == index 
    return (lengthvectors, rowindices, columnindices)
end

# calculategirsanovmatrixelement methods:

"""
    calculategirsanovmatrixelement(ψ1Xt, ψ2Xt, σ, Δt)
    calculategirsanovmatrixelement(samplevalueindices1, samplevalueindices2, ψ1Xt, ψ2Xt, σ, Δt)

Internal function, not exported!

Where ψ1Xt, ψ2Xt are arrays, and σ and Δt are either arrays or numbers.

Let ψ_a and ψ_b denote two basis elements.
calculategirsanovmatrixelement calculates the (a,b) element of the Girsanov
matrix defined by int_0^T ψ_a(X_t)ψ_b(X_t)/(σ^2(X_t)) dt, where ψ_a is given
by ψ1Xt (already evaluated in ψ_a) and ψ_b by ψ2Xt (already evaluated in ψ_b).

samplevalueindices1[i] == false should correspond to Ψ1Xt[i] == 0.0.
Similar for samplevalueindices2 and Ψ2Xt. 
"""
calculategirsanovmatrixelement(ψ1Xt, ψ2Xt, σXt, Δt) = sum(Δt .* ψ1Xt .* ψ2Xt ./ (σXt.^2))
#test written
calculategirsanovmatrixelement(ψ1Xt, ψ2Xt, σXt, Δt::Number) = Δt * sum(ψ1Xt .* ψ2Xt ./ (σXt.^2))
#test written
calculategirsanovmatrixelement(ψ1Xt, ψ2Xt, σXt::Number, Δt) = sum(Δt .* ψ1Xt .* ψ2Xt) / (σXt^2)
#test written
calculategirsanovmatrixelement(ψ1Xt, ψ2Xt, σXt::Number, Δt::Number) = Δt * sum(ψ1Xt .* ψ2Xt) / (σXt^2)
#test written
function calculategirsanovmatrixelement(samplevalueindices1, samplevalueindices2, ψ1Xt, ψ2Xt, σXt, Δt)
    samplevalueindices = samplevalueindices1 .& samplevalueindices2    
    filteredψ1Xt = ψ1Xt[samplevalueindices]
    filteredψ2Xt = ψ2Xt[samplevalueindices]
    filteredσXt = σXt[samplevalueindices]
    filteredΔt = Δt[samplevalueindices]
    return sum(filteredΔt .* filteredψ1Xt .* filteredψ2Xt ./ (filteredσXt.^2))
end
function calculategirsanovmatrixelement(samplevalueindices1, samplevalueindices2, ψ1Xt, ψ2Xt, σXt, Δt::Number)
    samplevalueindices = samplevalueindices1 .& samplevalueindices2
    filteredψ1Xt = ψ1Xt[samplevalueindices]
    filteredψ2Xt = ψ2Xt[samplevalueindices]
    filteredσXt = σXt[samplevalueindices]
    return Δt * sum(filteredψ1Xt .* filteredψ2Xt ./
        (filteredσXt.^2))
end
function calculategirsanovmatrixelement(samplevalueindices1, samplevalueindices2, ψ1Xt, ψ2Xt, σXt::Number, Δt)
    samplevalueindices = samplevalueindices1 .& samplevalueindices2
    filteredΔt = Δt[samplevalueindices]
    filteredψ1Xt = ψ1Xt[samplevalueindices]
    filteredψ2Xt = ψ2Xt[samplevalueindices]
    return sum(filteredΔt .* filteredψ1Xt .* filteredψ2Xt) / (σXt^2)
end
function calculategirsanovmatrixelement(samplevalueindices1, samplevalueindices2, ψ1Xt, ψ2Xt, σXt::Number, Δt::Number)
    samplevalueindices = samplevalueindices1 .& samplevalueindices2
    filteredψ1Xt = ψ1Xt[samplevalueindices]
    filteredψ2Xt = ψ2Xt[samplevalueindices]
    return Δt * sum(filteredψ1Xt .* filteredψ2Xt) / (σXt^2)
end

"""
    calculateΔt(timeinterval::AbstractRange) = step(timeinterval)
    calculateΔt(timeinterval) = timeinterval[2:end] - timeinterval[1:end-1]

Internal function, not exported!

Calculates Δt. It returns a number when timeinterval is a range object 
and an array of increments of the timeinterval vector otherwise.
"""
calculateΔt(timeinterval::AbstractRange) = step(timeinterval)
calculateΔt(timeinterval) = timeinterval[2:end] - timeinterval[1:end-1]
# Test written.

"""
    calculategirsanovmatrix(sizesquarematrix, timeinterval, ψXt, σXt)
    calculategirsanovmatrix(Π, samplevalueindices, timeinterval, ψXt, σXt)

Internal function, not exported!

Calculates the Girsanov matrix. In the case of a FaberSchauderExpansionWithGaussianCoefficients-object 
it will make use of the sparsity structure of the Faber-Schauder basis.
"""
function calculategirsanovmatrix(sizesquarematrix, timeinterval, ψXt, σXt)
    girsanovmatrix = Matrix{Float64}(undef, sizesquarematrix, sizesquarematrix)
    Δt = calculateΔt(timeinterval)
    for k in 1:sizesquarematrix
        for l in k:sizesquarematrix
            girsanovmatrix[k,l] = calculategirsanovmatrixelement(
                ψXt[k], ψXt[l], σXt, Δt)
        end
    end
    return Symmetric(girsanovmatrix)
end
function calculategirsanovmatrix(Π, samplevalueindices, timeinterval, ψXt, σXt)
    lengthvectors, rowindices, columnindices = calculatedependentfaberschauderfunctions(Π.higestlevel)
    d = length(Π)
    V = Vector{Float64}(undef, lengthvectors)
    Δt = calculateΔt(timeinterval)
    for i in 1:lengthvectors
        V[i] = calculategirsanovmatrixelement(samplevalueindices[rowindices[i]], 
            samplevalueindices[columnindices[i]], ψXt[rowindices[i]], ψXt[columnindices[i]], σXt, Δt)
    end
    return Symmetric(sparse(rowindices, columnindices, V, d, d))
end

"""
    calculategirsanovvectorelement(ΔXt, ψXt, σXt)
    calculategirsanovvectorelement(ΔXt, ψXt, σXt::Number)

    calculategirsanovvectorelement(samplevalueindices, ΔXt, ψXt, σ::Number)

Internal function, used by calculategirsanovvector, not exported!

Calculates the kth Girsanov vector element int_0^T ψ_k(X_t)/(σ^2(X_t)) dX_t 
with the Euler-Maruyama method where ψ_k is the kth basis function, and T is the end time.
"""
calculategirsanovvectorelement(ΔXt, ψXt, σXt) = sum(ΔXt .* ψXt ./ (σXt.^2))
#Test written.
calculategirsanovvectorelement(ΔXt, ψXt, σXt::Number) = sum(ΔXt .* ψXt) / (σXt^2)
#Test written.
function calculategirsanovvectorelement(samplevalueindices, ΔXt, ψXt, σXt)
    filteredΔXt = ΔXt[samplevalueindices]
    filterψXt = ψXt[samplevalueindices]
    filteredσXt = σXt[samplevalueindices]
    return sum(filteredΔXt .* filterψXt ./ (filteredσXt.^2))
end
#Test written.
function calculategirsanovvectorelement(samplevalueindices, ΔXt, ψXt, σXt::Number)
    filteredΔXt = ΔXt[samplevalueindices]
    filteredψXt = ψXt[samplevalueindices]
    return sum(filteredΔXt .* filteredψXt) / (σXt^2)
end
#Test written.
"""
    calculategirsanovvector(lengthvector, samplevalues, ψXt, σXt)
    calculategirsanovvector(lengthvector, samplevalueindices, samplevalues, ψXt, σXt)

Internal function, not exported!

Calculates the Girsanov vector.
"""
function calculategirsanovvector(lengthvector, samplevalues, ψXt, σXt)
    ΔXt = samplevalues[2:end] - samplevalues[1:end-1]
    return [calculategirsanovvectorelement(ΔXt, ψXt[k], σXt) for k in
        1:lengthvector]
end
# Test written.
function calculategirsanovvector(lengthvector, samplevalueindices, samplevalues, ψXt, σXt)
    ΔXt = samplevalues[2:end] - samplevalues[1:end-1]
    return [calculategirsanovvectorelement(samplevalueindices[k], ΔXt, ψXt[k],
        σXt) for k in 1:lengthvector]
end
# Test written.

"""
    calculateσXt(σ::Number, v) = σ
    calculateσXt(σ, v) = σ.(v)

Internal function, not exported!. 

Returns σ when it is a number, and if σ is a function, it evaluates v in σ. 
"""
calculateσXt(σ::Number, v) = σ
calculateσXt(σ, v) = σ.(v)
#Tests written.

"""
    calculateposterior(Π, X, model::SDEModel)
    calculateposterior(Π, X, σ)
    calculateposterior(Π::FaberSchauderExpansionWithGaussianCoefficients, X, σ)

Calculates the posterior distribution Π(⋅∣X) and returns a FaberSchauderExpansionWithGaussianCoefficients
object when Π is a FaberSchauderExpansionWithGaussianCoefficients-object. Otherwise, it returns a GaussianProcess-object. 
Uses model to determine σ.

# Examples

##Example with Faber-Schauder expansion.
```julia
model = SDEModel(1.0, 0.0, 10000.0, 0.1)
sde = SDE(x->sinpi(2*x), model)
X = rand(sde)
Π = FaberSchauderExpansionWithGaussianCoefficients([2.0^j for j in 0:5])
postΠ = calculateposterior(Π, X, model)
```
##Example with Fourier expansion.
```julia
using Distributions
α = 0.5
model = SDEModel(1.0, 0.0, 10000.0, 0.1)
sde = SDE(x->sinpi(2*x), model)
X = rand(sde)
distribution = MvNormalCanon([k^(α+0.5) for k in 1:50])
Π = GaussianProcess([fourier(k) for k in 1:50], distribution)
postΠ = calculateposterior(Π, X, model)
```
"""
calculateposterior(Π, X, model::SDEModel) = calculateposterior(Π, X, model.σ)
function calculateposterior(Π, X, σ)
    σXt = calculateσXt(σ, X.samplevalues[1:end-1])
    ψXt = [f.(X.samplevalues[1:end-1]) for f in Π.basis]
    lengthΠ = length(Π)
    precisionprior = invcov(Π.distribution)
    girsanovvector = calculategirsanovvector(lengthΠ, X.samplevalues, ψXt, σXt)
    girsanovmatrix = calculategirsanovmatrix(lengthΠ, X.timeinterval, ψXt, σXt)
    precisionmatrixposterior = girsanovmatrix + precisionprior
    potentialposterior = girsanovvector + precisionprior * mean(Π.distribution)
    posteriordistributiononcoefficients = MvNormalCanon(potentialposterior, precisionmatrixposterior)
    return GaussianProcess(Π.basis, posteriordistributiononcoefficients)
end 
function calculateposterior(Π::FaberSchauderExpansionWithGaussianCoefficients, X, σ)
    samplevaluesmod1 = mod.(X.samplevalues[1:end-1], 1.0)
    σXt = calculateσXt(σ, X.samplevalues[1:end-1])
    ψXt = [ψ.(X.samplevalues[1:end-1]) for ψ in Π.basis]
    # You only need to use the values which are in the support of the function.
    samplevalueindices = [Π.leftboundssupport[i] .≤
        samplevaluesmod1 .≤ Π.rightboundssupport[i] for i in 1:length(Π)]
    girsanovvector = calculategirsanovvector(length(Π), samplevalueindices,
        X.samplevalues, ψXt, σXt)
    girsanovmatrix = calculategirsanovmatrix(Π, samplevalueindices,
        X.timeinterval, ψXt, σXt)
    precisionprior = invcov(Π.distribution)
    precisionmatrixposterior = girsanovmatrix + precisionprior
    potentialposterior = girsanovvector + precisionprior * mean(Π.distribution)
    posteriordistributiononcoefficients = MvNormalCanon(potentialposterior, precisionmatrixposterior)
    return FaberSchauderExpansionWithGaussianCoefficients(Π.higestlevel, posteriordistributiononcoefficients)
end

function calculateposteriorcoefficients(d::GaussianVector,
        X::SamplePath, model::SDEModel)::GaussianProcess
    σXt = calculateσXt(model.σ, X.samplevalues[1:end-1])
    ψXt = [f.(X.samplevalues[1:end-1]) for f in Π.basis]
    lengthΠ = length(d)
    covariancematrixprior = Matrix(cov(Π.distribution))
    precisionprior = inv(covariancematrixprior)
    girsanovvector = calculategirsanovvector(lengthΠ, X.samplevalues, ψXt, σXt)
    girsanovmatrix = calculategirsanovmatrix(lengthΠ, X.timeinterval, ψXt, σXt)
    precisionmatrixposterior = girsanovmatrix + precisionprior
    potentialposterior = girsanovvector + precisionprior * mean(Π.distribution)
    meanposterior =  covariancematrixprior * potentialposterior
    posteriordistribution = GaussianVector(meanposterior, 
        precisionmatrixposterior^(-0.5))
    return GaussianProcess(Π.basis, posteriordistribution)
end

function GibbsIter(siteration, A, B, meanGPPost, precGPPost, numberofbasisfunctions, X, ψXt, σXt)
    meanGPPost /= sqrt(siteration)
    precGPPost *= sqrt(siteration)
    girsanovvector = calculategirsanovvector(numberofbasisfunctions, X.samplevalues, ψXt, σXt)
    girsanovmatrix = calculategirsanovmatrix(numberofbasisfunctions, X.timeinterval, ψXt, σXt)
    precGPPost += Matrix(girsanovmatrix)
    # Lijn hieronder moet de rekenfout hersteld worden.
    potentialGPPost = girsanovmatrix + meanGPPost
    meanGPPost = precGPPost \ potentialGPPost
    distGPPost = GaussianVector(meanGPPost, sqrt(inv(precGPPost)))
    ϴ = rand(distGPPost)
    siteration = rand(Gamma(A+numberofbasisfunctions/2, B + 0.5 * (ϴ - meanGPPost)' * 
        potentialGPPost * (ϴ - meanGPPost)))
    return (siteration, meanGPPost, precGPPost, ϴ)
end 


function GibbsSampler(Π::S, A::Float64, B::Float64, X::SamplePath, model::SDEModel,
        numSamples::Int, S0 = 1.0, burnin::Int = 100, intercept::Int = 1) where {S<:AbstractGaussianProcess}
    meanGPPost = mean(Π.distribution)
    precGPPost = inv(Matrix(cov(Π.distribution)))
    numberofbasisfunctions = convert(Float64, length(Π))

    σXt = calculateσXt(model.σ, X.samplevalues[1:end-1])
    ψXt = [f.(X.samplevalues[1:end-1]) for f in Π.basis]

    siteration = S0

    for i in 1:burnin
        siteration, meanGPPost, precGPPost = GibbsIter(siteration, A, B, meanGPPost, 
            precGPPost, numberofbasisfunctions, X, ψXt, σXt)
    end

    samplePostS = Array{Float64}(undef, numSamples)
    samplePostϴ = Array{Float64}(undef, numSamples, numberofbasisfunctions)
    for i in 1:numSamples
        for j in 1:(intercept+1)
            siteration, meanGPPost, precGPPost, ϴ = GibbsIter(siteration, A, B, 
                meanGPPost, precGPPost, numberofbasisfunctions, X, ψXt, σXt)
        end
        samplePostS[i], samplePostϴ[i,:] = siteration, ϴ
    end
    return (samplePostS, samplePostϴ)
end 





